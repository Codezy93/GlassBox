import os
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from data.preprocess import CONTINUOUS_FEATURES, ALL_FEATURES

class RobustnessEngine:
    def __init__(self, model_path, scaler_path=None):
        logger.info(f"Initializing RobustnessEngine with {model_path}")
        self.model = joblib.load(model_path)
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        if self.scaler is not None and hasattr(self.scaler, "scale_"):
            self.feature_scales = np.asarray(self.scaler.scale_, dtype=float)
        else:
            self.feature_scales = np.ones(len(CONTINUOUS_FEATURES), dtype=float)

    def audit_robustness(self, input_profile: dict, epsilon: float = 0.05, n_perturbations: int = 50):
        """
        Calculates the model's sensitivity to small input perturbations.
        Returns a 'Robustness Score' and the most sensitive features.
        """
        logger.info(f"🛡️ Auditing Adversarial Robustness (eps={epsilon})...")
        
        input_df = pd.DataFrame([input_profile])[ALL_FEATURES]
        # Perturbations are continuous-valued; keep continuous columns in float dtype.
        input_df[CONTINUOUS_FEATURES] = input_df[CONTINUOUS_FEATURES].astype(float)
        X_cont_base = input_df[CONTINUOUS_FEATURES].iloc[0].to_numpy(dtype=float)
        base_prob = self.model.predict_proba(input_df)[0, 0]  # Prob of No Default
        base_pred = self.model.predict(input_df)[0]

        n_cont_features = len(CONTINUOUS_FEATURES)
        
        flips = 0
        
        for _ in range(n_perturbations):
            # Random perturbation within epsilon ball, scaled by feature std
            noise = np.random.uniform(-epsilon, epsilon, n_cont_features) * self.feature_scales
            X_cont_perturbed = X_cont_base + noise
            
            # Reconstruct full profile for prediction
            X_perturbed_full = input_df.copy()
            X_perturbed_full[CONTINUOUS_FEATURES] = [X_cont_perturbed]
            
            perturbed_pred = self.model.predict(X_perturbed_full)[0]
            
            if perturbed_pred != base_pred:
                flips += 1

        robustness_score = 1.0 - (flips / n_perturbations)

        # Per-feature local sensitivity: one-feature perturbation around base profile
        feature_rank = {}
        for i, name in enumerate(CONTINUOUS_FEATURES):
            step = epsilon * self.feature_scales[i]
            deltas = []
            for sign in (-1.0, 1.0):
                x_tmp = input_df.copy()
                x_tmp.at[0, name] = float(x_tmp.at[0, name]) + sign * step
                p_tmp = self.model.predict_proba(x_tmp)[0, 0]
                deltas.append(abs(float(p_tmp) - float(base_prob)))
            feature_rank[name] = float(np.mean(deltas))
        
        # Normalize rankings (with safety)
        max_sens = max(feature_rank.values()) if feature_rank and feature_rank.values() else 0.0
        if max_sens < 1e-9:
            max_sens = 1.0
        feature_rank = {k: v / max_sens for k, v in feature_rank.items()}

        logger.success(f"✅ Robustness Audit complete. Score: {robustness_score:.2%}")
        
        return {
            "robustness_score": float(robustness_score),
            "epsilon_radius": epsilon,
            "feature_sensitivity_rank": feature_rank,
            "n_perturbations": n_perturbations
        }
