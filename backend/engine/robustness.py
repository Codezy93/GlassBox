import numpy as np
import pandas as pd
import joblib
from loguru import logger
from data.preprocess import CONTINUOUS_FEATURES, ALL_FEATURES

class RobustnessEngine:
    def __init__(self, model_path, scaler_path):
        logger.info(f"Initializing RobustnessEngine with {model_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def audit_robustness(self, input_profile: dict, epsilon: float = 0.05, n_perturbations: int = 50):
        """
        Calculates the model's sensitivity to small input perturbations.
        Returns a 'Robustness Score' and the most sensitive features.
        """
        logger.info(f"🛡️ Auditing Adversarial Robustness (eps={epsilon})...")
        
        # Scale only continuous features
        input_df = pd.DataFrame([input_profile])[ALL_FEATURES]
        X_cont_scaled = self.scaler.transform(input_df[CONTINUOUS_FEATURES])[0]
        
        # Base prediction needs ALL_FEATURES (categorical raw, continuous scaled)
        X_base_full = input_df.copy()
        X_base_full[CONTINUOUS_FEATURES] = [X_cont_scaled]
        
        base_prob = self.model.predict_proba(X_base_full)[0, 0] # Prob of No Default
        base_pred = self.model.predict(X_base_full)[0]

        n_cont_features = len(CONTINUOUS_FEATURES)
        
        flips = 0
        sensitivities = {name: [] for name in ALL_FEATURES}
        
        for _ in range(n_perturbations):
            # Random perturbation within epsilon ball (scaled continuous space)
            noise = np.random.uniform(-epsilon, epsilon, n_cont_features)
            X_cont_perturbed_scaled = X_cont_scaled + noise
            
            # Reconstruct full profile for prediction
            X_perturbed_full = input_df.copy()
            X_perturbed_full[CONTINUOUS_FEATURES] = [X_cont_perturbed_scaled]
            
            perturbed_pred = self.model.predict(X_perturbed_full)[0]
            perturbed_probs = self.model.predict_proba(X_perturbed_full)[0]
            
            if perturbed_pred != base_pred:
                flips += 1
            
            # Record individual feature sensitivity
            for i, name in enumerate(CONTINUOUS_FEATURES):
                sensitivities[name].append(abs(perturbed_probs[0] - base_prob))

        robustness_score = 1.0 - (flips / n_perturbations)
        feature_rank = {name: float(np.mean(vals)) for name, vals in sensitivities.items() if vals}
        
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
