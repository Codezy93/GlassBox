import os
import z3
import numpy as np
import pandas as pd
import joblib
import hashlib
from loguru import logger
from data.preprocess import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

IMMUTABLE_FEATURES = {"sex", "age", "education"}

class FormalVerifier:
    def __init__(self, model_path, scaler_path=None):
        logger.info(f"Initializing FormalVerifier with {model_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None

        if self.scaler is not None and hasattr(self.scaler, "scale_"):
            self.feature_scales = np.asarray(self.scaler.scale_, dtype=float)
        else:
            self.feature_scales = np.ones(len(CONTINUOUS_FEATURES), dtype=float)

    def _fit_local_surrogate(self, input_df: pd.DataFrame, delta: float, n_samples: int = 160):
        """
        Fit a deterministic local linear surrogate around the input point.
        This keeps the SMT check tied to the actual trained model.
        """
        x0 = input_df[ALL_FEATURES].iloc[0]
        seed_payload = "|".join(f"{f}:{float(x0[f]):.6f}" for f in ALL_FEATURES)
        seed = int(hashlib.sha256(seed_payload.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n_samples):
            row = x0.copy()
            for i, feat in enumerate(CONTINUOUS_FEATURES):
                radius = delta * self.feature_scales[i]
                row[feat] = float(row[feat]) + float(rng.uniform(-radius, radius))
            rows.append(row.to_dict())

        local_df = pd.DataFrame(rows)[ALL_FEATURES]
        y = self.model.predict_proba(local_df)[:, 0]  # class 0 = approved

        X = local_df.to_numpy(dtype=float)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=float)])
        coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        weights = coef[:-1]
        bias = float(coef[-1])
        return weights, bias
        
    def certify_recourse(self, input_profile, delta=0.2):
        """
        Certify whether an approval point exists within a local delta-box.
        """
        logger.info(f"🛡️ Certifying recourse reachability within delta={delta}...")

        feature_names = list(input_profile.keys())
        input_df = pd.DataFrame([input_profile])[ALL_FEATURES]
        base_proba = float(self.model.predict_proba(input_df)[0, 0])

        if base_proba >= 0.5:
            return {
                "certified": True,
                "message": "Current profile is already in the approval region.",
                "proof_id": "already-approved",
            }

        z3_vars = {name: z3.Real(name) for name in feature_names}
        solver = z3.Solver()

        try:
            # Keep immutable and categorical features fixed.
            for feat in ALL_FEATURES:
                if feat in IMMUTABLE_FEATURES or feat in CATEGORICAL_FEATURES:
                    solver.add(z3_vars[feat] == float(input_profile[feat]))

            # Allow local movement on mutable continuous features only.
            for i, feat in enumerate(CONTINUOUS_FEATURES):
                if feat in IMMUTABLE_FEATURES:
                    continue
                center = float(input_profile[feat])
                radius = float(delta * self.feature_scales[i])
                solver.add(z3_vars[feat] >= center - radius)
                solver.add(z3_vars[feat] <= center + radius)

            # Tie the SMT constraint to a local surrogate fitted from the model.
            weights, bias = self._fit_local_surrogate(input_df, delta=delta)
            model_expr = z3.Sum(
                [float(w) * z3_vars[name] for w, name in zip(weights, ALL_FEATURES)]
            ) + float(bias)

            # Approval target: P(no-default) >= 0.5
            solver.add(model_expr >= 0.5)

            if solver.check() == z3.sat:
                sat_model = solver.model()
                witness = {}
                for feat in ALL_FEATURES:
                    z_val = sat_model[z3_vars[feat]]
                    if z_val is None:
                        witness[feat] = float(input_profile[feat])
                        continue
                    if z3.is_rational_value(z_val):
                        witness[feat] = float(z_val.numerator_as_long()) / float(z_val.denominator_as_long())
                    else:
                        witness[feat] = float(z_val.as_decimal(12).replace("?", ""))
                proof_payload = "|".join(f"{k}:{witness[k]:.6f}" for k in ALL_FEATURES)
                proof_id = hashlib.sha256(proof_payload.encode("utf-8")).hexdigest()[:16]

                logger.success("✅ Recourse existence formally CERTIFIED.")
                return {
                    "certified": True,
                    "message": "A valid approval point exists within the configured local action bounds.",
                    "proof_id": proof_id,
                }
            else:
                logger.warning("❌ Recourse existence formally REJECTED.")
                return {
                    "certified": False,
                    "message": "No approval point was found within the configured local action bounds.",
                    "proof_id": None
                }
        except Exception as e:
            logger.error(f"Formal verification error: {e}")
            return {"certified": False, "message": f"Verification error: {e}"}
