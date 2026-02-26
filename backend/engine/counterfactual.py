"""
Counterfactual Explanation Engine powered by Microsoft's DiCE library.

Generates diverse, constrained counterfactual explanations that tell users
exactly what they need to change to get a different outcome.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import dice_ml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocess import (  # noqa: E402
    ALL_FEATURES,
    CONTINUOUS_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
)

SAVED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "saved")

# ── Constraint definitions ───────────────────────────────────

# Features that CANNOT be changed (immutable traits)
IMMUTABLE_FEATURES = ["sex", "age", "education"]

# Features the optimizer IS allowed to vary
MUTABLE_FEATURES = [f for f in ALL_FEATURES if f not in IMMUTABLE_FEATURES]

# Actionability constraints: realistic ranges for mutable features
PERMITTED_RANGES = {
    "credit_limit": [10000, 500000],
    "marriage": [1, 3],
    "repayment_sep": [-1, 8],
    "repayment_aug": [-1, 8],
    "repayment_jul": [-1, 8],
    "repayment_jun": [-1, 8],
    "repayment_may": [-1, 8],
    "repayment_apr": [-1, 8],
    "bill_sep": [-10000, 400000],
    "bill_aug": [-10000, 400000],
    "bill_jul": [-10000, 400000],
    "bill_jun": [-10000, 400000],
    "bill_may": [-10000, 400000],
    "bill_apr": [-10000, 400000],
    "pay_sep": [0, 60000],
    "pay_aug": [0, 60000],
    "pay_jul": [0, 60000],
    "pay_jun": [0, 60000],
    "pay_may": [0, 60000],
    "pay_apr": [0, 60000],
}


class CounterfactualEngine:
    """
    Generates diverse counterfactual explanations with immutability and
    actionability constraints using DiCE.
    """

    def __init__(self):
        # Load EBM (used for counterfactuals — fast inference, interpretable)
        self.ebm = joblib.load(os.path.join(SAVED_DIR, "ebm_model.pkl"))
        self.scaler = joblib.load(os.path.join(SAVED_DIR, "scaler.pkl"))

        # Load raw training data for DiCE data object
        raw = joblib.load(os.path.join(SAVED_DIR, "raw_splits.pkl"))
        train_df = raw["X_train"].copy()
        train_df[TARGET] = raw["y_train"].astype(int)

        # DiCE data interface — treat ALL features as continuous
        # This avoids the categorical string/int mismatch entirely
        self.dice_data = dice_ml.Data(
            dataframe=train_df,
            continuous_features=[f for f in ALL_FEATURES],
            outcome_name=TARGET,
        )

        # DiCE model interface (sklearn-compatible)
        self.dice_model = dice_ml.Model(
            model=self.ebm,
            backend="sklearn",
        )

        # DiCE explainer — random is most reliable across feature types
        self.explainer = dice_ml.Dice(
            self.dice_data,
            self.dice_model,
            method="random",
        )

    def generate(
        self,
        input_data: dict,
        num_cfs: int = 4,
        desired_class: int = 0,  # 0 = no default (approved)
    ) -> list[dict]:
        """
        Generate diverse counterfactual explanations.

        Uses a two-pass strategy: first with full constraints, then a relaxed
        fallback if the first pass finds nothing.
        """
        # Build input DataFrame
        input_df = pd.DataFrame([input_data])[ALL_FEATURES]

        counterfactuals = []

        # --- Pass 1: with permitted_range constraints ---
        try:
            cf_result = self.explainer.generate_counterfactuals(
                input_df,
                total_CFs=num_cfs,
                desired_class="opposite",
                features_to_vary=MUTABLE_FEATURES,
                permitted_range=PERMITTED_RANGES,
            )
            counterfactuals = self._parse_results(cf_result, input_data)
        except Exception as e:
            print(f"DiCE pass 1 error: {e}")

        # --- Pass 2: fallback without permitted_range ---
        if len(counterfactuals) == 0:
            print("Pass 1 found 0 counterfactuals, trying relaxed fallback…")
            try:
                cf_result = self.explainer.generate_counterfactuals(
                    input_df,
                    total_CFs=num_cfs,
                    desired_class="opposite",
                    features_to_vary=MUTABLE_FEATURES,
                )
                counterfactuals = self._parse_results(cf_result, input_data)
            except Exception as e:
                print(f"DiCE pass 2 error: {e}")

        return counterfactuals

    def _parse_results(self, cf_result, input_data: dict) -> list[dict]:
        """Parse a DiCE result object into a clean list of dicts."""
        counterfactuals = []
        if (
            cf_result.cf_examples_list
            and cf_result.cf_examples_list[0].final_cfs_df is not None
        ):
            cf_df = cf_result.cf_examples_list[0].final_cfs_df

            for _, row in cf_df.iterrows():
                changes = {}
                for feat in MUTABLE_FEATURES:
                    original = input_data.get(feat)
                    suggested = row[feat]
                    if original is not None and not np.isclose(
                        float(original), float(suggested), atol=0.5
                    ):
                        changes[feat] = {
                            "original": float(original),
                            "suggested": float(suggested),
                        }

                # Get prediction probability for this counterfactual
                cf_feats = {f: row[f] for f in ALL_FEATURES}
                cf_input = pd.DataFrame([cf_feats])[ALL_FEATURES]
                
                # IMPORTANT: Scale for EBM prediction
                cf_input_scaled = cf_input.copy()
                X_cont_scaled = self.scaler.transform(cf_input[CONTINUOUS_FEATURES])
                cf_input_scaled[CONTINUOUS_FEATURES] = X_cont_scaled
                
                cf_proba = self.ebm.predict_proba(cf_input_scaled)[0]

                counterfactuals.append(
                    {
                        "changes": changes,
                        "probability_no_default": float(cf_proba[0]),
                        "probability_default": float(cf_proba[1]),
                        "full_profile": {f: float(row[f]) for f in ALL_FEATURES},
                    }
                )

        return counterfactuals

    def predict(self, input_data: dict) -> dict:
        """Quick prediction using the EBM (with scaling)."""
        input_df = pd.DataFrame([input_data])[ALL_FEATURES]
        
        # Scale continuous features
        input_df_scaled = input_df.copy()
        X_cont_scaled = self.scaler.transform(input_df[CONTINUOUS_FEATURES])
        input_df_scaled[CONTINUOUS_FEATURES] = X_cont_scaled
        
        proba = self.ebm.predict_proba(input_df_scaled)[0]
        pred = int(self.ebm.predict(input_df_scaled)[0])
        return {
            "prediction": pred,
            "probability_default": float(proba[1]),
            "probability_no_default": float(proba[0]),
            "label": "Default (Rejected)" if pred == 1 else "No Default (Approved)",
        }


# Expose constraint metadata for the frontend
def get_feature_metadata() -> list[dict]:
    """Return feature info for the frontend form and sliders."""
    raw = joblib.load(os.path.join(SAVED_DIR, "raw_splits.pkl"))
    X_train = raw["X_train"]

    metadata = []
    for feat in ALL_FEATURES:
        info = {
            "name": feat,
            "type": "continuous" if feat in CONTINUOUS_FEATURES else "categorical",
            "immutable": feat in IMMUTABLE_FEATURES,
            "min": float(X_train[feat].min()),
            "max": float(X_train[feat].max()),
            "mean": float(X_train[feat].mean()),
            "median": float(X_train[feat].median()),
        }

        if feat in CATEGORICAL_FEATURES:
            info["values"] = sorted(X_train[feat].unique().tolist())

        if feat in PERMITTED_RANGES:
            info["permitted_range"] = PERMITTED_RANGES[feat]

        # Human-readable labels for categorical features
        if feat == "sex":
            info["labels"] = {1: "Male", 2: "Female"}
        elif feat == "education":
            info["labels"] = {
                1: "Graduate School",
                2: "University",
                3: "High School",
                4: "Others",
            }
        elif feat == "marriage":
            info["labels"] = {1: "Married", 2: "Single", 3: "Others"}
        elif feat.startswith("repayment_"):
            info["labels"] = {
                -1: "Paid on time",
                0: "Revolving credit",
                1: "1 month delay",
                2: "2 months delay",
                3: "3 months delay",
                4: "4 months delay",
                5: "5 months delay",
                6: "6 months delay",
                7: "7 months delay",
                8: "8+ months delay",
            }

        metadata.append(info)

    return metadata
