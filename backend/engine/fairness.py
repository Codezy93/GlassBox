import joblib
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    selection_rate
)
from loguru import logger
from data.preprocess import CONTINUOUS_FEATURES

class FairnessAuditor:
    def __init__(self, model_path, data_path, scaler_path=None):
        logger.info(f"Initializing FairnessAuditor with {model_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        splits = joblib.load(data_path)
        self.X_test = splits['X_test']
        self.y_test = splits['y_test']
        
        # Pre-calculate predictions (on SCALED data) for faster auditing
        X_to_predict = self.X_test.copy()
        if self.scaler:
            X_cont_scaled = self.scaler.transform(self.X_test[CONTINUOUS_FEATURES])
            X_to_predict[CONTINUOUS_FEATURES] = X_cont_scaled
            
        self.y_pred = self.model.predict(X_to_predict)

    def audit_demographic(self, sensitive_feature):
        """
        Audit the model based on a sensitive feature (e.g., 'sex', 'education', 'marriage').
        Returns a dict of fairness metrics.
        """
        if sensitive_feature not in self.X_test.columns:
            logger.warning(f"Feature {sensitive_feature} not found in test data")
            return {}

        sensitive_col = self.X_test[sensitive_feature]
        
        # Demographic Parity Difference: |P(Y_pred=1 | group=A) - P(Y_pred=1 | group=B)|
        dp_diff = demographic_parity_difference(
            self.y_test, 
            self.y_pred, 
            sensitive_features=sensitive_col
        )
        
        # Demographic Parity Ratio: P(Y_pred=1 | group=A) / P(Y_pred=1 | group=B) (min/max)
        dp_ratio = demographic_parity_ratio(
            self.y_test, 
            self.y_pred, 
            sensitive_features=sensitive_col
        )
        
        # Per-group selection rates
        groups = sensitive_col.unique()
        group_rates = {}
        for g in groups:
            mask = (sensitive_col == g)
            rate = selection_rate(self.y_test[mask], self.y_pred[mask])
            group_rates[str(g)] = float(rate)

        return {
            "feature": sensitive_feature,
            "demographic_parity_difference": float(dp_diff),
            "demographic_parity_ratio": float(dp_ratio),
            "group_selection_rates": group_rates,
            "status": "PASS" if dp_diff < 0.1 else "WARNING" if dp_diff < 0.2 else "FAIL"
        }

    def get_comprehensive_audit(self):
        """
        Audit across all typical protected attributes.
        """
        attributes = ["sex", "education", "marriage", "age"]
        results = {}
        for attr in attributes:
            results[attr] = self.audit_demographic(attr)
        return results
