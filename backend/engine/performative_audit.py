import numpy as np
import pandas as pd
import joblib
from loguru import logger
from data.preprocess import ALL_FEATURES, CONTINUOUS_FEATURES

class PerformativeAudit:
    def __init__(self, model_path, data_path, scaler_path=None):
        logger.info(f"Initializing PerformativeAudit with {model_path}")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        splits = joblib.load(data_path)
        self.X_test = splits['X_test']
        if not isinstance(self.X_test, pd.DataFrame):
            self.X_test = pd.DataFrame(self.X_test, columns=ALL_FEATURES)
        
        self.feature_names = self.X_test.columns.tolist()

    def simulate_strategic_response(self, adoption_rate=0.3):
        """
        Simulate distribution shift where a percentage of the population 
        adopts the model's recourse suggestions.
        """
        logger.info(f"🌀 Simulating Strategic Response (adoption={adoption_rate})...")
        
        # We simulate users shifting their features according to 'actionability'
        # For simplicity, we assume users increase their 'credit_limit' and 'pay_sep'
        # while decreasing their 'bill_sep'.
        
        X_shifted = self.X_test.copy()
        n_adopters = int(len(self.X_test) * adoption_rate)
        indices = np.random.choice(self.X_test.index, n_adopters, replace=False)
        
        # Strategic manipulation:
        # Cast to float to avoid dtype conflicts with *=
        for col in ['credit_limit', 'bill_sep', 'pay_sep']:
            X_shifted[col] = X_shifted[col].astype(float)
            
        X_shifted.loc[indices, 'credit_limit'] *= 1.10
        X_shifted.loc[indices, 'bill_sep'] *= 0.95
        X_shifted.loc[indices, 'pay_sep'] *= 1.10
        
        # Scale both for accurate prediction
        X_test_scaled = self.X_test.copy()
        X_shifted_scaled = X_shifted.copy()
        
        if self.scaler:
            X_test_scaled[CONTINUOUS_FEATURES] = self.scaler.transform(self.X_test[CONTINUOUS_FEATURES])
            X_shifted_scaled[CONTINUOUS_FEATURES] = self.scaler.transform(X_shifted[CONTINUOUS_FEATURES])
        
        # Calculate Decoupling Drift (Difference in error rates)
        orig_preds = self.model.predict(X_test_scaled)
        shifted_preds = self.model.predict(X_shifted_scaled)
        
        # How many users who would have defaulted are now predicted as 'No Default'?
        # (Assuming the underlying ground truth hasn't changed, this is 'strategic gaming')
        gaming_count = np.sum((orig_preds == 1) & (shifted_preds == 0))
        gaming_ratio = gaming_count / n_adopters if n_adopters > 0 else 0
        
        logger.success(f"✅ Performative Audit complete. Strategic Gaming Ratio: {gaming_ratio:.2%}")
        
        return {
            "strategic_gaming_ratio": float(gaming_ratio),
            "distribution_shift_score": float(np.mean(np.abs(shifted_preds - orig_preds))),
            "adoption_rate": adoption_rate
        }

    def get_gameability_report(self):
        """
        Expose which features are most sensitive to strategic manipulation.
        """
        # We perturb each feature and see how many predictions flip
        report = {}
        for feature in ['credit_limit', 'bill_sep', 'pay_sep']:
            X_temp = self.X_test.copy()
            X_temp[feature] *= 1.2 # 20% increase
            
            # Scale temp
            X_temp_predict = X_temp.copy()
            X_baseline_predict = self.X_test.copy()
            if self.scaler:
                X_temp_predict[CONTINUOUS_FEATURES] = self.scaler.transform(X_temp[CONTINUOUS_FEATURES])
                X_baseline_predict[CONTINUOUS_FEATURES] = self.scaler.transform(self.X_test[CONTINUOUS_FEATURES])
                
            new_preds = self.model.predict(X_temp_predict)
            flips = np.sum(new_preds != self.model.predict(X_baseline_predict))
            report[feature] = float(flips / len(self.X_test))
            
        return report
