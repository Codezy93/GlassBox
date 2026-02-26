import joblib
import numpy as np
import pandas as pd
from loguru import logger

class ConformalEngine:
    def __init__(self, model_path, data_path):
        logger.info(f"Initializing ConformalEngine with {model_path}")
        self.model = joblib.load(model_path)
        splits = joblib.load(data_path)
        self.X_test = splits['X_test']
        self.y_test = splits['y_test']
        
        # We use the test set for calibration (Split Conformal)
        self.cal_scores = None
        self.calibrate()

    def calibrate(self):
        """
        Calculate non-conformity scores on the calibration set.
        For classification, a common score is 1 - P(Y_true).
        """
        logger.info("🧪 Calibrating conformal scores...")
        # Get probabilities for the true classes
        probs = self.model.predict_proba(self.X_test)
        
        # Non-conformity score: 1 - probability of the actual class
        # y_test is 0 or 1
        n = len(self.y_test)
        scores = np.zeros(n)
        for i in range(n):
            true_class = int(self.y_test[i])
            scores[i] = 1.0 - probs[i, true_class]
            
        self.cal_scores = np.sort(scores)
        logger.success(f"✅ Calibration complete with {n} samples.")

    def get_prediction_set(self, input_profile, alpha=0.05):
        """
        Returns the set of labels {0, 1} that contains the true label 
        with probability at least 1 - alpha.
        """
        if self.cal_scores is None:
            return [0, 1]

        # Convert dict to df
        df = pd.DataFrame([input_profile])
        probs = self.model.predict_proba(df)[0]
        
        # Find the quantile q_hat
        n = len(self.cal_scores)
        q_level = np.ceil((n + 1) * (1 - alpha)) / n
        if q_level > 1.0: q_level = 1.0
        
        q_hat = np.quantile(self.cal_scores, q_level, method='higher')
        
        # Inclusion rule: include label y if 1 - P(y) <= q_hat (i.e., P(y) >= 1 - q_hat)
        prediction_set = []
        for label in [0, 1]:
            score = 1.0 - probs[label]
            if score <= q_hat:
                prediction_set.append(label)
                
        return prediction_set

    def get_uncertainty_metrics(self, input_profile):
        """
        Returns confidence metrics including sets at various alpha levels.
        """
        return {
            "set_95": self.get_prediction_set(input_profile, alpha=0.05),
            "set_99": self.get_prediction_set(input_profile, alpha=0.01),
            "q_hat_95": float(np.quantile(self.cal_scores, 0.95)) if self.cal_scores is not None else 0
        }
