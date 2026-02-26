import joblib
import pandas as pd
import numpy as np
import shap
from loguru import logger
from data.preprocess import ALL_FEATURES

class InterpretabilityEngine:
    def __init__(self, model_path, data_path):
        logger.info(f"Initializing InterpretabilityEngine with {model_path}")
        self.model = joblib.load(model_path)
        
        # Load background data for SHAP
        try:
            splits = joblib.load(data_path)
            self.X_train = splits['X_train']
            self.feature_names = self.X_train.columns.tolist()
            
            # Using a small subset for background data to keep SHAP fast
            background_df = self.X_train.sample(min(100, len(self.X_train)))
            background = background_df.values
                
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            logger.info("SHAP KernelExplainer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None
            # Ensure feature names are still loaded even if explainer fails
            try:
                splits = joblib.load(data_path)
                self.feature_names = splits['X_train'].columns.tolist()
                self.X_train = splits['X_train']
            except:
                self.feature_names = ALL_FEATURES
                self.X_train = None

    def get_local_explanation(self, input_data):
        """
        Get SHAP values for a single profile.
        Returns a dict of feature_name -> shap_value
        """
        if self.explainer is None:
            return {}
            
        df = pd.DataFrame([input_data])[ALL_FEATURES]
        shap_values = self.explainer.shap_values(df.values)
        
        # If binary classification, shap_values might be a list (one per class)
        # result is typically [N_samples, N_features]
        if isinstance(shap_values, list):
            # For predict_proba, shap_values[1] is the probability of class 1 (default)
            vals = shap_values[1][0]
        else:
            vals = shap_values[0]
            
        importance = {name: float(val) for name, val in zip(self.feature_names, vals)}
        return importance

    def get_global_importance(self):
        """
        Returns average absolute SHAP values across a validation set.
        Falls back to EBM's native term importances if SHAP is unavailable.
        """
        if self.explainer is None:
            logger.warning("SHAP explainer missing. Using EBM term importances as fallback.")
            try:
                # EBM native importance
                # term_importances is an array of importances per feature/term
                importances = self.model.term_importances()
                global_imp = {name: float(val) for name, val in zip(self.feature_names, importances)}
                return dict(sorted(global_imp.items(), key=lambda item: item[1], reverse=True))
            except Exception as e:
                logger.error(f"EBM fallback importance failed: {e}")
                return {}

        logger.info("Calculating global importance via SHAP...")
        try:
            sample_df = self.X_train.sample(min(20, len(self.X_train)))
            sample_X = sample_df.values
                
            shap_values = self.explainer.shap_values(sample_X)
            
            if isinstance(shap_values, list):
                # For classification, use proba of class 1
                mean_abs_shap = np.abs(shap_values[1]).mean(axis=0)
            else:
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
            global_imp = {name: float(val) for name, val in zip(self.feature_names, mean_abs_shap)}
            return dict(sorted(global_imp.items(), key=lambda item: item[1], reverse=True))
        except Exception as e:
            logger.error(f"SHAP global importance failed: {e}")
            return {}
