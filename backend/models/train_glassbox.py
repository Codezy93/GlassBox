"""
Train the Glassbox Explainable Boosting Machine (EBM) from InterpretML.

The EBM is a natively interpretable model that typically matches DNN accuracy
while providing exact, lossless explanations.
"""

import os
import sys
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocess import preprocess, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")


def train():
    """Train an ExplainableBoostingClassifier and save the model."""
    from interpret.glassbox import ExplainableBoostingClassifier

    X_train, X_test, y_train, y_test, features, _ = preprocess()

    # EBM works well with unscaled data — reload raw splits
    raw = joblib.load(os.path.join(SAVED_DIR, "raw_splits.pkl"))
    X_train_raw = raw["X_train"]
    X_test_raw = raw["X_test"]

    # Feature types: 'continuous' or 'nominal'
    feature_types = []
    for f in features:
        if f in CONTINUOUS_FEATURES:
            feature_types.append("continuous")
        else:
            feature_types.append("nominal")

    # Ensure categorical columns are ints (avoids 'Mix of labels' errors)
    for col in CATEGORICAL_FEATURES:
        if col in X_train_raw.columns:
            X_train_raw[col] = X_train_raw[col].astype(int)
        if col in X_test_raw.columns:
            X_test_raw[col] = X_test_raw[col].astype(int)

    y_train_int = y_train.astype(int)
    y_test_int = y_test.astype(int)

    print("Training Explainable Boosting Machine …")
    ebm = ExplainableBoostingClassifier(
        feature_names=features,
        feature_types=feature_types,
        max_bins=256,
        interactions=10,
        outer_bags=8,
        inner_bags=0,
        learning_rate=0.01,
        max_rounds=5000,
        min_samples_leaf=2,
        random_state=42,
    )
    ebm.fit(X_train_raw, y_train_int)

    # Evaluate
    y_pred_proba = ebm.predict_proba(X_test_raw)[:, 1]
    y_pred = ebm.predict(X_test_raw)

    auc = roc_auc_score(y_test_int, y_pred_proba)
    acc = accuracy_score(y_test_int, y_pred)
    f1 = f1_score(y_test_int, y_pred)

    print(f"\n📊 EBM Results:")
    print(f"   AUC-ROC : {auc:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Score: {f1:.4f}")

    # Save
    os.makedirs(SAVED_DIR, exist_ok=True)
    joblib.dump(ebm, os.path.join(SAVED_DIR, "ebm_model.pkl"))
    print(f"\n✅ EBM saved to {SAVED_DIR}/ebm_model.pkl")

    return ebm


if __name__ == "__main__":
    train()
