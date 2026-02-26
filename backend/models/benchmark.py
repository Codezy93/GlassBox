"""
Benchmark: compare Black-Box DNN vs Glassbox EBM side-by-side.
"""

import os
import sys
import numpy as np
import torch
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocess import preprocess  # noqa: E402
from models.train_blackbox import CreditDNN  # noqa: E402

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")


def benchmark():
    """Load both models and print a comparison table."""
    X_train, X_test, y_train, y_test, features, scaler = preprocess()

    raw = joblib.load(os.path.join(SAVED_DIR, "raw_splits.pkl"))
    X_test_raw = raw["X_test"]

    results = {}

    # ── DNN ──────────────────────────────────────────
    ckpt = torch.load(
        os.path.join(SAVED_DIR, "blackbox_model.pt"),
        map_location="cpu",
        weights_only=False,
    )
    dnn = CreditDNN(input_dim=ckpt["input_dim"])
    dnn.load_state_dict(ckpt["model_state_dict"])
    dnn.eval()

    with torch.no_grad():
        dnn_proba = (
            dnn(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
        )
    dnn_pred = (dnn_proba > 0.5).astype(int)

    results["Black-Box DNN"] = {
        "AUC-ROC": roc_auc_score(y_test, dnn_proba),
        "Accuracy": accuracy_score(y_test, dnn_pred),
        "F1": f1_score(y_test, dnn_pred),
    }

    # ── EBM ──────────────────────────────────────────
    ebm = joblib.load(os.path.join(SAVED_DIR, "ebm_model.pkl"))
    ebm_proba = ebm.predict_proba(X_test_raw)[:, 1]
    ebm_pred = ebm.predict(X_test_raw)

    results["Glassbox EBM"] = {
        "AUC-ROC": roc_auc_score(y_test, ebm_proba),
        "Accuracy": accuracy_score(y_test, ebm_pred),
        "F1": f1_score(y_test, ebm_pred),
    }

    # ── Print ────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Model':<20} {'AUC-ROC':>10} {'Accuracy':>10} {'F1':>10}")
    print("-" * 55)
    for name, m in results.items():
        print(f"{name:<20} {m['AUC-ROC']:>10.4f} {m['Accuracy']:>10.4f} {m['F1']:>10.4f}")
    print("=" * 55)

    return results


if __name__ == "__main__":
    benchmark()
