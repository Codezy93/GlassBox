"""
Data preprocessing pipeline for the UCI Credit Default dataset.

Renames raw columns to human-readable names, splits data, scales continuous
features, and persists artifacts.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(os.path.dirname(DATA_DIR), "models", "saved")

# ---------- human-readable column mapping ----------

RAW_TO_CLEAN = {
    "LIMIT_BAL": "credit_limit",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "repayment_sep",
    "PAY_2": "repayment_aug",
    "PAY_3": "repayment_jul",
    "PAY_4": "repayment_jun",
    "PAY_5": "repayment_may",
    "PAY_6": "repayment_apr",
    "BILL_AMT1": "bill_sep",
    "BILL_AMT2": "bill_aug",
    "BILL_AMT3": "bill_jul",
    "BILL_AMT4": "bill_jun",
    "BILL_AMT5": "bill_may",
    "BILL_AMT6": "bill_apr",
    "PAY_AMT1": "pay_sep",
    "PAY_AMT2": "pay_aug",
    "PAY_AMT3": "pay_jul",
    "PAY_AMT4": "pay_jun",
    "PAY_AMT5": "pay_may",
    "PAY_AMT6": "pay_apr",
    "default payment next month": "default",
    "default.payment.next.month": "default",
}

CONTINUOUS_FEATURES = [
    "credit_limit",
    "age",
    "bill_sep", "bill_aug", "bill_jul", "bill_jun", "bill_may", "bill_apr",
    "pay_sep", "pay_aug", "pay_jul", "pay_jun", "pay_may", "pay_apr",
]

CATEGORICAL_FEATURES = [
    "sex",          # 1=male, 2=female
    "education",    # 1=grad school, 2=university, 3=high school, 4=others
    "marriage",     # 1=married, 2=single, 3=others
    "repayment_sep", "repayment_aug", "repayment_jul",
    "repayment_jun", "repayment_may", "repayment_apr",
]

ALL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
TARGET = "default"


def load_raw() -> pd.DataFrame:
    """Load the raw CSV and rename columns."""
    csv_path = os.path.join(DATA_DIR, "credit_default.csv")
    if not os.path.exists(csv_path):
        try:
            from data.download_dataset import download
        except ModuleNotFoundError:
            from backend.data.download_dataset import download
        download()

    df = pd.read_csv(csv_path)

    # Drop row-ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df = df.rename(columns=RAW_TO_CLEAN)

    # Ensure target column exists
    if TARGET not in df.columns:
        # try alternative capitalization
        for c in df.columns:
            if "default" in c.lower():
                df = df.rename(columns={c: TARGET})
                break

    return df


def preprocess(test_size: float = 0.2, random_state: int = 42):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Scaled feature matrices and target vectors.
    feature_names : list[str]
        Ordered list of feature column names.
    scaler : StandardScaler
        Fitted scaler (saved to disk for inference).
    """
    df = load_raw()

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit scaler on continuous features only
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[CONTINUOUS_FEATURES] = scaler.fit_transform(
        X_train[CONTINUOUS_FEATURES]
    )
    X_test_scaled[CONTINUOUS_FEATURES] = scaler.transform(
        X_test[CONTINUOUS_FEATURES]
    )

    # Persist
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(ALL_FEATURES, os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Also persist unscaled splits for DiCE (needs original ranges)
    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        os.path.join(MODELS_DIR, "raw_splits.pkl"),
    )

    return (
        X_train_scaled.values.astype(np.float32),
        X_test_scaled.values.astype(np.float32),
        y_train,
        y_test,
        ALL_FEATURES,
        scaler,
    )


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, feats, sc = preprocess()
    print(f"Train: {X_tr.shape}  Test: {X_te.shape}")
    print(f"Features ({len(feats)}): {feats}")
    print(f"Default rate: {y_tr.mean():.2%} (train) / {y_te.mean():.2%} (test)")
