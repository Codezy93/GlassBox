"""
Download the UCI Default of Credit Card Clients dataset.
Source: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

Saves to backend/data/credit_default.csv
"""

import os
import requests
import io
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(DATA_DIR, "credit_default.csv")

# UCI ML Repository direct download link (Excel format)
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


def download():
    """Download and convert the dataset to CSV."""
    if os.path.exists(OUTPUT_PATH):
        print(f"Dataset already exists at {OUTPUT_PATH}")
        return OUTPUT_PATH

    print("Downloading UCI Credit Default dataset …")
    try:
        resp = requests.get(URL, timeout=60)
        resp.raise_for_status()
        df = pd.read_excel(io.BytesIO(resp.content), header=1)
    except Exception as e:
        print(f"Download failed ({e}). Generating synthetic dataset instead …")
        df = _generate_synthetic()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_PATH}")
    return OUTPUT_PATH


def _generate_synthetic(n: int = 30000) -> pd.DataFrame:
    """
    Generate a realistic synthetic credit-default dataset as a fallback
    when the UCI download is unavailable.
    """
    import numpy as np

    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "ID": range(1, n + 1),
            "LIMIT_BAL": rng.choice(
                [10000, 20000, 30000, 50000, 80000, 100000, 150000, 200000, 300000, 500000],
                size=n,
            ),
            "SEX": rng.choice([1, 2], size=n),  # 1=male, 2=female
            "EDUCATION": rng.choice([1, 2, 3, 4], size=n, p=[0.1, 0.45, 0.35, 0.1]),
            "MARRIAGE": rng.choice([1, 2, 3], size=n, p=[0.35, 0.55, 0.1]),
            "AGE": rng.integers(21, 80, size=n),
        }
    )

    # Repayment status for 6 months (-1=on time, 1-9=months delay)
    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        df[col] = rng.choice([-1, 0, 1, 2, 3, 4, 5, 6, 7, 8], size=n, p=[0.35, 0.30, 0.12, 0.08, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01])

    # Bill amounts
    for col in ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]:
        df[col] = rng.integers(-10000, 400000, size=n)

    # Payment amounts
    for col in ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]:
        df[col] = rng.integers(0, 60000, size=n)

    # Generate target: higher default prob with high repayment delay, low limit
    risk_score = (
        0.3 * (df["PAY_0"] > 0).astype(float)
        + 0.2 * (df["PAY_2"] > 0).astype(float)
        + 0.15 * (df["LIMIT_BAL"] < 50000).astype(float)
        + 0.1 * (df["AGE"] < 30).astype(float)
        + rng.normal(0, 0.15, size=n)
    )
    df["default.payment" + ".next.month"] = (risk_score > 0.35).astype(int)

    return df


if __name__ == "__main__":
    download()
