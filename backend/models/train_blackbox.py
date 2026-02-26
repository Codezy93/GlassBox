"""
Train a Black-Box PyTorch Deep Neural Network for credit default prediction.

Architecture: 3 hidden layers (256 → 128 → 64) with ReLU, dropout, and
sigmoid output for binary classification.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.preprocess import preprocess  # noqa: E402

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")


class CreditDNN(nn.Module):
    """3-layer DNN for binary credit-default prediction."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def train(epochs: int = 50, batch_size: int = 512, lr: float = 1e-3):
    """Train the DNN and save the model checkpoint."""
    X_train, X_test, y_train, y_test, features, _ = preprocess()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataloaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    model = CreditDNN(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_auc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        scheduler.step()

        # Evaluate
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            from sklearn.metrics import roc_auc_score, accuracy_score

            all_preds, all_labels = [], []
            with torch.no_grad():
                for xb, yb in test_dl:
                    xb = xb.to(device)
                    pred = model(xb).cpu().numpy().flatten()
                    all_preds.extend(pred)
                    all_labels.extend(yb.numpy().flatten())

            auc = roc_auc_score(all_labels, all_preds)
            acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
            avg_loss = running_loss / len(train_ds)
            print(
                f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} "
                f"| Test AUC: {auc:.4f} | Test Acc: {acc:.4f}"
            )

            if auc > best_auc:
                best_auc = auc
                os.makedirs(SAVED_DIR, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": X_train.shape[1],
                        "best_auc": best_auc,
                    },
                    os.path.join(SAVED_DIR, "blackbox_model.pt"),
                )

    print(f"\n✅ Best AUC: {best_auc:.4f} — model saved to {SAVED_DIR}/blackbox_model.pt")
    return model


if __name__ == "__main__":
    train()
