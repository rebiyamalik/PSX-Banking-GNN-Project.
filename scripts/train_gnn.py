"""Train and evaluate a small two-layer GCN for next-month bank returns and volatility.

Run after scripts/prepare_graph_dataset.py:
    python scripts/train_gnn.py

The scaler is fitted on training graphs only. Validation selects the epoch;
the test split is evaluated once after restoring the best validation checkpoint.
"""
from pathlib import Path
import json
import random
import sys

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it with: python -m pip install torch"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "graph_dataset.npz"
OUTPUT_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "reports"
SEED = 42
HIDDEN_DIM = 16
DROPOUT = 0.20
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
MAX_EPOCHS = 500
PATIENCE = 40


class GraphConvolution(nn.Module):
    """A simple graph convolution: A_normalized @ (X @ W) + b."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        return torch.matmul(adjacency, self.linear(x))


class BankingGCN(nn.Module):
    """Two graph-convolution layers and a node-wise two-target output head."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, adjacency):
        x = torch.relu(self.conv1(x, adjacency))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x, adjacency))
        return self.output(x)


def normalize_adjacency(adjacencies):
    """Symmetric D^-1/2 (A + I) D^-1/2 normalization for each graph."""
    eye = torch.eye(adjacencies.shape[-1], dtype=adjacencies.dtype)
    a = adjacencies + eye.unsqueeze(0)
    degree = a.sum(dim=-1).clamp_min(1e-8)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)


def regression_metrics(actual, predicted):
    """Return MAE, RMSE and R² for each target column."""
    result = {}
    for column, name in enumerate(("next_month_return", "next_month_volatility")):
        y_true = actual[..., column].reshape(-1)
        y_pred = predicted[..., column].reshape(-1)
        error = y_true - y_pred
        ss_res = float(np.square(error).sum())
        ss_total = float(np.square(y_true - y_true.mean()).sum())
        r2 = 1.0 - ss_res / ss_total if ss_total > 0 else None
        result[name] = {
            "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "R2": r2,
        }
    return result


def main():
    if not DATA_PATH.exists():
        raise SystemExit(
            "Graph dataset not found. Run python scripts/prepare_graph_dataset.py first."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data = np.load(DATA_PATH, allow_pickle=False)
    required = {
        "X", "A", "y", "months", "symbols", "feature_names", "target_names",
        "train_idx", "val_idx", "test_idx",
    }
    missing = required.difference(data.files)
    if missing:
        raise SystemExit(f"Graph dataset is missing arrays: {sorted(missing)}")

    X = data["X"].astype(np.float32)
    A = data["A"].astype(np.float32)
    y = data["y"].astype(np.float32)
    months = data["months"].astype(str)
    symbols = data["symbols"].astype(str)
    train_idx = data["train_idx"].astype(np.int64)
    val_idx = data["val_idx"].astype(np.int64)
    test_idx = data["test_idx"].astype(np.int64)

    if X.ndim != 3 or A.shape != (X.shape[0], X.shape[1], X.shape[1]):
        raise SystemExit(f"Incompatible X/A shapes: X={X.shape}, A={A.shape}")
    if y.shape[:2] != X.shape[:2] or y.shape[-1] != 2:
        raise SystemExit(f"Incompatible target shape: X={X.shape}, y={y.shape}")
    if not (len(train_idx) and len(val_idx) and len(test_idx)):
        raise SystemExit("Train, validation and test splits must all be non-empty.")
    if not (train_idx.max() < val_idx.min() and val_idx.max() < test_idx.min()):
        raise SystemExit("Splits are not chronological; rebuild the graph dataset.")

    # Fit feature and target standardization on training graphs only.
    x_train_raw = X[train_idx]
    y_train_raw = y[train_idx]
    x_mean = x_train_raw.mean(axis=(0, 1), keepdims=True)
    x_scale = x_train_raw.std(axis=(0, 1), keepdims=True)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    y_mean = y_train_raw.mean(axis=(0, 1), keepdims=True)
    y_scale = y_train_raw.std(axis=(0, 1), keepdims=True)
    y_scale = np.where(y_scale < 1e-8, 1.0, y_scale)

    X_scaled = (X - x_mean) / x_scale
    y_scaled = (y - y_mean) / y_scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    a_tensor = normalize_adjacency(
        torch.tensor(A, dtype=torch.float32, device=device)
    )
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=device)

    model = BankingGCN(
        input_dim=X.shape[-1],
        hidden_dim=HIDDEN_DIM,
        output_dim=y.shape[-1],
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        train_prediction = model(x_tensor[train_idx], a_tensor[train_idx])
        train_loss = loss_fn(train_prediction, y_tensor[train_idx])
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prediction = model(x_tensor[val_idx], a_tensor[val_idx])
            val_loss = float(loss_fn(val_prediction, y_tensor[val_idx]).item())

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss.item()),
            "validation_loss": val_loss,
        })

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss.item():.5f} | "
                f"val_loss={val_loss:.5f}"
            )
        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise SystemExit("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prediction_scaled = model(x_tensor[test_idx], a_tensor[test_idx]).cpu().numpy()

    test_prediction = test_prediction_scaled * y_scale + y_mean
    test_actual = y[test_idx]
    metrics = regression_metrics(test_actual, test_prediction)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / "banking_gcn.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": int(X.shape[-1]),
        "hidden_dim": HIDDEN_DIM,
        "output_dim": int(y.shape[-1]),
        "dropout": DROPOUT,
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "feature_names": data["feature_names"].astype(str).tolist(),
        "target_names": data["target_names"].astype(str).tolist(),
        "symbols": symbols.tolist(),
        "best_epoch": best_epoch,
        "seed": SEED,
    }, checkpoint_path)

    prediction_rows = []
    for local_i, sample_i in enumerate(test_idx):
        for node_i, symbol in enumerate(symbols):
            prediction_rows.append({
                "month": months[sample_i],
                "symbol": symbol,
                "actual_next_month_return": float(test_actual[local_i, node_i, 0]),
                "predicted_next_month_return": float(test_prediction[local_i, node_i, 0]),
                "actual_next_month_volatility": float(test_actual[local_i, node_i, 1]),
                "predicted_next_month_volatility": float(test_prediction[local_i, node_i, 1]),
            })
    import pandas as pd
    predictions_path = RESULTS_DIR / "gnn_test_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    report = {
        "model": "Two-layer GCN (plain PyTorch)",
        "device": str(device),
        "seed": SEED,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_validation_loss_standardized": best_val_loss,
        "split_sizes": {
            "train_graphs": int(len(train_idx)),
            "validation_graphs": int(len(val_idx)),
            "test_graphs": int(len(test_idx)),
        },
        "test_months": months[test_idx].tolist(),
        "metrics": metrics,
        "caution": (
            "Only a small number of monthly graph samples are available. "
            "Treat test metrics as exploratory, not conclusive evidence of predictive ability."
        ),
    }
    metrics_path = RESULTS_DIR / "gnn_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    history_path = RESULTS_DIR / "gnn_training_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"\nDevice: {device}")
    print(f"Best epoch: {best_epoch}; epochs run: {len(history)}")
    print("\nHeld-out test metrics (original target units):")
    for target_name, values in metrics.items():
        print(
            f"{target_name}: MAE={values['MAE']:.6f}, "
            f"RMSE={values['RMSE']:.6f}, R2={values['R2']}"
        )
    print(f"Saved model: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved test predictions: {predictions_path}")
    print(f"Saved training history: {history_path}")
    print("\nResearch caution: 32 monthly graphs are a very small deep-learning sample.")


if __name__ == "__main__":
    main()
