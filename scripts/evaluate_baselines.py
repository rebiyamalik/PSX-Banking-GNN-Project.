"""Compare mean, persistence, MLP and trained GCN on the held-out test period.

Run:
    python scripts/evaluate_baselines.py

All learned preprocessing/model parameters use training data only.
Validation is used only for MLP early stopping. The test split is evaluated
after model selection and is not used for tuning.
"""
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("PyTorch is required. Run: python -m pip install torch") from exc

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "graph_dataset.npz"
GCN_PATH = ROOT / "models" / "banking_gcn.pt"
OUTPUT_DIR = ROOT / "reports"
SEED = 42
HIDDEN_DIM = 16
DROPOUT = 0.20
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
MAX_EPOCHS = 500
PATIENCE = 40


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency):
        return torch.matmul(adjacency, self.linear(x))


class BankingGCN(nn.Module):
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


class NodeMLP(nn.Module):
    """Shared node-wise MLP; same node features and target dimensions as GCN."""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def normalize_adjacency(adjacencies):
    eye = torch.eye(adjacencies.shape[-1], dtype=adjacencies.dtype)
    a = adjacencies + eye.unsqueeze(0)
    degree = a.sum(dim=-1).clamp_min(1e-8)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)


def calculate_metrics(actual, predicted):
    output = {}
    target_names = ("next_month_return", "next_month_volatility")
    for j, name in enumerate(target_names):
        truth = actual[..., j].reshape(-1)
        estimate = predicted[..., j].reshape(-1)
        error = truth - estimate
        ss_total = float(np.square(truth - truth.mean()).sum())
        ss_res = float(np.square(error).sum())
        output[name] = {
            "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.square(error).mean())),
            "R2": float(1.0 - ss_res / ss_total) if ss_total > 0 else None,
        }
    return output


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Graph dataset missing. Run scripts/prepare_graph_dataset.py first.")
    if not GCN_PATH.exists():
        raise SystemExit("Trained GCN checkpoint missing. Run scripts/train_gnn.py first.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(DATA_PATH, allow_pickle=False)
    X = data["X"].astype(np.float32)
    A = data["A"].astype(np.float32)
    y = data["y"].astype(np.float32)
    months = data["months"].astype(str)
    symbols = data["symbols"].astype(str)
    train_idx = data["train_idx"].astype(np.int64)
    val_idx = data["val_idx"].astype(np.int64)
    test_idx = data["test_idx"].astype(np.int64)

    if not (train_idx.max() < val_idx.min() and val_idx.max() < test_idx.min()):
        raise SystemExit("Dataset splits are not chronological.")

    # Mean benchmark: training-set target mean, independently for each target.
    train_target_mean = y[train_idx].mean(axis=(0, 1), keepdims=True)
    mean_prediction = np.broadcast_to(train_target_mean, y[test_idx].shape).copy()

    # Persistence benchmark: forecast next-month values using current-month
    # return and realized volatility, which are feature columns 0 and 1.
    persistence_prediction = X[test_idx][:, :, [0, 1]].copy()

    # Recreate the already-trained GCN and use its stored training-only scalers.
    checkpoint = torch.load(GCN_PATH, map_location="cpu", weights_only=False)
    gcn = BankingGCN(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    gcn.load_state_dict(checkpoint["model_state_dict"])
    gcn.to(device).eval()

    x_mean_gcn = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_scale_gcn = np.asarray(checkpoint["x_scale"], dtype=np.float32)
    y_mean_gcn = np.asarray(checkpoint["y_mean"], dtype=np.float32)
    y_scale_gcn = np.asarray(checkpoint["y_scale"], dtype=np.float32)
    X_gcn = torch.tensor((X - x_mean_gcn) / x_scale_gcn, dtype=torch.float32, device=device)
    A_tensor = normalize_adjacency(torch.tensor(A, dtype=torch.float32, device=device))
    with torch.no_grad():
        gcn_scaled = gcn(X_gcn[test_idx], A_tensor[test_idx]).cpu().numpy()
    gcn_prediction = gcn_scaled * y_scale_gcn + y_mean_gcn

    # MLP: same feature columns and targets, but no adjacency information.
    x_mean = X[train_idx].mean(axis=(0, 1), keepdims=True)
    x_scale = X[train_idx].std(axis=(0, 1), keepdims=True)
    x_scale = np.where(x_scale < 1e-8, 1.0, x_scale)
    y_mean = y[train_idx].mean(axis=(0, 1), keepdims=True)
    y_scale = y[train_idx].std(axis=(0, 1), keepdims=True)
    y_scale = np.where(y_scale < 1e-8, 1.0, y_scale)
    X_scaled = (X - x_mean) / x_scale
    y_scaled = (y - y_mean) / y_scale

    mlp = NodeMLP(X.shape[-1], HIDDEN_DIM, y.shape[-1], DROPOUT).to(device)
    optimizer = torch.optim.Adam(
        mlp.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = nn.MSELoss()
    train_x = torch.tensor(X_scaled[train_idx].reshape(-1, X.shape[-1]), dtype=torch.float32, device=device)
    train_y = torch.tensor(y_scaled[train_idx].reshape(-1, y.shape[-1]), dtype=torch.float32, device=device)
    val_x = torch.tensor(X_scaled[val_idx].reshape(-1, X.shape[-1]), dtype=torch.float32, device=device)
    val_y = torch.tensor(y_scaled[val_idx].reshape(-1, y.shape[-1]), dtype=torch.float32, device=device)

    best_state, best_val, best_epoch, stale = None, float("inf"), 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        mlp.train()
        optimizer.zero_grad()
        train_loss = loss_fn(mlp(train_x), train_y)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
        optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(mlp(val_x), val_y).item())
        if val_loss < best_val - 1e-7:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"MLP epoch {epoch:03d} | train={train_loss.item():.5f} | val={val_loss:.5f}")
        if stale >= PATIENCE:
            print(f"MLP early stopping at epoch {epoch}.")
            break

    mlp.load_state_dict(best_state)
    mlp.eval()
    test_x = torch.tensor(
        X_scaled[test_idx].reshape(-1, X.shape[-1]),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        mlp_scaled = mlp(test_x).cpu().numpy().reshape(y[test_idx].shape)
    mlp_prediction = mlp_scaled * y_scale + y_mean

    actual = y[test_idx]
    predictions = {
        "Mean baseline": mean_prediction,
        "Persistence baseline": persistence_prediction,
        "MLP (no graph)": mlp_prediction,
        "GCN (with graph)": gcn_prediction,
    }

    metrics = {
        "device": str(device),
        "test_months": months[test_idx].tolist(),
        "test_graphs": int(len(test_idx)),
        "test_bank_nodes": int(len(test_idx) * len(symbols)),
        "mlp_best_epoch": int(best_epoch),
        "metrics": {
            model_name: calculate_metrics(actual, prediction)
            for model_name, prediction in predictions.items()
        },
        "caution": (
            "Only five held-out monthly graphs are available. Metrics are exploratory; "
            "do not claim general predictive performance from this test alone."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUTPUT_DIR / "baseline_comparison.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    rows = []
    for model_name, prediction in predictions.items():
        for local_i, sample_i in enumerate(test_idx):
            for node_i, symbol in enumerate(symbols):
                rows.append({
                    "model": model_name,
                    "month": months[sample_i],
                    "symbol": symbol,
                    "actual_next_month_return": float(actual[local_i, node_i, 0]),
                    "predicted_next_month_return": float(prediction[local_i, node_i, 0]),
                    "actual_next_month_volatility": float(actual[local_i, node_i, 1]),
                    "predicted_next_month_volatility": float(prediction[local_i, node_i, 1]),
                })
    predictions_path = OUTPUT_DIR / "baseline_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    print("\nHeld-out test comparison (original target units):")
    for model_name, model_metrics in metrics["metrics"].items():
        ret = model_metrics["next_month_return"]
        vol = model_metrics["next_month_volatility"]
        print(
            f"{model_name}: "
            f"Return MAE={ret['MAE']:.5f}, RMSE={ret['RMSE']:.5f}, R2={ret['R2']:.4f} | "
            f"Volatility MAE={vol['MAE']:.5f}, RMSE={vol['RMSE']:.5f}, R2={vol['R2']:.4f}"
        )
    print(f"MLP best epoch: {best_epoch}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print("Reminder: test results are exploratory because the test period has only five months.")


if __name__ == "__main__":
    main()
