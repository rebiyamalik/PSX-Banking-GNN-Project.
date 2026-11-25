"""Expanding-window walk-forward evaluation of Mean, Persistence, MLP and GCN.

Run: python scripts/walk_forward_evaluation.py
Each forecast origin uses only earlier samples for fitting, scaling and validation.
"""
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/graph_dataset.npz"
OUT = ROOT / "reports"
SEED, MIN_HISTORY, VAL_FRAC = 42, 16, 0.20
HIDDEN, DROPOUT, LR, WD, EPOCHS, PATIENCE = 16, 0.20, 0.005, 0.001, 300, 30


class Conv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.linear = nn.Linear(i, o)

    def forward(self, x, a):
        return a @ self.linear(x)


class GCN(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.c1, self.c2 = Conv(i, h), Conv(h, h)
        self.drop, self.out = nn.Dropout(DROPOUT), nn.Linear(h, o)

    def forward(self, x, a):
        x = torch.relu(self.c1(x, a))
        x = self.drop(x)
        return self.out(torch.relu(self.c2(x, a)))


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))

    def forward(self, x):
        return self.net(x)


def norm_adj(a):
    eye = torch.eye(a.shape[-1], dtype=a.dtype)
    a = a + eye.unsqueeze(0)
    d = a.sum(-1).clamp_min(1e-8).rsqrt()
    return d.unsqueeze(-1) * a * d.unsqueeze(-2)


def scales(x, y):
    xm, xs = x.mean((0, 1), keepdims=True), x.std((0, 1), keepdims=True)
    ym, ys = y.mean((0, 1), keepdims=True), y.std((0, 1), keepdims=True)
    return xm, np.where(xs < 1e-8, 1.0, xs), ym, np.where(ys < 1e-8, 1.0, ys)


def fit_model(model, X, A, y, fit_idx, val_idx, device, graph):
    xm, xs, ym, ys = scales(X[fit_idx], y[fit_idx])
    xt = torch.tensor((X - xm) / xs, dtype=torch.float32, device=device)
    yt = torch.tensor((y - ym) / ys, dtype=torch.float32, device=device)
    at = norm_adj(torch.tensor(A, dtype=torch.float32, device=device)) if graph else None
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    best, best_loss, stale = None, float("inf"), 0
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        p = model(xt[fit_idx], at[fit_idx]) if graph else model(xt[fit_idx])
        loss = loss_fn(p, yt[fit_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            p = model(xt[val_idx], at[val_idx]) if graph else model(xt[val_idx])
            vl = float(loss_fn(p, yt[val_idx]).item())
        if vl < best_loss - 1e-7:
            best_loss, stale = vl, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best is None:
        raise RuntimeError("Model training did not produce a checkpoint.")
    model.load_state_dict(best)
    model.eval()
    return model, (xm, xs, ym, ys)


def predict(model, x, a, scaler, device, graph):
    xm, xs, ym, ys = scaler
    xt = torch.tensor((x - xm) / xs, dtype=torch.float32, device=device)
    with torch.no_grad():
        if graph:
            at = norm_adj(torch.tensor(a[None], dtype=torch.float32, device=device))
            p = model(xt, at).cpu().numpy()
        else:
            p = model(xt).cpu().numpy()
    return p * ys + ym


def metric(actual, pred):
    out = {}
    for j, name in enumerate(("next_month_return", "next_month_volatility")):
        t, p = actual[..., j].reshape(-1), pred[..., j].reshape(-1)
        e = t - p
        total = float(np.square(t - t.mean()).sum())
        residual = float(np.square(e).sum())
        out[name] = {"MAE": float(np.abs(e).mean()),
                     "RMSE": float(np.sqrt(np.square(e).mean())),
                     "R2": float(1 - residual / total) if total > 0 else None}
    return out


def main():
    if not DATA.exists():
        raise SystemExit("Graph dataset missing. Run scripts/prepare_graph_dataset.py first.")
    random.seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(DATA, allow_pickle=False)
    X, A, y = d["X"].astype("float32"), d["A"].astype("float32"), d["y"].astype("float32")
    months, symbols = d["months"].astype(str), d["symbols"].astype(str)
    if len(X) <= MIN_HISTORY:
        raise SystemExit(f"Need more than {MIN_HISTORY} samples; found {len(X)}.")

    rows = []
    origins = range(MIN_HISTORY, len(X))
    for k, idx in enumerate(origins, 1):
        history = np.arange(idx)
        nv = max(2, int(np.ceil(len(history) * VAL_FRAC)))
        fit_idx, val_idx = history[:-nv], history[-nv:]
        if len(fit_idx) < 3:
            continue
        actual, current_x = y[idx], X[idx]
        mean = np.broadcast_to(y[fit_idx].mean((0, 1), keepdims=True), actual.shape).copy()
        persistence = current_x[:, [0, 1]].copy()

        seed = SEED + idx
        torch.manual_seed(seed)
        mlp, ms = fit_model(MLP(X.shape[-1], HIDDEN, y.shape[-1]).to(device),
                            X, A, y, fit_idx, val_idx, device, False)
        mp = predict(mlp, current_x, A[idx], ms, device, False)

        torch.manual_seed(seed)
        gcn, gs = fit_model(GCN(X.shape[-1], HIDDEN, y.shape[-1]).to(device),
                            X, A, y, fit_idx, val_idx, device, True)
        gp = predict(gcn, current_x, A[idx], gs, device, True)

        preds = {"Mean baseline": mean, "Persistence baseline": persistence,
                 "MLP (no graph)": mp, "GCN (with graph)": gp}
        for name, p in preds.items():
            for node, bank in enumerate(symbols):
                rows.append({
                    "forecast_month": months[idx], "symbol": bank, "model": name,
                    "actual_next_month_return": float(actual[node, 0]),
                    "predicted_next_month_return": float(p[node, 0]),
                    "actual_next_month_volatility": float(actual[node, 1]),
                    "predicted_next_month_volatility": float(p[node, 1]),
                    "historical_samples": int(idx), "fit_samples": int(len(fit_idx)),
                    "validation_samples": int(len(val_idx)),
                })
        print(f"Forecast {k}/{len(X)-MIN_HISTORY}: {months[idx]} "
              f"(fit={len(fit_idx)}, validation={len(val_idx)})")

    if not rows:
        raise SystemExit("No walk-forward forecasts were generated.")
    frame = pd.DataFrame(rows)
    forecast_months = sorted(frame.forecast_month.unique().tolist())
    summary = {
        "evaluation": "expanding-window walk-forward",
        "device": str(device),
        "minimum_historical_samples": MIN_HISTORY,
        "forecast_origins": len(forecast_months),
        "forecast_months": forecast_months,
        "banks": symbols.tolist(),
        "metrics": {},
        "caution": "Small historical sample; treat results as exploratory, not conclusive."
    }
    for name, group in frame.groupby("model", sort=False):
        actual = np.stack([
            group.pivot(index="forecast_month", columns="symbol", values=c)
            .reindex(index=forecast_months, columns=symbols).to_numpy()
            for c in ("actual_next_month_return", "actual_next_month_volatility")
        ], axis=-1)
        pred = np.stack([
            group.pivot(index="forecast_month", columns="symbol", values=c)
            .reindex(index=forecast_months, columns=symbols).to_numpy()
            for c in ("predicted_next_month_return", "predicted_next_month_volatility")
        ], axis=-1)
        summary["metrics"][name] = metric(actual, pred)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path, predictions_path = OUT / "walk_forward_metrics.json", OUT / "walk_forward_predictions.csv"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    frame.to_csv(predictions_path, index=False)
    print("\\nWalk-forward comparison (original target units):")
    for name, values in summary["metrics"].items():
        r, v = values["next_month_return"], values["next_month_volatility"]
        print(f"{name}: Return MAE={r['MAE']:.5f}, RMSE={r['RMSE']:.5f}, R2={r['R2']:.4f} | "
              f"Volatility MAE={v['MAE']:.5f}, RMSE={v['RMSE']:.5f}, R2={v['R2']:.4f}")
    print(f"Forecast origins: {len(forecast_months)}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved predictions: {predictions_path}")
    print("Note: expanding-window results remain exploratory with this limited history.")


if __name__ == "__main__":
    main()
