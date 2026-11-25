"""Prepare chronological graph samples for the PSX banking GNN."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from psx_banking.graph import rolling_correlation_matrix, correlation_adjacency

PANEL_PATH = ROOT / "data/processed/monthly_panel.csv"
CONFIG_PATH = ROOT / "config/banks.yaml"
OUTPUT_DIR = ROOT / "data/processed"
WINDOW, MIN_PERIODS, THRESHOLD = 6, 3, 0.30
FEATURE_COLUMNS = [
    "monthly_return", "realized_volatility", "log_volume",
    "intramonth_range_pct", "monthly_body_return", "daily_return_count",
]
TARGET_COLUMNS = ["target_next_month_return", "target_next_month_volatility"]


def main():
    if not PANEL_PATH.exists():
        raise SystemExit(f"Missing {PANEL_PATH}. Run python scripts/build_dataset.py first.")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    symbols = [str(s).upper() for s in config["banks"]]
    panel = pd.read_csv(PANEL_PATH, parse_dates=["month"])
    required = {
        "symbol", "month", "open", "high", "low", "close", "volume",
        "daily_return_count", "realized_volatility", "monthly_return",
        *TARGET_COLUMNS,
    }
    missing = required.difference(panel.columns)
    if missing:
        raise SystemExit(f"Monthly panel is missing columns: {sorted(missing)}")

    panel["symbol"] = panel["symbol"].str.upper()
    panel["month"] = panel["month"].dt.to_period("M").dt.to_timestamp()
    panel = panel[panel["symbol"].isin(symbols)].copy()
    if set(panel["symbol"].unique()) != set(symbols):
        raise SystemExit("Monthly panel does not contain all configured banks.")
    counts = panel.groupby("month")["symbol"].nunique()
    if (counts != len(symbols)).any():
        raise SystemExit("Every month must contain exactly one row per configured bank.")
    if panel.duplicated(["month", "symbol"]).any():
        raise SystemExit("Duplicate bank-month rows found.")

    panel["log_volume"] = np.log1p(panel["volume"].clip(lower=0))
    panel["intramonth_range_pct"] = (
        (panel["high"] - panel["low"]) / panel["close"].replace(0, np.nan)
    )
    panel["monthly_body_return"] = (
        panel["close"] / panel["open"].replace(0, np.nan) - 1.0
    )

    months = sorted(panel["month"].unique())
    return_wide = (
        panel.pivot(index="month", columns="symbol", values="monthly_return")
        .reindex(index=months, columns=symbols).sort_index()
    )
    # October 2023 has no prior month-end price, so it cannot contribute
    # a monthly return to a complete cross-bank correlation observation.
    complete_returns = return_wide.dropna(axis=0, how="any")

    xs, adjacencies, ys, sample_months, skipped = [], [], [], [], []
    for month in months:
        label = pd.Timestamp(month).strftime("%Y-%m")
        current = panel.loc[panel["month"] == month].set_index("symbol").reindex(symbols)
        x = current[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        target = current[TARGET_COLUMNS].to_numpy(dtype=np.float32)

        if not np.isfinite(x).all() or not np.isfinite(target).all():
            skipped.append(f"{label}: missing feature or next-month target")
            continue

        prior = complete_returns.loc[complete_returns.index < pd.Timestamp(month)].tail(WINDOW)
        if len(prior) < MIN_PERIODS:
            skipped.append(f"{label}: only {len(prior)} complete prior return months")
            continue

        corr = rolling_correlation_matrix(
            complete_returns, as_of=month, window=WINDOW, min_periods=MIN_PERIODS
        ).reindex(index=symbols, columns=symbols)
        if not np.isfinite(corr.to_numpy(dtype=float)).all():
            skipped.append(f"{label}: correlation matrix contains missing values")
            continue

        adjacency = correlation_adjacency(corr, threshold=THRESHOLD, absolute=True)
        xs.append(x)
        adjacencies.append(adjacency.astype(np.float32))
        ys.append(target)
        sample_months.append(label)

    if not xs:
        raise SystemExit("No graph samples built; inspect history and missing values.")

    X, A, y = np.stack(xs), np.stack(adjacencies), np.stack(ys)
    month_labels = np.asarray(sample_months)
    n = len(month_labels)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    if train_end < 1 or val_end <= train_end or val_end >= n:
        raise SystemExit(f"Not enough graph samples ({n}) for chronological split.")
    train_idx = np.arange(0, train_end, dtype=np.int64)
    val_idx = np.arange(train_end, val_end, dtype=np.int64)
    test_idx = np.arange(val_end, n, dtype=np.int64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "graph_dataset.npz"
    np.savez_compressed(
        output_path, X=X, A=A, y=y, months=month_labels,
        symbols=np.asarray(symbols), feature_names=np.asarray(FEATURE_COLUMNS),
        target_names=np.asarray(TARGET_COLUMNS), train_idx=train_idx,
        val_idx=val_idx, test_idx=test_idx,
    )
    manifest = {
        "samples": int(n), "nodes_per_graph": len(symbols),
        "features_per_node": len(FEATURE_COLUMNS),
        "targets_per_node": len(TARGET_COLUMNS),
        "feature_names": FEATURE_COLUMNS, "target_names": TARGET_COLUMNS,
        "correlation_window_months": WINDOW, "correlation_min_periods": MIN_PERIODS,
        "correlation_threshold": THRESHOLD, "correlation_uses_absolute_values": True,
        "graph_uses_only_complete_months_before_sample_month": True,
        "split": {
            "train": [int(train_idx[0]), int(train_idx[-1])],
            "validation": [int(val_idx[0]), int(val_idx[-1])],
            "test": [int(test_idx[0]), int(test_idx[-1])],
        },
        "sample_months": sample_months, "skipped_months": skipped,
        "note": "Features are not standardized. Fit any scaler on training data only.",
    }
    manifest_path = OUTPUT_DIR / "graph_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved graph dataset: {output_path}")
    print(f"X shape (samples, nodes, features): {X.shape}")
    print(f"A shape (samples, nodes, nodes): {A.shape}")
    print(f"y shape (samples, nodes, targets): {y.shape}")
    print(f"Chronological split: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}")
    print(f"Sample months: {sample_months[0]} to {sample_months[-1]}")
    print("Skipped months:")
    for item in skipped:
        print(f"  - {item}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
