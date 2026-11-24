"""Past-only rolling correlation graph helpers."""
import numpy as np
import pandas as pd

def rolling_correlation_matrix(returns: pd.DataFrame, as_of, window=6, min_periods=None):
    if window<2: raise ValueError("window must be at least 2")
    hist=returns.loc[pd.to_datetime(returns.index)<pd.Timestamp(as_of)].sort_index().tail(window)
    need=min_periods if min_periods is not None else max(2,window//2)
    if len(hist)<need: raise ValueError(f"Need {need} past observations; found {len(hist)}")
    return hist.corr(min_periods=need)

def correlation_adjacency(correlation: pd.DataFrame, threshold=0.3, absolute=True):
    v=correlation.fillna(0).to_numpy(dtype=float); score=np.abs(v) if absolute else v
    a=(score>=threshold).astype(np.float32); np.fill_diagonal(a,1.0); return a
