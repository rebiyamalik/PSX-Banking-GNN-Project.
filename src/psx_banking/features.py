"""Monthly features and next-month targets."""
import numpy as np
import pandas as pd

def daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"date","open","high","low","close","volume","symbol"}
    missing = required.difference(daily.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    x=daily.copy(); x["date"]=pd.to_datetime(x["date"],errors="coerce")
    x=x.dropna(subset=["date","close"]).sort_values(["symbol","date"])
    x["daily_log_return"]=x.groupby("symbol")["close"].transform(lambda s: np.log(s/s.shift(1)))
    x["month"]=x["date"].dt.to_period("M").dt.to_timestamp()
    m=x.groupby(["symbol","month"],as_index=False).agg(
      open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),
      volume=("volume","sum"),daily_return_count=("daily_log_return","count"),
      realized_volatility=("daily_log_return",lambda s:s.std(ddof=1)*np.sqrt(s.count())))
    m["monthly_return"]=m.groupby("symbol")["close"].pct_change()
    return m.sort_values(["symbol","month"]).reset_index(drop=True)

def add_next_month_targets(monthly: pd.DataFrame) -> pd.DataFrame:
    x=monthly.sort_values(["symbol","month"]).copy(); g=x.groupby("symbol",sort=False)
    x["target_next_month_return"]=g["monthly_return"].shift(-1)
    x["target_next_month_volatility"]=g["realized_volatility"].shift(-1)
    return x
