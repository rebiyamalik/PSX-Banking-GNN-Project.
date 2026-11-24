"""Download and validate daily PSX OHLCV data."""
from pathlib import Path
import time
import pandas as pd

REQUIRED=["date","open","high","low","close","volume"]

def download_symbol(symbol,start,end):
    import psxdata
    df=psxdata.stocks(symbol,start=start,end=end)
    if df is None or df.empty: raise ValueError(f"No observations for {symbol}")
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    df=df.copy(); df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df=df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date",keep="last")
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df["symbol"]=symbol.upper()
    return df.reset_index(drop=True)

def download_universe(symbols,start,end,output_dir="data/raw",pause_seconds=1.0):
    folder=Path(output_dir); folder.mkdir(parents=True,exist_ok=True)
    successful=[]; errors={}
    for symbol in symbols:
        try:
            df=download_symbol(symbol,start,end)
            df.to_csv(folder/f"{symbol.upper()}_daily.csv",index=False)
            successful.append(symbol.upper()); print(f"{symbol}: {len(df)} rows saved")
        except Exception as exc:
            errors[symbol.upper()]=str(exc); print(f"{symbol}: FAILED - {exc}")
        time.sleep(pause_seconds)
    return successful,errors
