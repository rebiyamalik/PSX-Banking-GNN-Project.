"""Build monthly feature/target panel from daily CSV files."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import pandas as pd
from psx_banking.features import daily_to_monthly,add_next_month_targets

def main():
    files=sorted((ROOT/"data/raw").glob("*_daily.csv"))
    if not files: raise SystemExit("No *_daily.csv files. Run python scripts/download_data.py first.")
    daily=pd.concat([pd.read_csv(p,parse_dates=["date"]) for p in files],ignore_index=True)
    panel=add_next_month_targets(daily_to_monthly(daily))
    out=ROOT/"data/processed"; out.mkdir(parents=True,exist_ok=True)
    path=out/"monthly_panel.csv"; panel.to_csv(path,index=False)
    print(f"Saved {len(panel)} bank-month rows to {path}")
if __name__=="__main__": main()
