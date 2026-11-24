"""Download daily data for configured banks."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import yaml
from psx_banking.data import download_universe

def main():
    cfg=yaml.safe_load((ROOT/"config/banks.yaml").read_text(encoding="utf-8"))
    ok,errors=download_universe(cfg["banks"],cfg["data"]["start_date"],cfg["data"]["end_date"],ROOT/"data/raw")
    print(f"Downloaded {len(ok)}/{len(cfg['banks'])} symbols")
    if errors:
        for symbol,error in errors.items(): print(f"{symbol}: {error}")
        raise SystemExit(1)
if __name__=="__main__": main()
