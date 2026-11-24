# PSX Banking Risk & Return Forecasting (GAT–LSTM)

Research prototype for forecasting next-month realized volatility and returns for a provisional universe of ten PSX commercial banks using past-only market features and rolling cross-bank relationships.

**Status:** Research scaffold and data pipeline. No model has been trained and no performance claim is made.

## Universe (provisional)
UBL, HBL, MEBL, MCB, BAHL, NBP, FABL, BAFL, ABL, AKBL. Verify symbols, listing continuity, history and corporate actions before analysis.

## Research design
- Collect daily OHLCV from 2023-10-01 through 2026-10-31; intended forecast analysis window is Nov 2023–Oct 2026.
- Aggregate daily observations to calendar months.
- Calculate realized volatility from daily log returns within each month.
- Align month-t features with month-(t+1) outcomes.
- Build correlation graphs using only returns available before the forecast month.
- Use chronological walk-forward validation, never random time-series splits.

## Setup (Windows PowerShell)
Python 3.11 or 3.12 is recommended for broad PyTorch ecosystem compatibility.
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
For the optional GAT-LSTM install PyTorch for your exact OS/Python/CPU-CUDA setup from https://pytorch.org/get-started/locally/ and a compatible PyTorch Geometric build from https://pytorch-geometric.readthedocs.io/.

## Run
```powershell
python scripts/download_data.py
python scripts/build_dataset.py
pytest -q
```
The downloader uses third-party `psxdata`, not an official PSX API. Verify package terms and data quality.

## Important safeguards
1. Raw prices may not be adjusted for dividends, splits, bonus issues or other corporate actions. Investigate sudden discontinuities before calculating returns.
2. Missing sessions are not automatically zero returns; check holidays, suspensions and collection gaps.
3. 36 forecast months across 10 banks is a very small sample for deep learning. Benchmark persistence and tree-based models; treat GAT-LSTM as exploratory.
4. Fit scalers on training data only. Construct graph edges from past-only data. Align predictors at t with targets at t+1.
5. Document bank membership and listing history to address survivorship/universe bias.

Report MAE and RMSE separately for return and volatility, along with forecast dates, training windows and baseline comparisons. This is an academic prototype, not investment advice.

## Repository layout
- `config/banks.yaml`: universe and date settings
- `scripts/`: data download and monthly panel builder
- `src/psx_banking/`: reusable data, features, graph, evaluation and model code
- `tests/`: basic tests
- `data/raw/` and `data/processed/`: local data, excluded from Git

## License
No license is selected yet; reuse is not explicitly licensed until one is added.
