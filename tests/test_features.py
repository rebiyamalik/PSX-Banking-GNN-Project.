import numpy as np
import pandas as pd

from psx_banking.features import daily_to_monthly, add_next_month_targets


def sample():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-02", "2024-01-03",
            "2024-02-01", "2024-02-02",
            "2024-03-01", "2024-03-04",
        ]),
        "open": [10, 11, 12, 13, 14, 15],
        "high": [11, 12, 13, 14, 15, 16],
        "low": [9, 10, 11, 12, 13, 14],
        "close": [10, 11, 12, 13, 14, 15],
        "volume": [100, 110, 120, 130, 140, 150],
        "symbol": ["UBL"] * 6,
    })


def test_monthly_aggregation():
    monthly = daily_to_monthly(sample())
    assert len(monthly) == 3
    assert monthly.month.dt.strftime("%Y-%m").tolist() == [
        "2024-01", "2024-02", "2024-03"
    ]


def test_realized_volatility_excludes_cross_month_return():
    monthly = daily_to_monthly(sample())

    # February has exactly one within-month return: log(13 / 12).
    # Its sample standard deviation is undefined, so monthly volatility is NaN.
    assert pd.isna(monthly.loc[1, "realized_volatility"])
    assert monthly.loc[1, "daily_return_count"] == 1


def test_next_month_targets():
    result = add_next_month_targets(daily_to_monthly(sample()))
    assert pd.notna(result.loc[0, "target_next_month_return"])
    assert pd.isna(result.loc[2, "target_next_month_return"])
