"""Monthly features and next-month targets."""

import numpy as np
import pandas as pd


def daily_to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "open", "high", "low", "close", "volume", "symbol"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    x = daily.copy()
    x["date"] = pd.to_datetime(x["date"], errors="coerce")
    x = x.dropna(subset=["date", "close"])
    x = x.sort_values(["symbol", "date"])

    # Assign months before calculating returns so that the first trading
    # day of a month is not paired with the previous month's final close.
    x["month"] = x["date"].dt.to_period("M").dt.to_timestamp()
    x["daily_log_return"] = (
        x.groupby(["symbol", "month"])["close"]
        .transform(lambda prices: np.log(prices / prices.shift(1)))
    )

    monthly = x.groupby(["symbol", "month"], as_index=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        daily_return_count=("daily_log_return", "count"),
        # Monthly realized volatility = square root of the sum of
        # squared daily log returns (the square root of realized variance).
        realized_volatility=(
            "daily_log_return",
            lambda returns: float(np.sqrt(np.square(returns).sum())),
        ),
    )

    monthly["monthly_return"] = (
        monthly.groupby("symbol")["close"].pct_change()
    )

    return monthly.sort_values(["symbol", "month"]).reset_index(drop=True)


def add_next_month_targets(monthly: pd.DataFrame) -> pd.DataFrame:
    x = monthly.sort_values(["symbol", "month"]).copy()
    grouped = x.groupby("symbol", sort=False)

    x["target_next_month_return"] = grouped["monthly_return"].shift(-1)
    x["target_next_month_volatility"] = (
        grouped["realized_volatility"].shift(-1)
    )

    return x
