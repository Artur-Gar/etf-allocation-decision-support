from __future__ import annotations

import numpy as np
import pandas as pd


def build_monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    monthly = prices.copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="coerce")
    monthly = monthly.dropna(subset=["date", "ticker", "adjusted_close"]).sort_values(["ticker", "date"])
    monthly["month_end"] = monthly["date"].dt.to_period("M").dt.to_timestamp("M")

    grouped = (
        monthly.groupby(["ticker", "month_end"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            adjusted_close=("adjusted_close", "last"),
            volume=("volume", "sum"),
            trading_days=("date", "count"),
        )
        .sort_values(["ticker", "month_end"])
        .reset_index(drop=True)
    )

    grouped["year"] = grouped["month_end"].dt.year
    grouped["month"] = grouped["month_end"].dt.month
    return grouped


def build_monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    returns = monthly_prices.copy().sort_values(["ticker", "month_end"]).reset_index(drop=True)
    grouped_price = returns.groupby("ticker")["adjusted_close"]
    returns["monthly_return"] = grouped_price.pct_change()
    returns["log_return"] = np.log(returns["adjusted_close"] / grouped_price.shift(1))
    returns["momentum_12m"] = grouped_price.pct_change(periods=12)
    returns["cumulative_return_index"] = returns.groupby("ticker")["monthly_return"].transform(
        lambda series: (1 + series.fillna(0)).cumprod()
    )
    return returns

