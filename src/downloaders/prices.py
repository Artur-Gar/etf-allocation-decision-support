from __future__ import annotations

import warnings

import pandas as pd
import yfinance as yf

from config import YAHOO_BATCH_SIZE, YAHOO_TIMEOUT


PRICE_COLUMNS = {
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adjusted_close",
    "Volume": "volume",
}


def _normalize_column_name(column: object) -> str:
    if isinstance(column, tuple):
        column = next((part for part in column if part), column[0])
    return PRICE_COLUMNS.get(str(column), str(column).lower().replace(" ", "_"))


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index:index + size] for index in range(0, len(values), size)]


def _extract_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if data.empty:
        return None

    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()

    level_0 = set(data.columns.get_level_values(0))
    level_1 = set(data.columns.get_level_values(1))

    if ticker in level_0:
        return data[ticker].copy()
    if ticker in level_1:
        return data.xs(ticker, axis=1, level=1).copy()
    return None


def _download_price_batch(tickers: list[str], start_date: str, end_date: str | None) -> pd.DataFrame:
    try:
        return yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
            timeout=YAHOO_TIMEOUT,
        )
    except Exception as exc:
        warnings.warn(f"Could not download Yahoo batch {tickers[:3]}...: {exc}")
        return pd.DataFrame()


def download_etf_prices(tickers: list[str], start_date: str, end_date: str | None) -> pd.DataFrame:
    cleaned_tickers = sorted({ticker.strip().upper() for ticker in tickers if isinstance(ticker, str) and ticker.strip()})
    frames: list[pd.DataFrame] = []

    for batch in _chunked(cleaned_tickers, YAHOO_BATCH_SIZE):
        batch_data = _download_price_batch(batch, start_date=start_date, end_date=end_date)

        for ticker in batch:
            ticker_frame = _extract_ticker_frame(batch_data, ticker)
            if ticker_frame is None or ticker_frame.empty:
                warnings.warn(f"No price data returned for {ticker}.")
                continue

            current = ticker_frame.reset_index()
            current.columns = [_normalize_column_name(column) for column in current.columns]
            if "adjusted_close" not in current.columns and "close" in current.columns:
                current["adjusted_close"] = current["close"]

            current["ticker"] = ticker
            frames.append(current[["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"]])

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "adjusted_close", "volume"])

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.date
    return prices.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)

