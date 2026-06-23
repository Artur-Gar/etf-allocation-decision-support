from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd

from config import END_DATE, PROCESSED_ETF_CLASSIFIED_PATH, RAW_ETF_PRICES_PATH, RAW_ETF_UNIVERSE_PATH, START_DATE
from downloaders.prices import download_etf_prices
from utils import ensure_directories, first_existing_path, save_table


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the Yahoo price download."""
    parser = argparse.ArgumentParser(description="Download ETF prices from Yahoo Finance.")
    parser.add_argument("--start-date", type=str, default=START_DATE)
    parser.add_argument("--end-date", type=str, default=END_DATE)
    return parser.parse_args()


def run_price_gathering(
    start_date: str = START_DATE,
    end_date: str | None = END_DATE,
) -> dict[str, Any]:
    """Download ETF prices from Yahoo Finance and save one raw CSV."""
    ensure_directories()

    resolved_input = first_existing_path(
        PROCESSED_ETF_CLASSIFIED_PATH.resolve(),
        RAW_ETF_UNIVERSE_PATH.resolve(),
    )
    resolved_output = RAW_ETF_PRICES_PATH.resolve()

    if not resolved_input.exists():
        raise FileNotFoundError(f"ETF universe file not found: {resolved_input}")

    etf_universe = pd.read_csv(resolved_input)
    if "ticker" not in etf_universe.columns:
        raise ValueError(f"Input file is missing the required 'ticker' column: {resolved_input}")

    tickers = etf_universe["ticker"].dropna().astype(str).str.strip()
    tickers = [ticker for ticker in tickers if ticker]
    if not tickers:
        raise ValueError("No ETF tickers were found in the input file.")

    prices = download_etf_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    save_table(prices, resolved_output)

    return {
        "input_path": str(resolved_input),
        "output_path": str(resolved_output),
        "tickers_requested": len(tickers),
        "tickers_downloaded": int(prices["ticker"].nunique()) if not prices.empty else 0,
        "rows": len(prices),
    }


def main() -> None:
    """Run price gathering from the command line."""
    args = parse_args()
    summary = run_price_gathering(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

