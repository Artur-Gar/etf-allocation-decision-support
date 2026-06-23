from __future__ import annotations

from config import (
    PROCESSED_ETF_MONTHLY_RETURNS_PATH,
    PROCESSED_MACRO_MONTHLY_PATH,
    PROCESSED_MACRO_QUALITY_PATH,
)
from preprocessing.io import load_preferred_macro_table, load_preferred_price_table
from preprocessing.macro import build_macro_monthly, build_macro_quality
from preprocessing.monthly import build_monthly_prices, build_monthly_returns
from utils import ensure_directories, save_table


def run_preprocessing() -> None:
    """Build the processed CSV files used by the final Tableau workbook."""
    ensure_directories()

    prices = load_preferred_price_table()
    macro = load_preferred_macro_table()

    monthly_prices = build_monthly_prices(prices)
    monthly_returns = build_monthly_returns(monthly_prices)
    macro_quality = build_macro_quality(macro)
    macro_monthly = build_macro_monthly(macro, macro_quality)

    save_table(monthly_returns, PROCESSED_ETF_MONTHLY_RETURNS_PATH)
    if not macro_quality.empty:
        save_table(macro_quality, PROCESSED_MACRO_QUALITY_PATH)
    if not macro_monthly.empty:
        save_table(macro_monthly, PROCESSED_MACRO_MONTHLY_PATH)

