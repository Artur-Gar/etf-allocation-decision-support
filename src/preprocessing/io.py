from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    PROCESSED_ETF_CLASSIFIED_PATH,
    RAW_DIR,
    RAW_ETF_PRICES_PATH,
    RAW_ETF_UNIVERSE_PATH,
    RAW_MACRO_PATH,
)
from utils import first_existing_path


def load_raw_table(name: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name}.csv")


def raw_table_exists(name: str) -> bool:
    return Path(RAW_DIR / f"{name}.csv").exists()


def load_preferred_etf_universe() -> pd.DataFrame:
    """Prefer the classified ETF universe, then fall back to the raw iShares universe."""
    path = first_existing_path(
        PROCESSED_ETF_CLASSIFIED_PATH,
        RAW_ETF_UNIVERSE_PATH,
        RAW_DIR / "etf_characteristics_llm_classified.csv",
        RAW_DIR / "etf_characteristics.csv",
    )
    return pd.read_csv(path)


def load_preferred_price_table() -> pd.DataFrame:
    """Load ETF prices from the new Yahoo file name, with legacy fallback support."""
    path = first_existing_path(
        RAW_ETF_PRICES_PATH,
        RAW_DIR / "etf_prices.csv",
    )
    return pd.read_csv(path)


def load_preferred_macro_table() -> pd.DataFrame | None:
    """Load macro indicators when available, with legacy fallback support."""
    path = first_existing_path(
        RAW_MACRO_PATH,
        RAW_DIR / "macro_indicators.csv",
    )
    if not path.exists():
        return None
    return pd.read_csv(path)

