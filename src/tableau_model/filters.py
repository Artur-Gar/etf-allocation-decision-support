from __future__ import annotations

import pandas as pd

from tableau_model.constants import GEOGRAPHY_ALIAS_TO_CANONICAL, MIN_MONTH_ID
from tableau_model.helpers import canonicalize_geography_name


def filter_monthly_macro_only(macro_monthly_source: pd.DataFrame) -> pd.DataFrame:
    """Keep only truly monthly macro rows for the relational Tableau export."""
    if macro_monthly_source.empty or "frequency_flag" not in macro_monthly_source.columns:
        return macro_monthly_source.copy()

    monthly_only = macro_monthly_source.loc[
        macro_monthly_source["frequency_flag"].astype(str).str.strip().eq("Monthly-ish")
    ].copy()
    return monthly_only.reset_index(drop=True)


def filter_equity_etfs_only(characteristics: pd.DataFrame) -> pd.DataFrame:
    """Keep only equity ETFs for the relational Tableau export."""
    if "asset_class" not in characteristics.columns:
        return characteristics.copy()

    equity_only = characteristics.loc[
        characteristics["asset_class"].astype(str).str.strip().eq("Equity")
    ].copy()
    return equity_only.reset_index(drop=True)


def filter_market_period(
    etf_market_source: pd.DataFrame,
    characteristics: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only equity ETF market rows from January 2016 onward."""
    current = etf_market_source.copy()
    current["month_id"] = current["month_end"].dt.strftime("%Y%m").astype("Int64")
    current = current.loc[current["month_id"] >= MIN_MONTH_ID].copy()

    tickers = set(characteristics["ticker"].dropna().astype(str).str.strip())
    current = current.loc[current["ticker"].astype(str).str.strip().isin(tickers)].copy()
    return current.drop(columns=["month_id"]).reset_index(drop=True)


def canonicalize_characteristics_geographies(
    characteristics: pd.DataFrame,
) -> tuple[pd.DataFrame, int, set[str]]:
    """Normalize ETF country exposure labels to canonical geography names."""
    current = characteristics.copy()
    remapped_cells = 0
    alias_names_found: set[str] = set()

    for column in ("top_country_1", "top_country_2", "top_country_3"):
        if column not in current.columns:
            continue
        original = current[column].copy()
        canonical = original.map(canonicalize_geography_name)
        changed_mask = (
            original.fillna("").astype(str).str.strip()
            != canonical.fillna("").astype(str).str.strip()
        )
        remapped_cells += int(changed_mask.sum())
        alias_names_found.update(
            value
            for value in original.loc[changed_mask].dropna().astype(str).str.strip()
            if value in GEOGRAPHY_ALIAS_TO_CANONICAL
        )
        current[column] = canonical

    return current, remapped_cells, alias_names_found


def canonicalize_macro_geographies(
    macro_monthly_source: pd.DataFrame,
) -> tuple[pd.DataFrame, int, set[str]]:
    """Normalize macro geography labels to canonical geography names."""
    current = macro_monthly_source.copy()
    if "country_or_region" not in current.columns:
        return current, 0, set()

    original = current["country_or_region"].copy()
    canonical = original.map(canonicalize_geography_name)
    changed_mask = (
        original.fillna("").astype(str).str.strip()
        != canonical.fillna("").astype(str).str.strip()
    )
    alias_names_found = {
        value
        for value in original.loc[changed_mask].dropna().astype(str).str.strip()
        if value in GEOGRAPHY_ALIAS_TO_CANONICAL
    }
    current["country_or_region"] = canonical
    return current, int(changed_mask.sum()), alias_names_found


def filter_macro_for_relational_export(macro_monthly_source: pd.DataFrame) -> pd.DataFrame:
    """Keep monthly macro data from January 2016 onward and deduplicate months."""
    current = filter_monthly_macro_only(macro_monthly_source)
    current["month_id"] = current["month_end"].dt.strftime("%Y%m").astype("Int64")
    current = current.loc[current["month_id"] >= MIN_MONTH_ID].copy()
    current = deduplicate_macro_series_month(current)
    return current.drop(columns=["month_id"]).reset_index(drop=True)


def deduplicate_macro_series_month(macro_monthly_source: pd.DataFrame) -> pd.DataFrame:
    """Ensure series_id + month_id is unique in macro observations."""
    if macro_monthly_source.empty:
        return macro_monthly_source.copy()

    current = macro_monthly_source.copy()
    current["source_date"] = pd.to_datetime(current["source_date"], errors="coerce")
    current["month_id"] = current["month_end"].dt.strftime("%Y%m").astype("Int64")
    current["_imputed_rank"] = current["is_imputed"].fillna(True).astype(int)
    current = current.sort_values(
        ["series_id", "month_id", "_imputed_rank", "source_date"],
        ascending=[True, True, True, False],
    )
    current = current.drop_duplicates(subset=["series_id", "month_id"], keep="first")
    return current.drop(columns=["_imputed_rank"]).reset_index(drop=True)

