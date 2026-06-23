from __future__ import annotations

from typing import Any

import pandas as pd

from tableau_model.dimensions import (
    build_benchmark_table,
    build_etf_table,
    build_geography_table,
    build_industry_table,
    build_macro_indicator_table,
    build_macro_series_table,
    build_month_table,
)
from tableau_model.facts import (
    build_etf_geography_exposure_monthly,
    build_etf_industry_exposure_monthly,
    build_etf_market_monthly,
    build_etf_metric_monthly,
    build_macro_observation_monthly,
)
from tableau_model.filters import (
    canonicalize_characteristics_geographies,
    canonicalize_macro_geographies,
    filter_equity_etfs_only,
    filter_macro_for_relational_export,
    filter_market_period,
)
from tableau_model.validation import validate_relational_tables


def build_relational_tables(
    *,
    characteristics: pd.DataFrame,
    etf_market_source: pd.DataFrame,
    macro_monthly_source: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Backward-compatible wrapper returning only the relational tables."""
    tables, _ = build_relational_tables_with_summary(
        characteristics=characteristics,
        etf_market_source=etf_market_source,
        macro_monthly_source=macro_monthly_source,
    )
    return tables


def build_relational_tables_with_summary(
    *,
    characteristics: pd.DataFrame,
    etf_market_source: pd.DataFrame,
    macro_monthly_source: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build all normalized tables for the relational Tableau workbook."""
    current_characteristics = filter_equity_etfs_only(characteristics.copy())
    current_characteristics, characteristics_remaps, alias_names_in_characteristics = (
        canonicalize_characteristics_geographies(current_characteristics)
    )

    current_market = etf_market_source.copy()
    current_macro = macro_monthly_source.copy()
    current_market["month_end"] = pd.to_datetime(current_market["month_end"], errors="coerce")
    current_macro["month_end"] = pd.to_datetime(current_macro["month_end"], errors="coerce")
    current_macro["source_date"] = pd.to_datetime(current_macro["source_date"], errors="coerce")

    current_market = filter_market_period(current_market, current_characteristics)
    current_macro, macro_remaps, alias_names_in_macro = canonicalize_macro_geographies(current_macro)
    current_macro = filter_macro_for_relational_export(current_macro)

    snapshot_month_end = current_market["month_end"].max()
    if pd.isna(snapshot_month_end):
        raise ValueError("ETF market data does not contain a valid month_end.")

    benchmark = build_benchmark_table(current_characteristics)
    geography = build_geography_table(current_characteristics, current_macro)
    etf = build_etf_table(current_characteristics, benchmark)
    industry = build_industry_table(current_characteristics)
    macro_indicator = build_macro_indicator_table(current_macro)
    month = build_month_table(current_market, current_macro, snapshot_month_end)
    macro_series, series_id_remap, macro_series_summary = build_macro_series_table(current_macro, geography, macro_indicator)

    etf_market_monthly = build_etf_market_monthly(current_market, etf, month)
    etf_metric_monthly = build_etf_metric_monthly(current_characteristics, etf)
    macro_observation_monthly, macro_observation_summary = build_macro_observation_monthly(current_macro, macro_series, month, series_id_remap)
    etf_geography_exposure_monthly, geography_exposure_summary = build_etf_geography_exposure_monthly(
        current_characteristics, etf, geography,
    )
    etf_industry_exposure_monthly = build_etf_industry_exposure_monthly(
        current_characteristics, etf, industry,
    )

    tables = {
        "benchmark": benchmark,
        "etf": etf,
        "month": month,
        "geography": geography,
        "industry": industry,
        "macro_indicator": macro_indicator,
        "macro_series": macro_series,
        "etf_market_monthly": etf_market_monthly,
        "etf_metric_monthly": etf_metric_monthly,
        "macro_observation_monthly": macro_observation_monthly,
        "etf_geography_exposure_monthly": etf_geography_exposure_monthly,
        "etf_industry_exposure_monthly": etf_industry_exposure_monthly,
    }
    validate_relational_tables(tables)

    summary = {
        "geography_alias_cells_remapped_in_characteristics": characteristics_remaps,
        "geography_rows_remapped_in_macro_source": macro_remaps,
        "duplicate_geography_rows_removed": len(alias_names_in_characteristics | alias_names_in_macro),
        "macro_series_rows_removed": macro_series_summary["rows_removed"],
        "macro_observation_rows_deduplicated": macro_observation_summary["rows_deduplicated"],
        "macro_observation_series_id_rows_remapped": macro_observation_summary["rows_with_series_id_remapped"],
        "etf_geography_exposure_duplicate_rows_aggregated": geography_exposure_summary["rows_aggregated"],
    }
    return tables, summary

