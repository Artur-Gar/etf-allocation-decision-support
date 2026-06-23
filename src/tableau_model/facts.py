from __future__ import annotations

from typing import Any

import pandas as pd

from tableau_model.constants import ETF_METRIC_COLUMNS, MARKET_COLUMNS
from tableau_model.dimensions import compute_other_countries_weight
from tableau_model.helpers import build_lookup, build_weight_row, clean_text


def build_etf_market_monthly(
    etf_market_source: pd.DataFrame,
    etf: pd.DataFrame,
    month: pd.DataFrame,
) -> pd.DataFrame:
    """Build the central ETF market fact table."""
    market = (
        etf_market_source.merge(etf[["etf_id", "ticker"]], on="ticker", how="left")
        .merge(month[["month_id", "month_end"]], on="month_end", how="left")
        .sort_values(["etf_id", "month_id"])
        .reset_index(drop=True)
    )
    market["running_peak_index"] = market.groupby("etf_id")["cumulative_return_index"].cummax()
    market["drawdown"] = market["cumulative_return_index"].div(market["running_peak_index"]).sub(1.0)
    return market[MARKET_COLUMNS]


def build_etf_metric_monthly(
    characteristics: pd.DataFrame,
    etf: pd.DataFrame,
) -> pd.DataFrame:
    """Build ETF metric rows from the latest available ETF snapshot."""
    metrics = characteristics.merge(etf[["etf_id", "ticker"]], on="ticker", how="left").copy()
    return metrics[ETF_METRIC_COLUMNS].sort_values("etf_id").reset_index(drop=True)


def build_macro_observation_monthly(
    macro_monthly_source: pd.DataFrame,
    macro_series: pd.DataFrame,
    month: pd.DataFrame,
    series_id_remap: dict[str, str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the monthly macro fact table."""
    current = macro_monthly_source.copy()
    original_series_id = current["series_id"].astype(str)
    current["series_id"] = original_series_id.map(lambda value: series_id_remap.get(value, value))
    rows_remapped = int((current["series_id"].astype(str) != original_series_id).sum())

    observation = (
        current.merge(macro_series[["series_id"]], on="series_id", how="left")
        .merge(month[["month_id", "month_end"]], on="month_end", how="left")
        .sort_values(["series_id", "month_id"])
        .reset_index(drop=True)
    )
    before_rows = len(observation)
    observation["source_date"] = pd.to_datetime(observation["source_date"], errors="coerce")
    observation["_imputed_rank"] = observation["is_imputed"].fillna(True).astype(int)
    observation = observation.sort_values(
        ["series_id", "month_id", "_imputed_rank", "source_date"],
        ascending=[True, True, True, False],
    )
    observation = observation.drop_duplicates(subset=["series_id", "month_id"], keep="first").reset_index(drop=True)
    observation = observation[["series_id", "month_id", "indicator_value", "is_imputed", "source_date"]]
    summary = {
        "rows_deduplicated": before_rows - len(observation),
        "rows_with_series_id_remapped": rows_remapped,
    }
    return observation, summary


def build_etf_geography_exposure_monthly(
    characteristics: pd.DataFrame,
    etf: pd.DataFrame,
    geography: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build ETF geography exposure rows from the latest snapshot."""
    rows = _build_exposure_rows(
        characteristics=characteristics,
        etf=etf,
        dimension=geography,
        dimension_name_column="geography_name",
        dimension_id_column="geography_id",
        top_prefix="top_country",
    )
    geography_lookup = build_lookup(geography, key="geography_name", value="geography_id")
    etf_lookup = build_lookup(etf, key="ticker", value="etf_id")
    for record in characteristics.to_dict(orient="records"):
        etf_id = etf_lookup.get(str(record.get("ticker", "")).strip())
        residual_weight = compute_other_countries_weight(record)
        if etf_id is not None and residual_weight is not None and residual_weight > 0:
            rows.extend(
                build_weight_row(
                    etf_id=etf_id,
                    dimension_id=geography_lookup.get("Other Countries"),
                    weight=residual_weight,
                    field_name="geography_id",
                )
            )

    exposure = pd.DataFrame(rows, columns=["etf_id", "geography_id", "weight"])
    if exposure.empty:
        return exposure, {"rows_aggregated": 0}

    before_rows = len(exposure)
    exposure = (
        exposure.groupby(["etf_id", "geography_id"], as_index=False)["weight"]
        .sum()
        .sort_values(["etf_id", "geography_id"])
        .reset_index(drop=True)
    )
    return exposure, {"rows_aggregated": before_rows - len(exposure)}


def build_etf_industry_exposure_monthly(
    characteristics: pd.DataFrame,
    etf: pd.DataFrame,
    industry: pd.DataFrame,
) -> pd.DataFrame:
    """Build ETF industry exposure rows from the latest snapshot."""
    rows = _build_exposure_rows(
        characteristics=characteristics,
        etf=etf,
        dimension=industry,
        dimension_name_column="industry_name",
        dimension_id_column="industry_id",
        top_prefix="top_industry",
    )
    industry_lookup = build_lookup(industry, key="industry_name", value="industry_id")
    etf_lookup = build_lookup(etf, key="ticker", value="etf_id")
    for record in characteristics.to_dict(orient="records"):
        etf_id = etf_lookup.get(str(record.get("ticker", "")).strip())
        if etf_id is not None and pd.notna(record.get("other_industries_weight")):
            rows.extend(
                build_weight_row(
                    etf_id=etf_id,
                    dimension_id=industry_lookup.get("Other Industries"),
                    weight=record.get("other_industries_weight"),
                    field_name="industry_id",
                )
            )

    exposure = pd.DataFrame(rows, columns=["etf_id", "industry_id", "weight"])
    return exposure.drop_duplicates().sort_values(["etf_id", "industry_id"]).reset_index(drop=True)


def _build_exposure_rows(
    *,
    characteristics: pd.DataFrame,
    etf: pd.DataFrame,
    dimension: pd.DataFrame,
    dimension_name_column: str,
    dimension_id_column: str,
    top_prefix: str,
) -> list[dict[str, Any]]:
    """Create top-1/top-2/top-3 ETF exposure rows for one dimension."""
    dimension_lookup = build_lookup(dimension, key=dimension_name_column, value=dimension_id_column)
    etf_lookup = build_lookup(etf, key="ticker", value="etf_id")
    rows: list[dict[str, Any]] = []

    for record in characteristics.to_dict(orient="records"):
        etf_id = etf_lookup.get(str(record.get("ticker", "")).strip())
        if etf_id is None:
            continue
        for index in range(1, 4):
            rows.extend(
                build_weight_row(
                    etf_id=etf_id,
                    dimension_id=dimension_lookup.get(clean_text(record.get(f"{top_prefix}_{index}"))),
                    weight=record.get(f"{top_prefix}_{index}_weight"),
                    field_name=dimension_id_column,
                )
            )
    return rows

