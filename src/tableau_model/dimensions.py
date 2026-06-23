from __future__ import annotations

from typing import Any

import pandas as pd

from tableau_model.constants import (
    GEOGRAPHY_REFERENCE,
    INDUSTRY_CLASSIFICATION_PATH,
    OTHER_COUNTRIES_ID,
)
from tableau_model.helpers import non_blank_values


def build_benchmark_table(characteristics: pd.DataFrame) -> pd.DataFrame:
    """Build the benchmark dimension."""
    benchmark = (
        characteristics[["benchmark_index"]]
        .dropna()
        .drop_duplicates()
        .sort_values("benchmark_index")
        .reset_index(drop=True)
    )
    benchmark["benchmark_id"] = range(1, len(benchmark) + 1)
    benchmark["benchmark_name"] = benchmark["benchmark_index"]
    return benchmark[["benchmark_id", "benchmark_name"]]


def build_etf_table(characteristics: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Build the ETF dimension with stable descriptive attributes only."""
    benchmark_lookup = benchmark.rename(columns={"benchmark_name": "benchmark_index"})
    etf = characteristics.merge(benchmark_lookup, on="benchmark_index", how="left")
    etf = etf.sort_values("ticker").reset_index(drop=True)
    etf["etf_id"] = range(1, len(etf) + 1)
    etf["provider_name"] = etf["provider"]
    columns = [
        "etf_id",
        "ticker",
        "etf_name",
        "provider_name",
        "benchmark_id",
        "asset_class",
        "geographic_scope",
        "developed_or_emerging",
        "style_focus",
        "size_focus",
        "sector_focus",
        "active_vs_index",
    ]
    return etf[columns]


def build_month_table(
    etf_market_source: pd.DataFrame,
    macro_monthly_source: pd.DataFrame,
    snapshot_month_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build the shared month dimension."""
    month_values = pd.concat(
        [
            etf_market_source[["month_end"]],
            macro_monthly_source[["month_end"]],
            pd.DataFrame({"month_end": [snapshot_month_end]}),
        ],
        ignore_index=True,
    )
    month_values["month_end"] = pd.to_datetime(month_values["month_end"], errors="coerce")
    month = month_values.dropna().drop_duplicates().sort_values("month_end").reset_index(drop=True)
    month["month_id"] = month["month_end"].dt.strftime("%Y%m").astype(int)
    month["year"] = month["month_end"].dt.year
    month["month_number"] = month["month_end"].dt.month
    month["quarter"] = "Q" + month["month_end"].dt.quarter.astype(str)
    month["year_month_label"] = month["month_end"].dt.strftime("%Y-%m")
    return month[["month_id", "month_end", "year", "month_number", "quarter", "year_month_label"]]


def build_geography_table(characteristics: pd.DataFrame, macro_monthly_source: pd.DataFrame) -> pd.DataFrame:
    """Build the country geography dimension used by macro and ETF exposures."""
    names: set[str] = set()
    names.update(non_blank_values(macro_monthly_source.get("country_or_region")))
    for column in ("top_country_1", "top_country_2", "top_country_3"):
        names.update(non_blank_values(characteristics.get(column)))

    if has_other_countries_exposure(characteristics):
        names.add("Other Countries")

    fallback_id = max(item["geography_id"] for item in GEOGRAPHY_REFERENCE.values()) + 1
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        if name == "Other Countries":
            rows.append(_geography_row(OTHER_COUNTRIES_ID, name, "Other / Global"))
            continue

        mapping = GEOGRAPHY_REFERENCE.get(name)
        if mapping is None:
            rows.append(_geography_row(fallback_id, name, "Other / Global"))
            fallback_id += 1
            continue

        rows.append(_geography_row(mapping["geography_id"], name, mapping["parent_region"]))

    geography = pd.DataFrame(rows).sort_values("geography_id").reset_index(drop=True)
    return geography[["geography_id", "geography_name", "parent_region"]]


def build_industry_table(characteristics: pd.DataFrame) -> pd.DataFrame:
    """Build the industry dimension from ETF exposure labels."""
    names: set[str] = set()
    for column in ("top_industry_1", "top_industry_2", "top_industry_3"):
        names.update(non_blank_values(characteristics.get(column)))

    if characteristics.get("other_industries_weight") is not None:
        if characteristics["other_industries_weight"].notna().any():
            names.add("Other Industries")

    industry = pd.DataFrame({"industry_name": sorted(names)}).reset_index(drop=True)
    industry["industry_id"] = range(1, len(industry) + 1)
    classification = load_industry_classification()
    industry["canonical_industry_label"] = industry["industry_name"].map(
        lambda name: classification.get(str(name), (str(name), "Other Industries"))[0]
    )
    industry["industry_cluster"] = industry["industry_name"].map(
        lambda name: classification.get(str(name), (str(name), "Other Industries"))[1]
    )

    if industry["canonical_industry_label"].isna().any() or industry["industry_cluster"].isna().any():
        raise ValueError("Industry classification produced null values.")

    return industry[["industry_id", "industry_name", "canonical_industry_label", "industry_cluster"]]


def build_macro_indicator_table(macro_monthly_source: pd.DataFrame) -> pd.DataFrame:
    """Build the macro indicator lookup."""
    indicator = (
        macro_monthly_source[["indicator_name", "unit"]]
        .dropna(subset=["indicator_name"])
        .drop_duplicates(subset=["indicator_name"], keep="first")
        .sort_values("indicator_name")
        .reset_index(drop=True)
    )
    indicator["indicator_id"] = range(1, len(indicator) + 1)
    indicator["default_unit"] = indicator["unit"]
    return indicator[["indicator_id", "indicator_name", "default_unit"]]


def build_macro_series_table(
    macro_monthly_source: pd.DataFrame,
    geography: pd.DataFrame,
    macro_indicator: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    """Build macro series metadata and collapse duplicate canonical series."""
    geography_lookup = geography.rename(columns={"geography_name": "country_or_region"})
    indicator_lookup = macro_indicator.rename(columns={"default_unit": "unit"})
    series = (
        macro_monthly_source[["series_id", "country_or_region", "indicator_name", "frequency_flag", "unit"]]
        .dropna(subset=["series_id", "country_or_region", "indicator_name"])
        .drop_duplicates(subset=["series_id"])
        .merge(geography_lookup[["geography_id", "country_or_region"]], on="country_or_region", how="left")
        .merge(indicator_lookup[["indicator_id", "indicator_name"]], on="indicator_name", how="left")
        .sort_values("series_id")
        .reset_index(drop=True)
    )
    series["source_name"] = "FRED"
    keys = ["geography_id", "indicator_id", "frequency_flag", "source_name", "unit"]
    series = series.sort_values(keys + ["series_id"]).reset_index(drop=True)
    series["canonical_series_id"] = series.groupby(keys)["series_id"].transform("first")
    series_id_remap = dict(zip(series["series_id"].astype(str), series["canonical_series_id"].astype(str)))
    deduplicated = (
        series.drop_duplicates(subset=["canonical_series_id"])
        .assign(series_id=lambda frame: frame["canonical_series_id"].astype(str))
        .reset_index(drop=True)
    )
    columns = ["series_id", "geography_id", "indicator_id", "frequency_flag", "source_name", "unit"]
    return deduplicated[columns], series_id_remap, {"rows_removed": len(series) - len(deduplicated)}


def load_industry_classification() -> dict[str, tuple[str, str]]:
    """Load the original-label to canonical industry lookup."""
    frame = pd.read_csv(INDUSTRY_CLASSIFICATION_PATH)
    return dict(
        zip(
            frame["industry_name"].astype(str),
            zip(frame["canonical_industry_label"].astype(str), frame["industry_cluster"].astype(str)),
        )
    )


def _geography_row(geography_id: int, geography_name: str, parent_region: str) -> dict[str, Any]:
    """Build one geography dimension row."""
    return {"geography_id": geography_id, "geography_name": geography_name, "parent_region": parent_region}


def has_other_countries_exposure(characteristics: pd.DataFrame) -> bool:
    """Check whether at least one ETF has a positive residual country exposure."""
    return any(
        (weight := compute_other_countries_weight(record)) is not None and weight > 0
        for record in characteristics.to_dict(orient="records")
    )


def compute_other_countries_weight(record: dict[str, Any]) -> float | None:
    """Compute residual country weight outside the reported top-3 countries."""
    weights = [
        _to_float_or_none(record.get("top_country_1_weight")),
        _to_float_or_none(record.get("top_country_2_weight")),
        _to_float_or_none(record.get("top_country_3_weight")),
    ]
    available_weights = [weight for weight in weights if weight is not None]
    if not available_weights:
        return None

    total_scale = 100.0 if any(abs(weight) > 1.5 for weight in available_weights) else 1.0
    return max(total_scale - sum(available_weights), 0.0)


def _to_float_or_none(value: Any) -> float | None:
    """Convert a possibly missing numeric-like cell to float."""
    if pd.isna(value):
        return None
    return float(value)

