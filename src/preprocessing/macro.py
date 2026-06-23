from __future__ import annotations

import pandas as pd


def _fields(text: str) -> list[str]:
    """Split compact pipe-delimited column definitions."""
    return [value.strip() for value in text.split("|")]


QUALITY_COLUMNS = _fields(
    "country_or_region|indicator_name|series_id|unit|observations|median_gap_days|min_date|max_date|"
    "frequency_flag|is_usable_monthly|is_quarterly_fill_candidate"
)
MONTHLY_COLUMNS = _fields(
    "month_end|country_or_region|indicator_name|indicator_value|unit|series_id|frequency_flag|is_imputed|source_date"
)


def build_macro_quality(macro: pd.DataFrame | None) -> pd.DataFrame:
    """Summarize macro series frequency and usability for downstream analysis."""
    if macro is None or macro.empty:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    current = macro.copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce")
    current = current.dropna(subset=["date", "country_or_region", "indicator_name", "series_id"])
    if current.empty:
        return pd.DataFrame(columns=QUALITY_COLUMNS)

    current = current.sort_values(["country_or_region", "indicator_name", "series_id", "date"]).reset_index(drop=True)
    current["diff_days"] = current.groupby("series_id")["date"].diff().dt.days

    quality = (
        current.groupby(["country_or_region", "indicator_name", "series_id", "unit"], as_index=False)
        .agg(
            observations=("date", "size"),
            median_gap_days=("diff_days", "median"),
            min_date=("date", "min"),
            max_date=("date", "max"),
        )
        .sort_values(["country_or_region", "indicator_name", "series_id"])
        .reset_index(drop=True)
    )

    quality["frequency_flag"] = quality["median_gap_days"].apply(_classify_frequency)
    quality["is_usable_monthly"] = quality["frequency_flag"].eq("Monthly-ish")
    quality["is_quarterly_fill_candidate"] = quality["frequency_flag"].eq("Quarterly-ish")

    # A USD exchange rate is not meaningful for the U.S. itself.
    us_mask = (quality["country_or_region"] == "US") & (quality["indicator_name"] == "USD Exchange Rate")
    quality.loc[us_mask, ["is_usable_monthly", "is_quarterly_fill_candidate"]] = False

    return quality[QUALITY_COLUMNS]


def build_macro_monthly(
    macro: pd.DataFrame | None,
    series_quality: pd.DataFrame,
) -> pd.DataFrame:
    """Create a monthly Tableau-ready macro table from the raw downloaded series."""
    if macro is None or macro.empty or series_quality.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    quality = series_quality[
        series_quality["is_usable_monthly"] | series_quality["is_quarterly_fill_candidate"]
    ].copy()
    if quality.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    current = macro.copy()
    current["date"] = pd.to_datetime(current["date"], errors="coerce")
    current = current.dropna(subset=["date", "country_or_region", "indicator_name", "series_id", "indicator_value"])
    if current.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    merged = current.merge(
        quality[
            [
                "country_or_region",
                "indicator_name",
                "series_id",
                "unit",
                "frequency_flag",
                "is_usable_monthly",
                "is_quarterly_fill_candidate",
            ]
        ],
        on=["country_or_region", "indicator_name", "series_id", "unit"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    merged["month_end"] = merged["date"].dt.to_period("M").dt.to_timestamp("M")
    base_monthly = (
        merged.sort_values(["series_id", "date"])
        .groupby(
            [
                "country_or_region",
                "indicator_name",
                "series_id",
                "unit",
                "frequency_flag",
                "is_usable_monthly",
                "is_quarterly_fill_candidate",
                "month_end",
            ],
            as_index=False,
        )
        .agg(
            indicator_value=("indicator_value", "last"),
            source_date=("date", "max"),
        )
    )

    monthly_frames: list[pd.DataFrame] = []
    key_columns = ["country_or_region", "indicator_name", "series_id", "unit", "frequency_flag"]

    for _, group in base_monthly.groupby(key_columns, sort=False):
        group = group.sort_values("month_end").reset_index(drop=True)
        frequency_flag = group.at[0, "frequency_flag"]

        if frequency_flag == "Quarterly-ish":
            month_index = pd.date_range(
                start=group["month_end"].min(),
                end=group["month_end"].max(),
                freq="ME",
            )
            expanded = (
                group.set_index("month_end")[["indicator_value", "source_date"]]
                .reindex(month_index)
                .rename_axis("month_end")
                .reset_index()
            )
            expanded["indicator_value"] = expanded["indicator_value"].ffill()
            expanded["source_date"] = expanded["source_date"].ffill()
            expanded["is_imputed"] = ~expanded["month_end"].isin(group["month_end"])
            for column in key_columns:
                expanded[column] = group.at[0, column]
            expanded = expanded.dropna(subset=["indicator_value"])
            monthly_frames.append(expanded[MONTHLY_COLUMNS])
            continue

        direct = group.copy()
        direct["is_imputed"] = False
        monthly_frames.append(direct[MONTHLY_COLUMNS])

    if not monthly_frames:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    return (
        pd.concat(monthly_frames, ignore_index=True)
        .sort_values(["country_or_region", "indicator_name", "month_end"])
        .reset_index(drop=True)
    )


def _classify_frequency(days: float | int | None) -> str:
    """Convert median day gaps into a simple frequency label."""
    if pd.isna(days):
        return "Single point"
    if days <= 35:
        return "Monthly-ish"
    if days <= 100:
        return "Quarterly-ish"
    return "Irregular / sparse"

