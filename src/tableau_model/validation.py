from __future__ import annotations

import pandas as pd


def validate_relational_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Validate key constraints in the relational export."""
    geography = tables["geography"]
    macro_series = tables["macro_series"]
    macro_observation_monthly = tables["macro_observation_monthly"]
    etf_geography_exposure_monthly = tables["etf_geography_exposure_monthly"]

    if geography["geography_id"].duplicated().any():
        raise ValueError("geography.geography_id must be unique.")

    if geography["geography_name"].isin({"US", "Korea (South)"}).any():
        raise ValueError("geography contains duplicate non-canonical country names.")

    valid_geography_ids = set(geography["geography_id"].dropna().astype(int))
    macro_series_geographies = set(macro_series["geography_id"].dropna().astype(int))
    exposure_geographies = set(etf_geography_exposure_monthly["geography_id"].dropna().astype(int))

    if not macro_series_geographies.issubset(valid_geography_ids):
        raise ValueError("macro_series contains invalid geography_id values.")

    if not exposure_geographies.issubset(valid_geography_ids):
        raise ValueError("etf_geography_exposure_monthly contains invalid geography_id values.")

    if etf_geography_exposure_monthly.duplicated(subset=["etf_id", "geography_id"]).any():
        raise ValueError("etf_geography_exposure_monthly contains duplicate keys.")

    if macro_observation_monthly.duplicated(subset=["series_id", "month_id"]).any():
        raise ValueError("macro_observation_monthly contains duplicate series_id + month_id keys.")

