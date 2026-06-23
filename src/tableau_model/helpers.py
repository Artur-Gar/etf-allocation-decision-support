from __future__ import annotations

from typing import Any

import pandas as pd

from tableau_model.constants import GEOGRAPHY_ALIAS_TO_CANONICAL


def non_blank_values(series: pd.Series | None) -> set[str]:
    """Extract non-empty text values from a series-like object."""
    if series is None:
        return set()
    cleaned = series.dropna().astype(str).str.strip()
    return {value for value in cleaned if value}


def clean_text(value: Any) -> str:
    """Normalize a possibly missing text cell."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def canonicalize_geography_name(value: Any) -> Any:
    """Map duplicate geography aliases to their canonical names."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    return GEOGRAPHY_ALIAS_TO_CANONICAL.get(text, text) if text else text


def build_lookup(frame: pd.DataFrame, *, key: str, value: str) -> dict[str, Any]:
    """Build a simple dictionary lookup from a table."""
    return dict(zip(frame[key].astype(str), frame[value]))


def to_float_or_none(value: Any) -> float | None:
    """Convert a possibly missing numeric-like cell to float."""
    if pd.isna(value):
        return None
    return float(value)


def build_weight_row(
    *,
    etf_id: int,
    dimension_id: Any,
    weight: Any,
    field_name: str,
) -> list[dict[str, Any]]:
    """Create one exposure row when the dimension and weight are usable."""
    if dimension_id is None or pd.isna(weight):
        return []
    return [{"etf_id": etf_id, field_name: int(dimension_id), "weight": float(weight)}]

