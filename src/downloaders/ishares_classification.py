from __future__ import annotations

import re

import pandas as pd

from downloaders.ishares_config import (
    ASSET_CLASS_MAP,
    COUNTRY_GROUP_KEYWORDS,
    DEVELOPED_COUNTRIES,
    EMERGING_COUNTRIES,
    PRIMARY_REGION_RULES,
    SECTOR_RULES,
    SINGLE_COUNTRY_PRIMARY_REGIONS,
    SIZE_RULES,
    STYLE_RULES,
)


def first_match(text: str, rules: list[tuple[str, list[str]]], default: str | None = None) -> str | None:
    """Return the first label whose regex rule matches the text."""
    for label, patterns in rules:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            return label
    return default


def map_asset_class(raw_asset_class: str) -> str:
    """Normalize iShares asset class labels to project labels."""
    return ASSET_CLASS_MAP.get(raw_asset_class, "Other")


def classify_primary_region(text: str, country_pairs: list[tuple[str, float]]) -> str:
    """Classify the ETF primary region from text and country exposure."""
    for country, weight in sorted(country_pairs, key=lambda item: item[1], reverse=True):
        if weight >= 50:
            return {"United States": "US", "Korea (South)": "South Korea"}.get(country, country)

    primary_region = first_match(text, PRIMARY_REGION_RULES)
    if primary_region is not None:
        return primary_region
    if country_pairs:
        top_country = sorted(country_pairs, key=lambda item: item[1], reverse=True)[0][0]
        return {"United States": "US", "Korea (South)": "South Korea"}.get(top_country, top_country)
    return "No information"


def classify_geographic_scope(text: str, primary_region: str, asset_class: str) -> str:
    """Classify geographic scope from name, benchmark, and description text."""
    if asset_class != "Equity":
        return "No information"
    if primary_region in SINGLE_COUNTRY_PRIMARY_REGIONS:
        return "Single Country"
    if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in COUNTRY_GROUP_KEYWORDS):
        return "Country Group"
    if primary_region in {"Global", "Global ex-US", "Developed ex-US", "Emerging Markets"}:
        return "Global"
    if primary_region in {"Other", "No information"}:
        return "No information"
    return "Regional"


def classify_developed_or_emerging(
    text: str,
    primary_region: str,
    country_pairs: list[tuple[str, float]],
) -> str:
    """Classify developed, emerging, mixed, or unknown exposure."""
    if primary_region == "Emerging Markets":
        return "Emerging"
    if primary_region in {"Developed ex-US", "Europe", "Japan", "US"}:
        return "Developed"
    if primary_region in {"Global", "Global ex-US", "Asia Pacific ex-Japan"}:
        return "Mixed"
    if re.search(r"\bfrontier\b", text, flags=re.IGNORECASE):
        return "Frontier"

    countries = {country for country, _ in country_pairs}
    has_developed = bool(countries & DEVELOPED_COUNTRIES)
    has_emerging = bool(countries & EMERGING_COUNTRIES)
    if has_developed or has_emerging:
        return "Mixed" if has_developed and has_emerging else "Developed" if has_developed else "Emerging"
    return "No information"


def classify_style_focus(headline_text: str, full_text: str, asset_class: str) -> str:
    """Classify the ETF style focus."""
    if asset_class != "Equity":
        return "Other"
    style = first_match(headline_text, STYLE_RULES) or first_match(full_text, STYLE_RULES)
    if style:
        return style
    if re.search(r"\bcore\b|\bbroad\b|\bmarket\b|\bindex\b", full_text, flags=re.IGNORECASE):
        return "Core"
    return "Other"


def classify_size_focus(text: str, asset_class: str, number_of_holdings: object) -> str | None:
    """Classify equity market-cap focus."""
    if asset_class != "Equity":
        return None
    size = first_match(text, SIZE_RULES)
    if size:
        return size
    if pd.notna(number_of_holdings):
        try:
            return "Broad/Multi-Cap" if float(number_of_holdings) >= 500 else "Large/Mid Cap"
        except (TypeError, ValueError):
            pass
    return "Broad/Multi-Cap"


def classify_sector_focus(text: str, asset_class_raw: str, geographic_scope: str, style_focus: str) -> str:
    """Classify sector or broad-market focus."""
    if asset_class_raw == "Real Estate":
        return "Real Estate"
    sector = first_match(text, SECTOR_RULES)
    if sector and (geographic_scope not in {"Global", "Regional", "Single Country"} or style_focus == "Thematic"):
        return sector
    if geographic_scope in {"Global", "Regional", "Single Country", "Country Group"}:
        return "Broad Market"
    return sector or "Other"


def classify_active_vs_index(text: str, etf_name: str, raw_asset_class: str, benchmark_index: str) -> str:
    """Classify active versus index management."""
    combined = f"{etf_name} {text} {raw_asset_class}".lower()
    if "active" in raw_asset_class.lower() or "actively managed" in combined or "active etf" in combined:
        return "Active"
    if "seeks to track" in combined or benchmark_index:
        return "Index"
    return "Active"

