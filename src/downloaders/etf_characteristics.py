from __future__ import annotations

import json
import re
import warnings
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import ISHARES_BASE_URL, ISHARES_ETF_LIST_URL, ISHARES_MAX_FUNDS, ISHARES_USER_AGENT
from downloaders.ishares_classification import (
    classify_active_vs_index as _classify_active_vs_index,
    classify_developed_or_emerging as _classify_developed_or_emerging,
    classify_geographic_scope as _classify_geographic_scope,
    classify_primary_region as _classify_primary_region,
    classify_sector_focus as _classify_sector_focus,
    classify_size_focus as _classify_size_focus,
    classify_style_focus as _classify_style_focus,
    map_asset_class as _map_asset_class,
)
from downloaders.ishares_config import (
    COMPONENT_MARKERS,
    FINAL_COLUMNS,
    LIST_COLUMNS,
    PRODUCT_PATH_PATTERN,
)


def _make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": ISHARES_USER_AGENT})
    return session


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _join_text(*values: object) -> str:
    """Join text fragments and normalize whitespace."""
    return _clean_text(" ".join(str(value or "") for value in values))



def _component_data_points(components: dict[str, dict[str, object]], component_id: str, container_name: str) -> dict[str, dict[str, object]]:
    component = components.get(component_id, {})
    return component.get("containersByNameMap", {}).get(container_name, {}).get("dataPointsByNameMap", {})


def _component_value(
    components: dict[str, dict[str, object]],
    component_id: str,
    container_name: str,
    data_point: str,
    *,
    prefer_formatted: bool = False,
) -> object:
    data_points = _component_data_points(components, component_id, container_name)
    item = data_points.get(data_point, {})
    if not item:
        return None
    if prefer_formatted:
        return item.get("formattedValue") or item.get("value")
    return item.get("value") if item.get("value") is not None else item.get("formattedValue")


def _parse_required_components(html: str) -> tuple[dict[str, dict[str, object]], str]:
    soup = BeautifulSoup(html, "html.parser")
    components: dict[str, dict[str, object]] = {}
    description = ""

    for tag in soup.find_all("walrus-render-on-client"):
        props = tag.get("componentprops", "")
        if not props.startswith("{"):
            continue

        if not description and "fund-description" in props and '"content"' in props:
            try:
                content_block = json.loads(props)
                description_items = content_block.get("content", {}).get("fund-description", [])
                if description_items:
                    description = _clean_text(description_items[0].get("text"))
            except json.JSONDecodeError:
                pass
            continue

        for component_id, marker in COMPONENT_MARKERS.items():
            if component_id in components:
                continue
            if marker not in props:
                continue
            try:
                components[component_id] = json.loads(props)
            except json.JSONDecodeError:
                warnings.warn(f"Could not parse {component_id} component payload.")
            break

    return components, description


def _build_master_universe(session: requests.Session, timeout: int) -> pd.DataFrame:
    response = session.get(ISHARES_ETF_LIST_URL, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    records: list[dict[str, object]] = []

    for row in soup.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) != 10:
            continue

        ticker_link = cells[0].find("a", href=True)
        if ticker_link is None:
            continue

        href = ticker_link["href"]
        match = PRODUCT_PATH_PATTERN.search(href)
        if match is None:
            continue

        values = [" ".join(cell.get_text(" ", strip=True).split()) for cell in cells]
        if not values[0].isupper():
            continue

        record = dict(zip(LIST_COLUMNS, values))
        record["product_id"] = match.group("product_id")
        record["product_slug"] = match.group("slug")
        record["product_url"] = urljoin(ISHARES_BASE_URL, href)
        records.append(record)

    universe = pd.DataFrame(records).drop_duplicates(subset=["ticker"]).reset_index(drop=True)
    if ISHARES_MAX_FUNDS is not None:
        universe = universe.head(ISHARES_MAX_FUNDS).copy()
    return universe


def _get_exposure_pairs(
    components: dict[str, dict[str, object]],
    container_name: str,
    *,
    subcontainer_name: str | None = None,
    exclude_labels: set[str] | None = None,
) -> list[tuple[str, float]]:
    exposure = components.get("exposureBreakdowns", {}).get("containersByNameMap", {}).get(container_name, {})
    if not exposure:
        return []

    if subcontainer_name is not None:
        data_points = exposure.get("subContainersByNameMap", {}).get(subcontainer_name, {}).get("dataPointsByNameMap", {})
    else:
        data_points = exposure.get("dataPointsByNameMap", {})

    labels = data_points.get("type", {}).get("value", []) or []
    weights = data_points.get("fund", {}).get("value", []) or []
    exclude = exclude_labels or set()

    pairs: list[tuple[str, float]] = []
    for label, weight in zip(labels, weights):
        clean_label = _clean_text(label)
        if not clean_label or clean_label in exclude:
            continue
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            continue
        if numeric_weight <= 0:
            continue
        pairs.append((clean_label, numeric_weight))
    return pairs


def _collapse_exposures(pairs: list[tuple[str, float]], prefix: str) -> dict[str, object]:
    other_pairs = pairs[3:]

    record: dict[str, object] = {}
    for index in range(3):
        name_key = f"{prefix}_{index + 1}"
        weight_key = f"{prefix}_{index + 1}_weight"
        record[name_key] = pairs[index][0] if index < len(pairs) else None
        record[weight_key] = pairs[index][1] if index < len(pairs) else None

    suffix = "countries" if "country" in prefix else "industries"
    if other_pairs:
        other_name_key = f"other_{suffix}"
        other_weight_key = f"other_{suffix}_weight"
        record[other_name_key] = "; ".join(name for name, _ in other_pairs)
        record[other_weight_key] = sum(weight for _, weight in other_pairs)
    else:
        record[f"other_{suffix}"] = None
        record[f"other_{suffix}_weight"] = None

    return record



def _fetch_detail_record(
    session: requests.Session,
    base_row: dict[str, object],
    timeout: int,
) -> dict[str, object]:
    product_url = str(base_row["product_url"])
    try:
        response = session.get(product_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        warnings.warn(f"Could not download iShares page {product_url}: {exc}")
        return {column: None for column in FINAL_COLUMNS}

    components, description = _parse_required_components(response.text)
    value = lambda component, container, point, formatted=False: _component_value(
        components,
        component,
        container,
        point,
        prefer_formatted=formatted,
    )
    benchmark_index = _clean_text(value("keyFundFacts", "default", "indexSeriesName", True))
    product_slug_text = str(base_row.get("product_slug", "")).replace("-", " ")
    headline_text = _join_text(base_row.get("etf_name", ""), benchmark_index, product_slug_text)
    text_for_classification = _join_text(base_row.get("etf_name", ""), description, benchmark_index, product_slug_text)
    raw_asset_class = _clean_text(value("keyFundFacts", "default", "assetClass", True))
    asset_class = _map_asset_class(raw_asset_class)

    country_pairs = _get_exposure_pairs(
        components,
        "geography",
        subcontainer_name="countries",
        exclude_labels={"Cash and/or Derivatives"},
    )
    industry_pairs = _get_exposure_pairs(
        components,
        "sector",
        exclude_labels={"Cash and/or Derivatives"},
    )

    primary_region = _classify_primary_region(text_for_classification, country_pairs)
    geographic_scope = _classify_geographic_scope(text_for_classification, primary_region, asset_class)
    developed_or_emerging = _classify_developed_or_emerging(text_for_classification, primary_region, country_pairs)
    style_focus = _classify_style_focus(headline_text, text_for_classification, asset_class)
    number_of_holdings = value("fundamentalsAndRisk", "default", "numHoldings")
    size_focus = _classify_size_focus(text_for_classification, asset_class, number_of_holdings)
    sector_focus = _classify_sector_focus(text_for_classification, raw_asset_class, geographic_scope, style_focus)
    active_vs_index = _classify_active_vs_index(text_for_classification, str(base_row.get("etf_name", "")), raw_asset_class, benchmark_index)

    country_record = _collapse_exposures(country_pairs, "top_country")
    industry_record = _collapse_exposures(industry_pairs, "top_industry")

    record = {
        "ticker": base_row.get("ticker"),
        "etf_name": _clean_text(value("fundHeader", "fundNav", "fundName", True) or base_row.get("etf_name")),
        "description": description or None,
        "provider": "iShares / BlackRock",
        "asset_class": asset_class,
        "geographic_scope": geographic_scope,
        "primary_region": primary_region,
        "developed_or_emerging": developed_or_emerging,
        "style_focus": style_focus,
        "size_focus": size_focus,
        "sector_focus": sector_focus,
        "active_vs_index": active_vs_index,
        "benchmark_index": benchmark_index or None,
        "expense_ratio": value("fundHeader", "fees", "expr") or base_row.get("net_expense_ratio_pct") or base_row.get("gross_expense_ratio_pct"),
        "net_assets_usd": value("keyFundFacts", "default", "totalNetAssetsFundLevel") or base_row.get("net_assets_usd_list"),
        "bid_ask_spread_30d": value("keyFundFacts", "default", "thirtyDayMedianBidAskSpread"),
        "number_of_holdings": number_of_holdings,
        "pe_ratio": value("fundamentalsAndRisk", "default", "priceEarnings"),
        "pb_ratio": value("fundamentalsAndRisk", "default", "priceBook"),
        "standard_deviation_3y": value("fundamentalsAndRisk", "default", "standardDeviation3Yr"),
        "equity_beta_3y": value("fundamentalsAndRisk", "default", "beta3Yr"),
    }
    record.update(country_record)
    record.update(industry_record)
    return record


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [column for column in FINAL_COLUMNS if column.endswith("_weight")] + [
        "expense_ratio",
        "net_assets_usd",
        "bid_ask_spread_30d",
        "number_of_holdings",
        "pe_ratio",
        "pb_ratio",
        "standard_deviation_3y",
        "equity_beta_3y",
    ]
    normalized = df.copy()
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def download_etf_characteristics(timeout: int = 30) -> pd.DataFrame:
    session = _make_session()
    universe = _build_master_universe(session=session, timeout=timeout)

    records: list[dict[str, object]] = []
    for row in universe.to_dict(orient="records"):
        records.append(_fetch_detail_record(session=session, base_row=row, timeout=timeout))

    characteristics = pd.DataFrame(records)
    characteristics = _normalize_dataframe(characteristics)
    return characteristics[FINAL_COLUMNS].sort_values("ticker").reset_index(drop=True)


