from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import re
import threading
import warnings
import tqdm

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.errors import ParserError

from config import (
    MACRO_COUNTRY_CODE_MAP,
    MACRO_EXCLUDED_PRIMARY_REGIONS,
    MACRO_INDICATOR_SPECS,
    MACRO_MAX_WORKERS,
    MACRO_SPECIAL_SERIES,
)


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
FRED_SEARCH_URL = "https://fred.stlouisfed.org/searchresults/?st={query}"
MACRO_OUTPUT_COLUMNS = ["date", "country_or_region", "indicator_name", "indicator_value", "unit", "series_id"]
DIRECT_SERIES_ATTEMPTS = 3
FALLBACK_SERIES_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30
INDICATOR_SEARCH_HINTS = {
    "Inflation": "consumer price index inflation",
    "Unemployment": "unemployment rate",
    "Interest Rate": "short-term interest rate policy rate",
    "USD Exchange Rate": "exchange rate to u.s. dollar",
}
INDICATOR_MATCH_KEYWORDS = {
    "Inflation": ["consumer price index", "cpi", "inflation"],
    "Unemployment": ["unemployment"],
    "Interest Rate": ["interest rate", "policy rate", "immediate rates", "short-term"],
    "USD Exchange Rate": ["exchange rate", "u.s. dollar", "usd"],
}
SERIES_LINK_PATTERN = re.compile(r"^/series/(?P<series_id>[A-Z0-9]+)$")
_THREAD_LOCAL = threading.local()


def build_macro_series_definitions(characteristics: pd.DataFrame) -> list[dict[str, str]]:
    """Build FRED series definitions from the ETF primary-region universe."""
    if "primary_region" not in characteristics.columns:
        raise ValueError("ETF characteristics must contain a 'primary_region' column.")

    primary_regions = {
        str(value).strip()
        for value in characteristics["primary_region"].dropna().astype(str)
        if str(value).strip()
    }

    frames: list[dict[str, str]] = []
    unsupported_regions = sorted(
        region
        for region in primary_regions
        if region not in MACRO_COUNTRY_CODE_MAP
        and region not in MACRO_EXCLUDED_PRIMARY_REGIONS
    )
    if unsupported_regions:
        warnings.warn(
            "No macro mapping configured for these primary_region labels: "
            + ", ".join(unsupported_regions)
        )

    for region in sorted(primary_regions):
        if region in MACRO_SPECIAL_SERIES:
            frames.extend(MACRO_SPECIAL_SERIES[region])
            continue

        country_code = MACRO_COUNTRY_CODE_MAP.get(region)
        if country_code is None:
            continue

        for spec in MACRO_INDICATOR_SPECS:
            frames.append(
                {
                    "series_id": spec["series_pattern"].format(country_code=country_code),
                    "country_or_region": region,
                    "indicator_name": spec["indicator_name"],
                    "unit": spec["unit"],
                }
            )

    return frames


def download_macro_indicators(
    characteristics: pd.DataFrame,
    timeout: int = DEFAULT_TIMEOUT,
    max_workers: int = MACRO_MAX_WORKERS,
) -> pd.DataFrame:
    """Download monthly macro indicators from FRED for the ETF country universe."""
    macro_series = build_macro_series_definitions(characteristics)
    frames: list[pd.DataFrame] = []
    if not macro_series:
        return pd.DataFrame(columns=MACRO_OUTPUT_COLUMNS)

    worker_count = max(1, min(max_workers, len(macro_series)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(
            _download_macro_series_for_worker,
            macro_series,
            [timeout] * len(macro_series),
        )
        for current in tqdm.tqdm(iterator, total=len(macro_series), desc="Downloading macro indicators"):
            if current is None:
                continue
            frames.append(current[MACRO_OUTPUT_COLUMNS])

    if not frames:
        return pd.DataFrame(columns=MACRO_OUTPUT_COLUMNS)

    macro = pd.concat(frames, ignore_index=True)
    macro["date"] = pd.to_datetime(macro["date"]).dt.date
    return (
        macro.dropna(subset=["indicator_value"])
        .sort_values(["country_or_region", "indicator_name", "date"])
        .reset_index(drop=True)
    )


def _download_macro_series(
    *,
    session: requests.Session,
    series: dict[str, str],
    timeout: int,
) -> pd.DataFrame | None:
    """Download one macro series, trying the guessed ID before searching FRED."""
    last_error: Exception | None = None
    direct_result, last_error = _try_download_series_id(
        session=session,
        series=series,
        series_id=series["series_id"],
        timeout=timeout,
        attempts=DIRECT_SERIES_ATTEMPTS,
    )
    if direct_result is not None:
        return direct_result

    resolved_id = _resolve_series_id_from_search(session=session, series=series, timeout=timeout)
    if resolved_id and resolved_id != series["series_id"]:
        fallback_result, last_error = _try_download_series_id(
            session=session,
            series=series,
            series_id=resolved_id,
            timeout=timeout,
            attempts=FALLBACK_SERIES_ATTEMPTS,
        )
        if fallback_result is not None:
            return fallback_result

    if last_error is not None:
        warnings.warn(
            f"Could not download FRED series {series['series_id']} for "
            f"{series['country_or_region']} / {series['indicator_name']}: {last_error}"
        )
    return None


def _download_macro_series_for_worker(series: dict[str, str], timeout: int) -> pd.DataFrame | None:
    """Worker entrypoint used by the thread pool."""
    session = _get_thread_session()
    return _download_macro_series(session=session, series=series, timeout=timeout)


def _try_download_series_id(
    *,
    session: requests.Session,
    series: dict[str, str],
    series_id: str,
    timeout: int,
    attempts: int,
) -> tuple[pd.DataFrame | None, Exception | None]:
    """Try downloading one concrete FRED series ID."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            response = session.get(FRED_CSV_URL.format(series_id=series_id), timeout=timeout)
            response.raise_for_status()
            return _parse_macro_csv(response.text, series=series, series_id=series_id), None
        except requests.RequestException as exc:
            last_error = exc

    return None, last_error


def _get_thread_session() -> requests.Session:
    """Return one requests session per worker thread."""
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "BI-course-project/1.0"})
        _THREAD_LOCAL.session = session
    return session


def _parse_macro_csv(
    csv_text: str,
    *,
    series: dict[str, str],
    series_id: str,
) -> pd.DataFrame | None:
    """Parse a FRED CSV response into the standard macro schema."""
    try:
        current = pd.read_csv(StringIO(csv_text))
    except ParserError:
        warnings.warn(f"FRED series {series_id} did not return valid CSV data.")
        return None

    if current.shape[1] < 2:
        warnings.warn(f"FRED series {series_id} returned an unexpected format.")
        return None

    current.columns = ["date", "indicator_value"]
    current["indicator_value"] = pd.to_numeric(current["indicator_value"], errors="coerce")
    current["country_or_region"] = series["country_or_region"]
    current["indicator_name"] = series["indicator_name"]
    current["unit"] = series["unit"]
    current["series_id"] = series_id
    return current


def _resolve_series_id_from_search(
    *,
    session: requests.Session,
    series: dict[str, str],
    timeout: int,
) -> str | None:
    """Search FRED for a better matching series ID when the direct pattern is missing."""
    query_hint = INDICATOR_SEARCH_HINTS.get(series["indicator_name"])
    if not query_hint:
        return None

    query = f"{series['country_or_region']} {query_hint}"
    try:
        response = session.get(FRED_SEARCH_URL.format(query=requests.utils.quote(query)), timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None

    return _extract_best_series_id(
        html=response.text,
        country_or_region=series["country_or_region"],
        indicator_name=series["indicator_name"],
    )


def _extract_best_series_id(
    *,
    html: str,
    country_or_region: str,
    indicator_name: str,
) -> str | None:
    """Extract the most plausible series ID from a FRED search result page."""
    soup = BeautifulSoup(html, "html.parser")
    country_text = country_or_region.lower()
    keywords = INDICATOR_MATCH_KEYWORDS.get(indicator_name, [])

    candidates: list[tuple[int, str]] = []
    for link in soup.find_all("a", href=True):
        match = SERIES_LINK_PATTERN.match(link["href"])
        if not match:
            continue

        title = " ".join(link.get_text(" ", strip=True).split()).lower()
        score = sum(keyword in title for keyword in keywords)
        if country_text in title:
            score += 2
        if score == 0:
            continue

        candidates.append((score, match.group("series_id")))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]

