from __future__ import annotations

from pathlib import Path


def _fields(text: str) -> list[str]:
    """Split compact pipe-delimited column definitions."""
    return [value.strip() for value in text.split("|")]


MIN_MONTH_ID = 201601
OTHER_COUNTRIES_ID = 999

MARKET_COLUMNS = _fields(
    "etf_id|month_id|open|high|low|close|adjusted_close|volume|trading_days|monthly_return|"
    "log_return|momentum_12m|cumulative_return_index|running_peak_index|drawdown"
)
ETF_METRIC_COLUMNS = _fields(
    "etf_id|expense_ratio|net_assets_usd|bid_ask_spread_30d|number_of_holdings|pe_ratio|"
    "pb_ratio|standard_deviation_3y|equity_beta_3y"
)

GEOGRAPHY_ALIAS_TO_CANONICAL = {
    "US": "United States",
    "Korea (South)": "South Korea",
    "Asia ex Japan": "Asia Pacific ex-Japan",
    "Asia ex-Japan": "Asia Pacific ex-Japan",
    "Global ex US": "Global ex-US",
    "Developed ex US": "Developed ex-US",
}

GEOGRAPHY_ROWS_TEXT = (
    "1|Australia|Asia-Pacific;2|Austria|Europe;3|Belgium|Europe;4|Brazil|Latin America;"
    "5|Canada|North America;6|Chile|Latin America;7|China|Asia-Pacific;8|Denmark|Europe;"
    "9|Finland|Europe;10|France|Europe;11|Germany|Europe;12|Hong Kong|Asia-Pacific;"
    "13|India|Asia-Pacific;14|Indonesia|Asia-Pacific;15|Ireland|Europe;16|Israel|Middle East & Africa;"
    "17|Italy|Europe;18|Japan|Asia-Pacific;20|Mexico|Latin America;21|Netherlands|Europe;"
    "22|New Zealand|Asia-Pacific;23|Norway|Europe;25|Peru|Latin America;26|Poland|Europe;"
    "27|Saudi Arabia|Middle East & Africa;28|Singapore|Asia-Pacific;29|South Africa|Middle East & Africa;"
    "30|South Korea|Asia-Pacific;31|Spain|Europe;32|Sweden|Europe;33|Switzerland|Europe;"
    "34|Taiwan|Asia-Pacific;35|Turkey|Middle East & Africa;37|United Kingdom|Europe;38|United States|North America"
)

GEOGRAPHY_ROWS = [
    (int(geography_id), name, parent_region)
    for geography_id, name, parent_region in (
        line.split("|") for line in GEOGRAPHY_ROWS_TEXT.split(";")
    )
]

GEOGRAPHY_REFERENCE = {
    name: {"geography_id": geography_id, "parent_region": parent_region}
    for geography_id, name, parent_region in GEOGRAPHY_ROWS
}

INDUSTRY_CLASSIFICATION_PATH = Path(__file__).with_name("industry_classification.csv")

