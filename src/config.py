from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TABLEAU_INPUT_DIR = BASE_DIR / "tableau_input"
OUTPUT_DIR = TABLEAU_INPUT_DIR

RAW_ETF_UNIVERSE_PATH = RAW_DIR / "etf_universe_ishares.csv"
RAW_ETF_PRICES_PATH = RAW_DIR / "etf_prices_yahoo.csv"
RAW_MACRO_PATH = RAW_DIR / "macro_indicators_fred.csv"

PROCESSED_ETF_CLASSIFIED_PATH = PROCESSED_DIR / "etf_universe_classified.csv"
PROCESSED_ETF_MONTHLY_RETURNS_PATH = PROCESSED_DIR / "etf_monthly_returns.csv"
PROCESSED_MACRO_QUALITY_PATH = PROCESSED_DIR / "macro_quality.csv"
PROCESSED_MACRO_MONTHLY_PATH = PROCESSED_DIR / "macro_monthly.csv"

TABLEAU_RELATIONAL_FINAL_PATH = OUTPUT_DIR / "tableau_data_final.xlsx"
TABLEAU_RELATIONAL_OUTPUT_PATH = TABLEAU_RELATIONAL_FINAL_PATH
PORTFOLIO_OPTIMIZER_PRECOMPUTED_PATH = OUTPUT_DIR / "portfolio_optimizer_precomputed.xlsx"

START_DATE = "2016-01-01"
END_DATE = None
ISHARES_ETF_LIST_URL = "https://www.ishares.com/us/products/etf-investments"
ISHARES_BASE_URL = "https://www.ishares.com"
ISHARES_USER_AGENT = "BI-course-project/1.0"
ISHARES_MAX_FUNDS = None
YAHOO_BATCH_SIZE = 25
YAHOO_TIMEOUT = 30
MACRO_MAX_WORKERS = 4

MACRO_INDICATOR_SPECS = [
    {"indicator_name": name, "unit": unit, "series_pattern": pattern}
    for name, unit, pattern in [
        ("Inflation", "Percent", "CPALTT01{country_code}M657N"),
        ("Unemployment", "Percent", "LRUNTTTT{country_code}M156S"),
        ("Interest Rate", "Percent", "IR3TIB01{country_code}M156N"),
        ("USD Exchange Rate", "Exchange Rate", "CCUSMA02{country_code}M618N"),
    ]
]

MACRO_COUNTRY_CODE_MAP = dict(
    Australia="AU", Austria="AT", Belgium="BE", Brazil="BR", Canada="CA", Chile="CL", China="CN",
    Denmark="DK", Finland="FI", France="FR", Germany="DE", India="IN", Indonesia="ID", Israel="IL",
    Italy="IT", Japan="JP", Kuwait="KW", Malaysia="MY", Mexico="MX", Netherlands="NL", Norway="NO",
    Philippines="PH", Poland="PL", Qatar="QA", Singapore="SG", Spain="ES", Sweden="SE", Switzerland="CH",
    Taiwan="TW", Thailand="TH", Turkey="TR", US="US",
    **{
        "Hong Kong": "HK",
        "New Zealand": "NZ",
        "Saudi Arabia": "SA",
        "South Africa": "ZA",
        "South Korea": "KR",
        "United Kingdom": "GB",
    },
)

MACRO_SPECIAL_SERIES = {
    "US": [
        {"series_id": "CPALTT01USM657N", "country_or_region": "US", "indicator_name": "Inflation", "unit": "Percent"},
        {"series_id": "LRUNTTTTUSM156S", "country_or_region": "US", "indicator_name": "Unemployment", "unit": "Percent"},
        {"series_id": "FEDFUNDS", "country_or_region": "US", "indicator_name": "Interest Rate", "unit": "Percent"},
    ],
}
MACRO_EXCLUDED_PRIMARY_REGIONS = {
    "Asia Pacific ex-Japan", "Developed ex-US", "Emerging Markets", "Europe", "Global",
    "Global ex-US", "Latin America", "No information", "Other",
}
