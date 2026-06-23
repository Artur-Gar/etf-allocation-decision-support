from __future__ import annotations


REQUESTED_TICKERS = ["DIVB", "DVY", "DVYA", "HDV", "IDV", "IVVW", "PFF"]
SCENARIO_ID = "S001"
SCENARIO_NAME = "Max Sharpe Portfolio"
OPTIMIZATION_OBJECTIVE = "Maximize Sharpe Ratio"
LOOKBACK_WINDOW = "Full History"
RISK_FREE_RATE = 0.03
MAX_ETF_WEIGHT = 0.30
LONG_ONLY = True
RANDOM_STATE = 42
NUM_SIMULATIONS = 50_000
MIN_PERIODS = 12
MIN_WEIGHT_TO_KEEP = 0.0001

REQUIRED_SHEETS = {
    "etf",
    "etf_market_monthly",
    "etf_metric_monthly",
    "etf_geography_exposure_monthly",
    "etf_industry_exposure_monthly",
    "geography",
    "industry",
}

