from __future__ import annotations

import numpy as np
import pandas as pd


def build_optimizer_weights(
    *,
    used_universe: pd.DataFrame,
    selected_weights: pd.Series,
    metric_lookup: pd.DataFrame,
    scenario_id: str,
    portfolio_id: str,
    min_weight_to_keep: float,
) -> pd.DataFrame:
    """Build the selected-portfolio ETF weights table."""
    weights = selected_weights[selected_weights > min_weight_to_keep].copy()
    weights = weights / weights.sum()
    output = (
        used_universe[["etf_id", "ticker", "etf_name"]]
        .merge(weights.rename("weight"), left_on="ticker", right_index=True, how="inner")
        .merge(metric_lookup, on="etf_id", how="left")
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    output["scenario_id"] = scenario_id
    output["portfolio_id"] = portfolio_id
    output["rank"] = np.arange(1, len(output) + 1, dtype=int)
    output = output.rename(columns={"bid_ask_spread_30d": "bid_ask_spread"})
    columns = ["scenario_id", "portfolio_id", "etf_id", "ticker", "etf_name", "weight", "expense_ratio", "bid_ask_spread", "rank"]
    return output[columns].reset_index(drop=True)


def build_optimizer_kpis(
    *,
    optimizer_weights: pd.DataFrame,
    selected_portfolio: pd.Series,
    scenario_id: str,
    portfolio_id: str,
) -> pd.DataFrame:
    """Build the selected-portfolio KPI summary."""
    weighted_expense_ratio = weighted_nullable_sum(
        optimizer_weights["weight"],
        optimizer_weights.get("expense_ratio"),
    )
    weighted_bid_ask_spread = weighted_nullable_sum(
        optimizer_weights["weight"],
        optimizer_weights.get("bid_ask_spread"),
    )
    max_drawdown = float(selected_portfolio["max_drawdown"])

    return pd.DataFrame(
        [
            {
                "scenario_id": scenario_id,
                "portfolio_id": portfolio_id,
                "expected_return": float(selected_portfolio["expected_return"]),
                "expected_volatility": float(selected_portfolio["expected_volatility"]),
                "sharpe_ratio": float(selected_portfolio["sharpe_ratio"]),
                "max_drawdown": max_drawdown,
                "weighted_expense_ratio": weighted_expense_ratio,
                "weighted_bid_ask_spread": weighted_bid_ask_spread,
                "number_of_etfs": int(len(optimizer_weights)),
            }
        ]
    )


def build_optimizer_exposure(
    *,
    optimizer_weights: pd.DataFrame,
    etf_geography_exposure: pd.DataFrame,
    etf_industry_exposure: pd.DataFrame,
    geography: pd.DataFrame,
    industry: pd.DataFrame,
    scenario_id: str,
    portfolio_id: str,
) -> pd.DataFrame:
    """Build the selected-portfolio geography and industry exposure table."""
    geography_part = build_weighted_exposure(
        optimizer_weights=optimizer_weights,
        exposure_frame=latest_snapshot_exposure(etf_geography_exposure, "geography_id"),
        exposure_id_column="geography_id",
        exposure_weight_column="weight",
        exposure_lookup=geography.rename(
            columns={"geography_name": "exposure_name", "parent_region": "exposure_group"}
        )[["geography_id", "exposure_name", "exposure_group"]],
        exposure_type="Country",
    )
    industry_name = "canonical_industry_label" if "canonical_industry_label" in industry.columns else "industry_name"
    industry_group = "industry_cluster" if "industry_cluster" in industry.columns else None
    industry_lookup = industry.rename(columns={"industry_id": "industry_id", industry_name: "exposure_name"})[
        ["industry_id", "exposure_name"]
    ].copy()
    industry_lookup["exposure_group"] = industry[industry_group] if industry_group else "Unknown"
    industry_part = build_weighted_exposure(
        optimizer_weights=optimizer_weights,
        exposure_frame=latest_snapshot_exposure(etf_industry_exposure, "industry_id"),
        exposure_id_column="industry_id",
        exposure_weight_column="weight",
        exposure_lookup=industry_lookup,
        exposure_type="Industry",
    )
    exposure = pd.concat([geography_part, industry_part], ignore_index=True)
    columns = ["scenario_id", "portfolio_id", "exposure_type", "exposure_name", "exposure_group", "portfolio_exposure_weight", "rank"]
    if exposure.empty:
        return pd.DataFrame(columns=columns)

    exposure["scenario_id"] = scenario_id
    exposure["portfolio_id"] = portfolio_id
    exposure["rank"] = (
        exposure.sort_values(["exposure_type", "portfolio_exposure_weight"], ascending=[True, False])
        .groupby("exposure_type")
        .cumcount()
        .add(1)
    )
    return exposure[columns].sort_values(["exposure_type", "rank"]).reset_index(drop=True)


def build_weighted_exposure(
    *,
    optimizer_weights: pd.DataFrame,
    exposure_frame: pd.DataFrame,
    exposure_id_column: str,
    exposure_weight_column: str,
    exposure_lookup: pd.DataFrame,
    exposure_type: str,
) -> pd.DataFrame:
    """Build one exposure slice, such as Country or Industry."""
    columns = ["exposure_type", "exposure_name", "exposure_group", "portfolio_exposure_weight"]
    current = exposure_frame.copy()
    if current.empty:
        return pd.DataFrame(columns=columns)

    current[exposure_weight_column] = normalize_percent_like(current[exposure_weight_column])
    portfolio_weights = optimizer_weights[["etf_id", "weight"]].rename(columns={"weight": "portfolio_weight"})
    merged = portfolio_weights.merge(current.rename(columns={exposure_weight_column: "etf_exposure_weight"}), on="etf_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["portfolio_exposure_weight"] = merged["portfolio_weight"] * merged["etf_exposure_weight"]
    merged = merged.merge(exposure_lookup, on=exposure_id_column, how="left")
    merged["exposure_name"] = merged["exposure_name"].fillna("Unknown")
    merged["exposure_group"] = merged["exposure_group"].fillna("Unknown")
    grouped = (
        merged.groupby(["exposure_name", "exposure_group"], as_index=False)["portfolio_exposure_weight"]
        .sum()
        .sort_values("portfolio_exposure_weight", ascending=False)
        .reset_index(drop=True)
    )
    grouped["exposure_type"] = exposure_type
    return grouped[columns]


def latest_snapshot_exposure(exposure: pd.DataFrame, bucket_column: str) -> pd.DataFrame:
    """Use the latest exposure snapshot when the source contains month_id."""
    current = exposure.copy()
    if current.empty:
        return current
    if "month_id" not in current.columns:
        return current.groupby(["etf_id", bucket_column], as_index=False)["weight"].sum()

    current = current.sort_values(["etf_id", "month_id"]).reset_index(drop=True)
    latest_month = current.groupby("etf_id")["month_id"].transform("max")
    current = current.loc[current["month_id"] == latest_month].copy()
    return current.groupby(["etf_id", bucket_column], as_index=False)["weight"].sum()


def prepare_metric_lookup(etf_metric_monthly: pd.DataFrame) -> pd.DataFrame:
    """Prepare one latest metric row per ETF, normalized to decimal percentages."""
    if etf_metric_monthly.empty:
        return pd.DataFrame(columns=["etf_id", "expense_ratio", "bid_ask_spread_30d"])

    current = etf_metric_monthly.copy()
    if "month_id" in current.columns:
        current = current.sort_values(["etf_id", "month_id"]).drop_duplicates("etf_id", keep="last")
    else:
        current = current.drop_duplicates("etf_id", keep="first")

    current["expense_ratio"] = normalize_percent_like(current.get("expense_ratio", pd.Series(np.nan, index=current.index)))
    spread_column = "bid_ask_spread_30d" if "bid_ask_spread_30d" in current.columns else "bid_ask_spread"
    current["bid_ask_spread_30d"] = normalize_percent_like(current.get(spread_column, pd.Series(np.nan, index=current.index)))
    return current[["etf_id", "expense_ratio", "bid_ask_spread_30d"]].reset_index(drop=True)


def prepare_requested_universe(etf: pd.DataFrame, requested_tickers: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Filter the ETF dimension down to the requested tickers and record warnings."""
    current = etf.copy()
    current["ticker"] = current["ticker"].astype(str).str.strip()
    matched = current.loc[current["ticker"].isin(requested_tickers)].copy()
    missing_tickers = [ticker for ticker in requested_tickers if ticker not in set(matched["ticker"])]
    warnings = [f"Ticker {ticker} is not present in the input workbook and was excluded." for ticker in missing_tickers]
    return matched.reset_index(drop=True), warnings


def prepare_return_matrix(
    *,
    universe: pd.DataFrame,
    etf_market_monthly: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build the clean month-by-ETF return matrix used for simulation."""
    market = etf_market_monthly.merge(universe[["etf_id", "ticker", "etf_name"]], on="etf_id", how="inner")
    pivot = market.pivot_table(index="month_id", columns="ticker", values="monthly_return", aggfunc="first").sort_index()
    if pivot.empty:
        return pivot, universe.iloc[0:0].copy(), warnings

    usable_tickers = [ticker for ticker in pivot.columns if pivot[ticker].notna().any()]
    warnings.extend(
        f"Ticker {ticker} has no usable monthly return history and was excluded."
        for ticker in sorted(set(universe["ticker"]) - set(usable_tickers))
    )
    pivot = pivot.loc[:, usable_tickers].dropna(how="any")
    used_universe = universe.loc[universe["ticker"].isin(pivot.columns)].copy()
    used_universe = used_universe.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    return pivot.loc[:, used_universe["ticker"]], used_universe, warnings


def normalize_percent_like(values: pd.Series | np.ndarray) -> pd.Series:
    """Normalize percentage-like values so the output is decimal-based."""
    numeric = pd.to_numeric(pd.Series(values, copy=True), errors="coerce")
    if numeric.dropna().empty:
        return numeric
    return numeric / 100.0 if numeric.abs().max() > 1 else numeric


def weighted_nullable_sum(weights: pd.Series, values: pd.Series | None) -> float | None:
    """Compute a weighted sum while tolerating missing metrics."""
    if values is None:
        return None
    numeric_values = pd.to_numeric(values, errors="coerce")
    mask = numeric_values.notna()
    if not mask.any():
        return None
    return float((weights[mask] * numeric_values[mask]).sum())

