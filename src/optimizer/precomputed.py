from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    PORTFOLIO_OPTIMIZER_PRECOMPUTED_PATH,
    TABLEAU_RELATIONAL_FINAL_PATH,
    TABLEAU_RELATIONAL_OUTPUT_PATH,
)
from optimizer.constants import (
    LONG_ONLY,
    LOOKBACK_WINDOW,
    MAX_ETF_WEIGHT,
    MIN_PERIODS,
    MIN_WEIGHT_TO_KEEP,
    NUM_SIMULATIONS,
    OPTIMIZATION_OBJECTIVE,
    RANDOM_STATE,
    REQUIRED_SHEETS,
    REQUESTED_TICKERS,
    RISK_FREE_RATE,
    SCENARIO_ID,
    SCENARIO_NAME,
)
from optimizer.simulation import build_optimizer_frontier, simulate_random_portfolios
from optimizer.tables import (
    build_optimizer_exposure,
    build_optimizer_kpis,
    build_optimizer_weights,
    prepare_metric_lookup,
    prepare_requested_universe,
    prepare_return_matrix,
)
from utils import ensure_directories, first_existing_path, save_excel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the portfolio optimizer workbook export."""
    parser = argparse.ArgumentParser(description="Build a precomputed portfolio-optimizer workbook.")
    parser.add_argument("--num-simulations", type=int, default=NUM_SIMULATIONS)
    parser.add_argument("--max-weight", type=float, default=MAX_ETF_WEIGHT)
    parser.add_argument("--risk-free-rate", type=float, default=RISK_FREE_RATE)
    parser.add_argument("--etfs-list", type=str, default=",".join(REQUESTED_TICKERS))
    return parser.parse_args()


def run_portfolio_optimizer_precomputed(
    *,
    num_simulations: int = NUM_SIMULATIONS,
    requested_tickers: list[str] | None = None,
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float = MAX_ETF_WEIGHT,
) -> dict[str, Any]:
    """Read the final Tableau workbook and export the optimizer workbook."""
    ensure_directories()
    resolved_input = _resolve_input_workbook()
    resolved_output = PORTFOLIO_OPTIMIZER_PRECOMPUTED_PATH.resolve()
    tables = read_input_workbook(resolved_input)
    optimizer_tables, summary = build_optimizer_workbook(
        tables,
        requested_tickers=requested_tickers or REQUESTED_TICKERS,
        num_simulations=num_simulations,
        random_state=RANDOM_STATE,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        min_periods=MIN_PERIODS,
        min_weight_to_keep=MIN_WEIGHT_TO_KEEP,
    )
    save_excel(optimizer_tables, resolved_output)

    return {
        "input_path": str(resolved_input),
        "output_path": str(resolved_output),
        "sheet_count": len(optimizer_tables),
        "sheets": {name: len(frame) for name, frame in optimizer_tables.items()},
        **summary,
    }


def read_input_workbook(path: Path) -> dict[str, pd.DataFrame]:
    """Load the required sheets from the existing Tableau workbook."""
    workbook = pd.ExcelFile(path)
    missing_sheets = sorted(REQUIRED_SHEETS - set(workbook.sheet_names))
    if missing_sheets:
        raise ValueError("The input workbook is missing required sheets: " + ", ".join(missing_sheets))
    return {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in REQUIRED_SHEETS}


def build_optimizer_workbook(
    tables: dict[str, pd.DataFrame],
    *,
    requested_tickers: list[str],
    num_simulations: int,
    random_state: int,
    risk_free_rate: float,
    max_weight: float,
    min_periods: int,
    min_weight_to_keep: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Build all optimizer output sheets from the Tableau relational input tables."""
    if num_simulations < 50_000:
        raise ValueError("num_simulations must be at least 50,000.")

    universe, warnings = prepare_requested_universe(tables["etf"], requested_tickers)
    return_matrix, used_universe, warnings = prepare_return_matrix(
        universe=universe,
        etf_market_monthly=tables["etf_market_monthly"],
        warnings=warnings,
    )
    if return_matrix.empty:
        raise ValueError("No complete monthly return history remains after ETF filtering.")
    if return_matrix.shape[1] * max_weight < 1 - 1e-12:
        raise ValueError(
            "Optimization is infeasible after return-history validation. "
            f"Need at least {math.ceil(1 / max_weight)} ETFs for max_weight={max_weight:.2f}, "
            f"but only {return_matrix.shape[1]} remain."
        )

    annual_mu = return_matrix.mean() * 12.0
    annual_cov = return_matrix.cov() * 12.0
    weights = simulate_random_portfolios(
        num_assets=return_matrix.shape[1],
        num_portfolios=num_simulations,
        max_weight=max_weight,
        random_state=random_state,
    )
    frontier = build_optimizer_frontier(
        weights=weights,
        monthly_returns=return_matrix,
        annual_mu=annual_mu,
        annual_cov=annual_cov,
        risk_free_rate=risk_free_rate,
    )
    selected_portfolio = frontier.loc[frontier["sharpe_ratio"].fillna(-np.inf).idxmax()].copy()
    selected_portfolio_id = str(selected_portfolio["portfolio_id"])
    selected_weight_vector = pd.Series(
        weights[int(selected_portfolio["portfolio_number"])],
        index=return_matrix.columns,
        dtype=float,
    )

    optimizer_weights = build_optimizer_weights(
        used_universe=used_universe,
        selected_weights=selected_weight_vector,
        metric_lookup=prepare_metric_lookup(tables["etf_metric_monthly"]),
        scenario_id=SCENARIO_ID,
        portfolio_id=selected_portfolio_id,
        min_weight_to_keep=min_weight_to_keep,
    )
    optimizer_kpis = build_optimizer_kpis(
        optimizer_weights=optimizer_weights,
        selected_portfolio=selected_portfolio,
        scenario_id=SCENARIO_ID,
        portfolio_id=selected_portfolio_id,
    )
    optimizer_exposure = build_optimizer_exposure(
        optimizer_weights=optimizer_weights,
        etf_geography_exposure=tables["etf_geography_exposure_monthly"],
        etf_industry_exposure=tables["etf_industry_exposure_monthly"],
        geography=tables["geography"],
        industry=tables["industry"],
        scenario_id=SCENARIO_ID,
        portfolio_id=selected_portfolio_id,
    )
    output_tables = _assemble_output_tables(frontier, optimizer_weights, optimizer_kpis, optimizer_exposure, max_weight, risk_free_rate, used_universe, selected_portfolio_id)
    return output_tables, _build_summary(requested_tickers, used_universe, selected_portfolio, selected_portfolio_id, warnings)


def _assemble_output_tables(
    frontier: pd.DataFrame,
    optimizer_weights: pd.DataFrame,
    optimizer_kpis: pd.DataFrame,
    optimizer_exposure: pd.DataFrame,
    max_weight: float,
    risk_free_rate: float,
    used_universe: pd.DataFrame,
    selected_portfolio_id: str,
) -> dict[str, pd.DataFrame]:
    """Assemble the five Tableau optimizer sheets."""
    optimizer_scenarios = pd.DataFrame(
        [
            {
                "scenario_id": SCENARIO_ID,
                "scenario_name": SCENARIO_NAME,
                "optimization_objective": OPTIMIZATION_OBJECTIVE,
                "lookback_window": LOOKBACK_WINDOW,
                "max_etf_weight": max_weight,
                "risk_free_rate": risk_free_rate,
                "long_only": LONG_ONLY,
                "max_number_of_etfs": int(len(used_universe)),
                "selected_portfolio_id": selected_portfolio_id,
            }
        ]
    )
    return {
        "optimizer_scenarios": optimizer_scenarios,
        "optimizer_frontier": frontier[
            ["scenario_id", "portfolio_id", "expected_return", "expected_volatility", "sharpe_ratio", "max_drawdown", "is_selected_portfolio"]
        ].copy(),
        "optimizer_weights": optimizer_weights,
        "optimizer_kpis": optimizer_kpis,
        "optimizer_exposure": optimizer_exposure,
    }


def _build_summary(
    requested_tickers: list[str],
    used_universe: pd.DataFrame,
    selected_portfolio: pd.Series,
    selected_portfolio_id: str,
    warnings: list[str],
) -> dict[str, Any]:
    """Summarize the optimizer run for the command-line output."""
    return {
        "number_of_etfs_requested": len(requested_tickers),
        "number_of_etfs_actually_used": int(len(used_universe)),
        "selected_portfolio_id": selected_portfolio_id,
        "selected_expected_return": float(selected_portfolio["expected_return"]),
        "selected_volatility": float(selected_portfolio["expected_volatility"]),
        "selected_sharpe_ratio": float(selected_portfolio["sharpe_ratio"]),
        "selected_max_drawdown": float(selected_portfolio["max_drawdown"]),
        "warnings": warnings,
    }


def _resolve_input_workbook() -> Path:
    """Resolve the optimizer input workbook, preferring the final country-only model."""
    resolved = first_existing_path(
        OUTPUT_DIR / "tableau_data_final.xlsx",
        TABLEAU_RELATIONAL_FINAL_PATH,
        TABLEAU_RELATIONAL_OUTPUT_PATH,
    )
    return resolved.resolve()


def parse_etfs_list(value: str) -> list[str]:
    """Parse a comma-separated ETF list into unique ticker symbols."""
    tickers = [item.strip() for item in value.split(",")]
    return list(dict.fromkeys(ticker for ticker in tickers if ticker))


def main() -> None:
    """Run the precomputed portfolio optimizer workbook build from the command line."""
    args = parse_args()
    summary = run_portfolio_optimizer_precomputed(
        num_simulations=args.num_simulations,
        requested_tickers=parse_etfs_list(args.etfs_list),
        risk_free_rate=args.risk_free_rate,
        max_weight=args.max_weight,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

