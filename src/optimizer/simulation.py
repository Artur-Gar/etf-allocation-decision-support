from __future__ import annotations

import numpy as np
import pandas as pd

from optimizer.constants import SCENARIO_ID


def simulate_random_portfolios(
    *,
    num_assets: int,
    num_portfolios: int,
    max_weight: float,
    random_state: int,
) -> np.ndarray:
    """Sample random long-only portfolio weights under the ETF weight cap."""
    rng = np.random.default_rng(random_state)
    accepted_batches: list[np.ndarray] = []
    accepted_count = 0
    attempts = 0
    max_attempts = max(num_portfolios * 200, 1_000_000)
    batch_size = max(50_000, num_portfolios)

    while accepted_count < num_portfolios:
        draws = rng.dirichlet(np.ones(num_assets), size=batch_size)
        valid = draws.max(axis=1) <= max_weight + 1e-12
        if valid.any():
            accepted = draws[valid]
            accepted_batches.append(accepted)
            accepted_count += len(accepted)
        attempts += batch_size
        if attempts > max_attempts and accepted_count == 0:
            raise ValueError("Could not sample any valid portfolios under the max ETF weight constraint.")

    return np.vstack(accepted_batches)[:num_portfolios]


def build_optimizer_frontier(
    *,
    weights: np.ndarray,
    monthly_returns: pd.DataFrame,
    annual_mu: pd.Series,
    annual_cov: pd.DataFrame,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Build the simulated portfolio frontier table."""
    annual_mu_values = annual_mu.to_numpy(dtype=float)
    annual_cov_values = annual_cov.to_numpy(dtype=float)

    expected_returns = weights @ annual_mu_values
    expected_variances = np.einsum("ij,jk,ik->i", weights, annual_cov_values, weights)
    expected_volatility = np.sqrt(np.maximum(expected_variances, 0.0))
    sharpe_ratio = np.divide(
        expected_returns - risk_free_rate,
        expected_volatility,
        out=np.full_like(expected_returns, np.nan),
        where=expected_volatility > 0,
    )
    portfolio_monthly_returns = monthly_returns.to_numpy(dtype=float) @ weights.T
    max_drawdown = compute_max_drawdown(portfolio_monthly_returns)

    frontier = pd.DataFrame(
        {
            "scenario_id": SCENARIO_ID,
            "portfolio_number": np.arange(len(weights), dtype=int),
            "portfolio_id": [f"P{i:06d}" for i in range(1, len(weights) + 1)],
            "expected_return": expected_returns,
            "expected_volatility": expected_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }
    )
    selected_index = int(frontier["sharpe_ratio"].fillna(-np.inf).idxmax())
    frontier["is_selected_portfolio"] = False
    frontier.loc[selected_index, "is_selected_portfolio"] = True
    return frontier


def compute_max_drawdown(portfolio_monthly_returns: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each simulated portfolio return series."""
    cumulative = np.cumprod(1.0 + portfolio_monthly_returns, axis=0)
    running_peaks = np.maximum.accumulate(cumulative, axis=0)
    drawdowns = np.divide(
        cumulative,
        running_peaks,
        out=np.ones_like(cumulative),
        where=running_peaks != 0,
    ) - 1.0
    return drawdowns.min(axis=0)

