from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MONTHS_PER_YEAR = 12


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert monthly ETF prices into monthly simple returns."""
    if prices.empty:
        raise ValueError("Price data is empty.")

    returns = prices.sort_index().apply(pd.to_numeric, errors="coerce")
    returns = returns.pct_change(fill_method=None).dropna(how="all").dropna(axis=1, how="all")
    if returns.empty:
        raise ValueError("Could not compute returns from the provided prices.")
    return returns


def estimate_mu_sigma(returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate mean monthly returns and covariance from historical returns."""
    if returns.empty:
        raise ValueError("Return data is empty.")

    current = returns.dropna(how="any").copy()
    if current.empty:
        raise ValueError("No complete return rows remain after removing missing values.")

    mu = current.mean()
    sigma = current.cov()
    if mu.isna().any() or sigma.isna().any().any():
        raise ValueError("Could not estimate expected returns or covariance matrix.")
    return mu, sigma


def optimize_portfolio(mu: pd.Series, sigma: pd.DataFrame, target_return: float | None = None, max_weight: float = 0.3, objective: str = "min_variance", risk_free_rate: float | None = None) -> pd.Series:
    """Solve the long-only Markowitz problem under the selected objective."""
    mu_series = pd.Series(mu, dtype=float)
    sigma_frame = pd.DataFrame(sigma, index=mu_series.index, columns=mu_series.index, dtype=float)

    _validate_optimization_inputs(mu_series, sigma_frame, target_return, max_weight, objective)

    min_feasible_return, max_feasible_return = _feasible_return_range(mu_series, max_weight)
    if objective == "min_variance" and target_return is not None and target_return > max_feasible_return + 1e-10:
        raise ValueError(
            "Target return is infeasible under the long-only weight constraints. "
            f"Feasible monthly range: [{min_feasible_return:.6f}, {max_feasible_return:.6f}], target={target_return:.6f}."
        )

    mu_values = mu_series.to_numpy(dtype=float)
    sigma_values = sigma_frame.to_numpy(dtype=float)
    asset_count = len(mu_series)
    initial_weights = np.full(asset_count, 1.0 / asset_count, dtype=float)
    bounds = [(0.0, max_weight)] * asset_count
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if objective == "min_variance":
        constraints.append({"type": "ineq", "fun": lambda w: float(np.dot(w, mu_values) - target_return)})
        objective_function = lambda w: float(w @ sigma_values @ w)
    else:
        resolved_risk_free = 0.0 if risk_free_rate is None else float(risk_free_rate)
        objective_function = lambda w: -_compute_sharpe_ratio(
            w,
            mu_values=mu_values,
            sigma_values=sigma_values,
            risk_free_rate=resolved_risk_free,
        )

    result = minimize(
        fun=objective_function,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    if not result.success:
        raise ValueError(
            "Portfolio optimization failed. "
            "The constraints or optimizer settings may be too restrictive. "
            f"Solver message: {result.message}"
        )

    weights = pd.Series(result.x, index=mu_series.index, name="weight").where(lambda item: item > 1e-10, 0.0)
    return weights / weights.sum()


def portfolio_stats(weights: pd.Series | np.ndarray, mu: pd.Series, sigma: pd.DataFrame, risk_free_rate: float | None = None, annualize: bool = False) -> dict[str, float]:
    """Compute expected return, volatility, and Sharpe ratio for a portfolio."""
    mu_series = pd.Series(mu, dtype=float)
    sigma_frame = pd.DataFrame(sigma, index=mu_series.index, columns=mu_series.index, dtype=float)
    weight_series = _coerce_weights(weights, mu_series.index)

    expected_return = float(weight_series @ mu_series)
    variance = float(weight_series @ sigma_frame @ weight_series)
    volatility = math.sqrt(max(variance, 0.0))

    period_multiplier = MONTHS_PER_YEAR if annualize else 1
    expected_return *= period_multiplier
    volatility *= math.sqrt(period_multiplier)
    adjusted_risk_free = None if risk_free_rate is None else float(risk_free_rate) * period_multiplier

    sharpe_ratio = np.nan
    if adjusted_risk_free is not None and volatility > 0:
        sharpe_ratio = float((expected_return - adjusted_risk_free) / volatility)

    return {
        "expected_portfolio_return": expected_return,
        "portfolio_volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
    }


def maximum_achievable_return(mu: pd.Series, max_weight: float = 0.3, annualize: bool = False) -> float:
    """Return the highest feasible expected return under the long-only weight cap."""
    mu_series = pd.Series(mu, dtype=float)
    if mu_series.empty:
        raise ValueError("Expected return vector is empty.")
    if mu_series.isna().any():
        raise ValueError("Expected return vector must not contain missing values.")
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be between 0 and 1.")
    if len(mu_series) * max_weight < 1 - 1e-10:
        raise ValueError(
            "The optimization constraints are infeasible because the ETF count is too small "
            f"for max_weight={max_weight:.2f}. Need at least {math.ceil(1 / max_weight)} assets."
        )

    result = _extreme_feasible_return(mu_series, max_weight=max_weight, ascending=False)
    return result * MONTHS_PER_YEAR if annualize else result


def run_markowitz_optimization(tickers: Sequence[str], prices: pd.DataFrame, target_return: float | None = None, risk_free_rate: float | None = None, max_weight: float = 0.3, annualize: bool = False, objective: str = "min_variance") -> pd.DataFrame:
    """Run the full Markowitz workflow and return a Tableau-friendly result table."""
    if not tickers:
        raise ValueError("At least one ETF ticker must be provided.")

    selected_tickers = list(dict.fromkeys(tickers))
    missing_tickers = [ticker for ticker in selected_tickers if ticker not in prices.columns]
    if missing_tickers:
        raise ValueError(f"Missing price columns for tickers: {', '.join(missing_tickers)}")

    selected_prices = prices.loc[:, selected_tickers]
    returns = compute_returns(selected_prices)
    mu, sigma = estimate_mu_sigma(returns)
    max_return = maximum_achievable_return(mu, max_weight=max_weight, annualize=annualize)
    weights = optimize_portfolio(
        mu,
        sigma,
        target_return=target_return,
        max_weight=max_weight,
        objective=objective,
        risk_free_rate=risk_free_rate,
    )
    stats = portfolio_stats(weights, mu, sigma, risk_free_rate=risk_free_rate, annualize=annualize)

    result = weights.rename_axis("ticker").reset_index()
    result["weight"] = result["weight"].round(3)
    for key, value in stats.items():
        result[key] = value
    result["max_achievable_return"] = max_return
    result["optimization_objective"] = objective
    return result.sort_values("ticker").reset_index(drop=True)


def _validate_optimization_inputs(mu: pd.Series, sigma: pd.DataFrame, target_return: float | None, max_weight: float, objective: str) -> None:
    """Validate the inputs before optimization starts."""
    if mu.empty:
        raise ValueError("Expected return vector is empty.")
    if sigma.empty:
        raise ValueError("Covariance matrix is empty.")
    if not mu.index.equals(sigma.index) or not sigma.index.equals(sigma.columns):
        raise ValueError("mu and sigma must refer to the same ETF tickers in the same order.")
    if mu.isna().any() or sigma.isna().any().any():
        raise ValueError("mu and sigma must not contain missing values.")
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be between 0 and 1.")
    if len(mu) * max_weight < 1 - 1e-10:
        raise ValueError(
            "The optimization constraints are infeasible because the ETF count is too small "
            f"for max_weight={max_weight:.2f}. Need at least {math.ceil(1 / max_weight)} assets."
        )
    if objective not in {"min_variance", "max_sharpe"}:
        raise ValueError("objective must be either 'min_variance' or 'max_sharpe'.")
    if objective == "min_variance":
        if target_return is None:
            raise ValueError("target_return is required when objective='min_variance'.")
        if not np.isfinite(target_return):
            raise ValueError("target_return must be a finite number.")


def _compute_sharpe_ratio(weights: np.ndarray, *, mu_values: np.ndarray, sigma_values: np.ndarray, risk_free_rate: float) -> float:
    """Compute the Sharpe ratio for a weight vector."""
    expected_return = float(np.dot(weights, mu_values))
    variance = float(weights @ sigma_values @ weights)
    volatility = math.sqrt(max(variance, 0.0))
    if volatility <= 1e-12:
        return -1e12
    return (expected_return - risk_free_rate) / volatility


def _feasible_return_range(mu: pd.Series, max_weight: float) -> tuple[float, float]:
    """Compute the minimum and maximum feasible portfolio return under the weight cap."""
    return (
        _extreme_feasible_return(mu, max_weight=max_weight, ascending=True),
        _extreme_feasible_return(mu, max_weight=max_weight, ascending=False),
    )


def _extreme_feasible_return(mu: pd.Series, max_weight: float, *, ascending: bool) -> float:
    """Build the most defensive or aggressive feasible return using greedy allocation."""
    remaining_weight = 1.0
    total_return = 0.0

    for asset_return in mu.sort_values(ascending=ascending):
        allocation = min(max_weight, remaining_weight)
        total_return += allocation * float(asset_return)
        remaining_weight -= allocation
        if remaining_weight <= 1e-12:
            break

    return total_return


def _coerce_weights(weights: pd.Series | np.ndarray, index: pd.Index) -> pd.Series:
    """Align weight inputs to the ETF order used by mu and sigma."""
    if isinstance(weights, pd.Series):
        weight_series = weights.reindex(index).astype(float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if len(weight_array) != len(index):
            raise ValueError("weights length does not match the ETF universe.")
        weight_series = pd.Series(weight_array, index=index, dtype=float)

    if weight_series.isna().any():
        raise ValueError("weights must cover every ETF and contain no missing values.")

    return weight_series

