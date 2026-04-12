"""
Ascent Capital — VaR, CVaR, and Risk Metrics
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk. Returns negative number (loss)."""
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def historical_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall). Average loss beyond VaR."""
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    if len(tail) == 0:
        return var
    return float(tail.mean())


def portfolio_risk_report(
    portfolio_returns: pd.Series,
    weights: pd.Series | None = None,
) -> dict:
    """Compute risk metrics for a portfolio."""
    return {
        "var_95": historical_var(portfolio_returns, 0.95),
        "var_99": historical_var(portfolio_returns, 0.99),
        "cvar_95": historical_cvar(portfolio_returns, 0.95),
        "cvar_99": historical_cvar(portfolio_returns, 0.99),
        "daily_vol": portfolio_returns.std(),
        "annual_vol": portfolio_returns.std() * np.sqrt(252),
        "skewness": portfolio_returns.skew(),
        "kurtosis": portfolio_returns.kurtosis(),
        "worst_day": portfolio_returns.min(),
        "best_day": portfolio_returns.max(),
    }


def stress_test(
    weights: pd.Series,
    returns_data: pd.DataFrame,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """
    Run stress test scenarios.

    Each scenario is a dict of {symbol: return_shock}.
    Returns portfolio P&L under each scenario.
    """
    if scenarios is None:
        scenarios = {
            "market_crash_20pct": {sym: -0.20 for sym in weights.index},
            "tech_crash_30pct": {sym: -0.30 if sym in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
                                  else -0.05 for sym in weights.index},
            "rate_shock": {sym: -0.15 if sym in ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "CRM", "ADBE", "NFLX"]
                           else -0.05 for sym in weights.index},
        }

    results = {}
    for name, shocks in scenarios.items():
        pnl = 0.0
        for sym, w in weights.items():
            shock = shocks.get(sym, 0.0)
            pnl += w * shock
        results[name] = pnl

    return results
