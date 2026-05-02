"""ascent/portfolio/tc_aware_optimizer.py

Transaction-cost-aware rebalancing.

Given a target weight vector from BL/MV and current holdings, computes
the optimal trade that maximizes utility net of transaction costs:

    max  mu^T w - (lam/2) w^T Sigma w - kappa * |w - w_current|

where kappa = TC budget per unit turnover (default 0.0010 = 10bps round-trip).

This prevents over-trading when the alpha signal improvement doesn't
justify the cost of getting there. Small alpha improvements → hold.
Large alpha improvements → trade toward target.

Falls back to BL target weights when covariance is unavailable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


_KAPPA = 0.0010   # 10bps round-trip TC per unit turnover (realistic for liquid US equities)
_LAMBDA = 3.0


def tc_aware_weights(
    target_weights: pd.Series,
    current_weights: pd.Series,
    mu: pd.Series,
    cov: np.ndarray,
    kappa: float = _KAPPA,
    lam: float = _LAMBDA,
    max_weight: float = 0.15,
    min_trade: float = 0.005,
) -> pd.Series:
    """
    TC-aware rebalance between current and BL target weights.

    Parameters
    ----------
    target_weights  : Series — unconstrained BL/MV optimal weights
    current_weights : Series — current holdings (0-filled for absent names)
    mu              : Series — BL posterior expected returns
    cov             : ndarray — annualized covariance matrix (aligned to target_weights)
    kappa           : float — TC per unit turnover
    min_trade       : float — ignore trades smaller than this (deadband)

    Returns
    -------
    pd.Series — final weights after TC-aware trade sizing
    """
    if target_weights is None or target_weights.empty:
        return current_weights if current_weights is not None else pd.Series(dtype=float)

    if current_weights is None or current_weights.empty:
        return target_weights

    syms = target_weights.index
    n = len(syms)

    w_curr = current_weights.reindex(syms).fillna(0.0).values
    w_tgt  = target_weights.values
    mu_arr = mu.reindex(syms).fillna(0.0).values if mu is not None else np.zeros(n)

    try:
        from scipy.optimize import minimize

        def neg_utility(w):
            port_return = mu_arr @ w
            port_risk   = 0.5 * lam * w @ cov @ w
            tc_cost     = kappa * np.abs(w - w_curr).sum()
            return -(port_return - port_risk - tc_cost)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, max_weight)] * n

        # Start from midpoint between current and target
        w0 = 0.5 * w_curr + 0.5 * w_tgt
        w0 = np.clip(w0, 0.0, max_weight)
        s = w0.sum()
        if s > 0:
            w0 = w0 / s

        res = minimize(
            neg_utility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if res.success:
            w_opt = np.clip(res.x, 0.0, max_weight)
            total = w_opt.sum()
            if total > 1e-8:
                w_opt = w_opt / total

            # Apply deadband: don't trade below min_trade threshold
            trades = np.abs(w_opt - w_curr)
            mask = trades < min_trade
            w_opt[mask] = w_curr[mask]
            s = w_opt.sum()
            if s > 1e-8:
                w_opt = w_opt / s

            return pd.Series(w_opt, index=syms)

    except Exception as e:
        print(f"[TCOptimizer] scipy minimize failed: {e}")

    # Fallback: deadband on target
    w_tgt_adj = w_tgt.copy()
    trades = np.abs(w_tgt_adj - w_curr)
    mask = trades < min_trade
    w_tgt_adj[mask] = w_curr[mask]
    s = w_tgt_adj.sum()
    if s > 1e-8:
        w_tgt_adj = w_tgt_adj / s
    return pd.Series(w_tgt_adj, index=syms)
