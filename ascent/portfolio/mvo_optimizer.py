"""ascent/portfolio/mvo_optimizer.py

Mean-Variance Optimizer (MVO) using cvxpy.

Objective: maximize w'α - λ·(w'Σw) - κ·‖w - w_prev‖₁
Falls back gracefully to None if cvxpy is unavailable or the problem is infeasible.
Caller (optimizer.py) handles the fallback to rank-weighting.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _diagonal_covariance_proxy(n_assets: int, vol_estimate: float = 0.25) -> np.ndarray:
    """Diagonal proxy Σ when Plan 1 covariance is unavailable."""
    daily_var = (vol_estimate / np.sqrt(252)) ** 2
    return np.eye(n_assets) * daily_var


def optimize_mvo(
    alpha_scores: pd.Series,
    covariance: Optional[np.ndarray] = None,
    current_weights: Optional[pd.Series] = None,
    factor_constraints: Optional[list] = None,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.002,
    max_weight: float = 0.10,
    min_weight: float = 0.02,
    top_n: int = 15,
) -> Optional[pd.Series]:
    """
    MVO portfolio optimizer.

    Returns a weight Series summing to 1.0, or None if infeasible.
    factor_constraints: list of dicts from ascent.risk.factor_constraints.build_factor_constraints()
    """
    try:
        import cvxpy as cp
    except ImportError:
        log.warning("[MVO] cvxpy not installed — returning None")
        return None

    if alpha_scores.empty:
        return None

    # ── Pre-screen: keep only top-N by alpha ──────────────────────────────────
    if len(alpha_scores) > top_n:
        alpha_scores = alpha_scores.nlargest(top_n)

    syms = alpha_scores.index.tolist()
    n = len(syms)

    if n == 0:
        return None

    alpha_vec = alpha_scores.values.astype(float)

    # ── Covariance ────────────────────────────────────────────────────────────
    if covariance is None:
        Sigma = _diagonal_covariance_proxy(n)
    else:
        # Slice covariance to the selected symbols if needed
        if covariance.shape[0] != n:
            Sigma = _diagonal_covariance_proxy(n)
        else:
            Sigma = covariance.astype(float)

    # ── Previous weights (for turnover penalty) ────────────────────────────────
    if current_weights is not None:
        w_prev = current_weights.reindex(syms).fillna(0.0).values.astype(float)
    else:
        w_prev = np.zeros(n)

    # ── cvxpy problem ─────────────────────────────────────────────────────────
    w = cp.Variable(n)

    portfolio_return = alpha_vec @ w
    portfolio_risk   = cp.quad_form(w, Sigma)
    turnover         = cp.norm1(w - w_prev)

    objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk - turnover_penalty * turnover)

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= max_weight,
    ]

    # Factor exposure constraints from Plan 1
    if factor_constraints:
        for fc in factor_constraints:
            bvec = fc.get("loadings_vector")
            lb   = fc.get("lb")
            ub   = fc.get("ub")
            if bvec is not None and len(bvec) == n:
                b = bvec.astype(float)
                if lb is not None:
                    constraints.append(b @ w >= lb)
                if ub is not None:
                    constraints.append(b @ w <= ub)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception as e:
            log.warning("[MVO] Solver failed: %s", e)
            return None

    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        log.warning("[MVO] Problem infeasible or unbounded (status=%s)", prob.status)
        return None

    w_opt = np.array(w.value, dtype=float)

    # Enforce min_weight threshold: zero out tiny positions
    w_opt[w_opt < min_weight * 0.5] = 0.0

    total = w_opt.sum()
    if total < 1e-8:
        return None
    w_opt /= total

    result = pd.Series(w_opt, index=syms)
    result = result[result > 0]
    return result


def compute_expected_turnover_cost(
    current_weights: pd.Series,
    proposed_weights: pd.Series,
    one_way_cost: float = 0.002,
) -> float:
    """
    Expected total transaction cost as a fraction of NAV.
    one_way_cost: half-spread + market impact estimate (default 20bps).
    """
    all_syms = current_weights.index.union(proposed_weights.index)
    curr = current_weights.reindex(all_syms).fillna(0.0)
    prop = proposed_weights.reindex(all_syms).fillna(0.0)
    turnover = (prop - curr).abs().sum()
    return float(turnover * one_way_cost)
