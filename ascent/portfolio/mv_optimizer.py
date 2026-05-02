"""ascent/portfolio/mv_optimizer.py

Black-Litterman Mean-Variance Optimizer.

Combines market-cap equilibrium returns (prior) with alpha score "views"
(signals from the alpha stack). Uses Ledoit-Wolf shrinkage covariance.

BL formula:
    mu_bl = [(tau * Sigma)^-1 + P^T * Omega^-1 * P]^-1
            * [(tau * Sigma)^-1 * pi + P^T * Omega^-1 * Q]

where:
    pi    = market-cap equilibrium excess returns (lambda * Sigma * w_mkt)
    P     = view matrix (identity for absolute views on each asset)
    Q     = alpha z-scores scaled to return units
    Omega = diagonal uncertainty matrix (proportional to view confidence)
    tau   = scaling factor for prior uncertainty (default 0.05)

MV optimization:
    max   mu_bl^T * w - (lambda/2) * w^T * Sigma * w
    s.t.  w >= 0, sum(w) = 1, w_i <= max_weight

Uses scipy.optimize.minimize with SLSQP.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from .covariance import get_cov


_LAMBDA = 3.0    # risk-aversion coefficient (typical: 2-4 for equities)
_TAU    = 0.05   # prior uncertainty scaling


def _equilibrium_returns(
    cov: np.ndarray,
    mkt_weights: np.ndarray,
    lam: float = _LAMBDA,
) -> np.ndarray:
    """pi = lambda * Sigma * w_mkt"""
    return lam * cov @ mkt_weights


def _bl_posterior(
    pi: np.ndarray,
    cov: np.ndarray,
    views: np.ndarray,
    view_conf: np.ndarray,
    tau: float = _TAU,
) -> np.ndarray:
    """
    Black-Litterman posterior expected returns.

    views     : shape (n,) — absolute views on each asset (alpha z-scores)
    view_conf : shape (n,) — confidence per view (0=ignore, 1=fully trust)
    """
    n = len(pi)
    P = np.eye(n)                    # absolute views on every asset
    Q = views

    # Omega: diagonal uncertainty — lower confidence = higher uncertainty
    # Omega_i = (1 - conf_i) / conf_i * var(view_i); degenerate conf→0 → Omega→∞ → prior dominates
    conf = np.clip(view_conf, 1e-4, 1.0)
    sigma_diag = np.diag(cov)
    omega_diag = (1.0 - conf) / conf * sigma_diag * tau
    omega_diag = np.maximum(omega_diag, 1e-8)   # numerical floor
    Omega_inv = np.diag(1.0 / omega_diag)

    prior_prec = np.linalg.inv(tau * cov + 1e-8 * np.eye(n))
    view_prec  = P.T @ Omega_inv @ P

    post_cov_inv = prior_prec + view_prec
    post_cov = np.linalg.inv(post_cov_inv + 1e-8 * np.eye(n))

    mu_bl = post_cov @ (prior_prec @ pi + P.T @ Omega_inv @ Q)
    return mu_bl


def _mv_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    max_weight: float = 0.15,
    lam: float = _LAMBDA,
) -> np.ndarray:
    """
    Mean-variance optimization: max mu^T w - (lam/2) w^T Sigma w
    s.t. w >= 0, sum(w) = 1, w_i <= max_weight
    Returns weight array (n,).
    """
    try:
        from scipy.optimize import minimize

        n = len(mu)

        def neg_utility(w):
            return -(mu @ w - (lam / 2) * w @ cov @ w)

        def neg_utility_grad(w):
            return -(mu - lam * cov @ w)

        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bounds = [(0.0, max_weight)] * n

        w0 = np.ones(n) / n
        res = minimize(
            neg_utility,
            w0,
            jac=neg_utility_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )

        if res.success:
            w = np.clip(res.x, 0.0, max_weight)
            total = w.sum()
            if total > 1e-8:
                return w / total
    except Exception as e:
        print(f"[MVOptimizer] scipy minimize failed: {e}")

    # Fallback: equal weight
    n = len(mu)
    return np.ones(n) / n


def bl_weights(
    alpha_scores: pd.Series,
    returns: pd.DataFrame,
    max_weight: float = 0.15,
    tau: float = _TAU,
    lam: float = _LAMBDA,
    min_history: int = 63,
) -> Optional[pd.Series]:
    """
    Black-Litterman portfolio weights.

    Parameters
    ----------
    alpha_scores : Series (symbols) — composite alpha z-scores for current date
    returns      : DataFrame (dates × symbols) — recent daily returns
    max_weight   : float — maximum position size
    min_history  : int — minimum rows required in returns

    Returns
    -------
    pd.Series (symbols) or None if insufficient data
    """
    if alpha_scores is None or alpha_scores.empty:
        return None

    # Align symbols — only keep names present in both
    syms = alpha_scores.dropna().index.intersection(returns.columns)
    if len(syms) < 3:
        return None

    scores = alpha_scores[syms]
    ret = returns[syms].dropna(how="any")

    if len(ret) < min_history:
        return None

    cov = get_cov(ret, min_periods=min_history)
    if cov is None:
        return None

    n = len(syms)

    # Market-cap proxy: equal-weight market portfolio (no live cap data)
    w_mkt = np.ones(n) / n
    pi = _equilibrium_returns(cov, w_mkt, lam=lam)

    # Views: alpha z-scores scaled to annualized return units
    # Map z-scores linearly: z=1 → 5% expected outperformance (reasonable for cross-sectional)
    z = scores.values
    views = pi + z * 0.05

    # View confidence: proportional to |z-score|, capped at 0.9
    view_conf = np.clip(np.abs(z) / 3.0, 0.05, 0.9)

    mu_bl = _bl_posterior(pi, cov, views, view_conf, tau=tau)

    w = _mv_optimize(mu_bl, cov, max_weight=max_weight, lam=lam)

    return pd.Series(w, index=syms)
