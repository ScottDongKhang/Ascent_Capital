"""ascent/portfolio/black_litterman.py

Black-Litterman posterior expected return blending.

Runs as a pre-processing step before alpha enters the MVO optimizer.
Blends quant composite alpha (prior) with LLM fundamental sleeve output (view).
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def get_blending_weight(llm_ic_ir: float) -> float:
    """Map LLM sleeve IC IR to a BL blending weight (tau)."""
    if llm_ic_ir < 0.30:
        return 0.05
    if llm_ic_ir < 0.60:
        return 0.10
    return 0.15


def black_litterman_views(
    quant_alpha: pd.Series,
    llm_alpha: Optional[pd.Series],
    covariance: Optional[np.ndarray] = None,
    tau: float = 0.05,
    omega_scale: float = 1.0,
) -> pd.Series:
    """
    Black-Litterman posterior expected return.

    For absolute views (P = I), the closed-form solution is:
        μ_post = [(τΣ)⁻¹ + Ω⁻¹]⁻¹ · [(τΣ)⁻¹ μ_prior + Ω⁻¹ Q]

    When covariance is unavailable, falls back to simple shrinkage:
        μ_post = (1 - τ) · quant_alpha + τ · llm_alpha

    llm_alpha may cover a subset of quant_alpha symbols — unmapped symbols
    keep their quant prior unchanged.
    """
    if llm_alpha is None or llm_alpha.empty:
        return quant_alpha.copy()

    # Align to quant universe; fill missing LLM views with quant prior
    common = quant_alpha.index.intersection(llm_alpha.index)
    if common.empty:
        return quant_alpha.copy()

    mu_prior = quant_alpha.reindex(common).fillna(0.0).values.astype(float)
    # fillna using aligned quant prior for any NaN LLM values
    llm_aligned = llm_alpha.reindex(common)
    Q = llm_aligned.where(llm_aligned.notna(), quant_alpha.reindex(common)).values.astype(float)
    n        = len(common)

    if covariance is not None and covariance.shape[0] == n:
        Sigma = covariance.astype(float)
        try:
            tau_Sigma_inv = np.linalg.inv(tau * Sigma + np.eye(n) * 1e-8)
            Omega_inv     = np.eye(n) / (omega_scale * tau)

            posterior_cov_inv = tau_Sigma_inv + Omega_inv
            posterior_mean    = np.linalg.solve(
                posterior_cov_inv,
                tau_Sigma_inv @ mu_prior + Omega_inv @ Q,
            )
        except np.linalg.LinAlgError:
            log.warning("[BL] Matrix inversion failed — using shrinkage fallback")
            posterior_mean = (1 - tau) * mu_prior + tau * Q
    else:
        # Shrinkage fallback when covariance is unavailable
        posterior_mean = (1 - tau) * mu_prior + tau * Q

    # Rebuild full-universe series: BL posterior for common symbols, quant prior elsewhere
    result = quant_alpha.copy()
    result[common] = pd.Series(posterior_mean, index=common)
    return result
