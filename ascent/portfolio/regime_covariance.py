"""ascent/portfolio/regime_covariance.py

Regime-conditional covariance blending.

Computes separate sample covariances for each regime label using only dates
observed in that regime, then blends them by regime probability.
"""
from __future__ import annotations
import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def estimate_regime_covariance(
    prices_df: pd.DataFrame,
    regime_labels: pd.Series,
    target_regime: str,
    min_obs: int = 63,
) -> np.ndarray:
    """
    Sample covariance using only dates labeled as target_regime.
    Falls back to full-history Ledoit-Wolf if fewer than min_obs dates available.

    prices_df: wide DataFrame (dates × symbols), tz-naive index.
    regime_labels: Series indexed by date, values are regime label strings.
    """
    returns = prices_df.pct_change().dropna(how="all")

    # Filter to target-regime dates
    regime_dates = regime_labels[regime_labels == target_regime].index
    regime_returns = returns.reindex(regime_dates).dropna(how="all")

    n = prices_df.shape[1]

    if len(regime_returns) >= min_obs:
        R = regime_returns.fillna(0.0).values
        cov = np.cov(R, rowvar=False)
        log.debug("[RegimeCov] %s covariance: %d obs, shape %s", target_regime, len(regime_returns), cov.shape)
        return cov
    else:
        log.warning(
            "[RegimeCov] %s: only %d obs < %d min — using full-history Ledoit-Wolf",
            target_regime, len(regime_returns), min_obs,
        )
        return _ledoit_wolf_cov(returns.fillna(0.0).values)


def _ledoit_wolf_cov(R: np.ndarray) -> np.ndarray:
    """Oracle approximating shrinkage (Ledoit-Wolf analytic formula)."""
    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf()
        lw.fit(R)
        return lw.covariance_
    except Exception:
        return np.cov(R, rowvar=False)


def blend_regime_covariances(
    calm_bull_cov: np.ndarray,
    stressed_cov: np.ndarray,
    crisis_cov: np.ndarray,
    regime_label: str,
    regime_confidence: float = 0.7,
) -> np.ndarray:
    """
    Blend three regime covariance matrices using current regime and confidence.

    In crisis at confidence 0.9: 90% crisis + 10% equal blend of others.
    In uncertain: equal thirds.
    """
    label = str(regime_label).lower()
    conf  = float(np.clip(regime_confidence, 0.0, 1.0))
    remainder = (1.0 - conf) / 2.0

    if "crisis" in label:
        w_calm, w_stressed, w_crisis = remainder, remainder, conf
    elif "stress" in label:
        w_calm, w_stressed, w_crisis = remainder, conf, remainder
    elif "calm" in label or "bull" in label:
        w_calm, w_stressed, w_crisis = conf, remainder, remainder
    else:
        # uncertain — equal thirds
        w_calm = w_stressed = w_crisis = 1.0 / 3.0

    blended = w_calm * calm_bull_cov + w_stressed * stressed_cov + w_crisis * crisis_cov

    # Verify weights sum to 1.0
    assert abs(w_calm + w_stressed + w_crisis - 1.0) < 1e-9, "Blend weights must sum to 1.0"
    return blended


def build_regime_covariance(
    prices_df: pd.DataFrame,
    regime_labels: pd.Series,
    regime_label: str,
    regime_confidence: float = 0.7,
    min_obs: int = 63,
) -> Optional[np.ndarray]:
    """
    Convenience wrapper: estimate all three regime covariances and blend.
    Returns None if prices_df is too small.
    """
    if prices_df is None or prices_df.empty or prices_df.shape[1] < 2:
        return None

    try:
        calm_cov    = estimate_regime_covariance(prices_df, regime_labels, "calm_bull", min_obs)
        stressed_cov = estimate_regime_covariance(prices_df, regime_labels, "stressed", min_obs)
        crisis_cov  = estimate_regime_covariance(prices_df, regime_labels, "crisis", min_obs)

        return blend_regime_covariances(
            calm_bull_cov=calm_cov,
            stressed_cov=stressed_cov,
            crisis_cov=crisis_cov,
            regime_label=regime_label,
            regime_confidence=regime_confidence,
        )
    except Exception as exc:
        log.warning("[RegimeCov] Failed: %s — returning None", exc)
        return None
