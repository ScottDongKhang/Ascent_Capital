"""ascent/portfolio/covariance.py

Ledoit-Wolf shrinkage covariance estimator for portfolio construction.

Returns an annualized covariance matrix suitable for BL and MV optimization.
Uses sklearn's LedoitWolf which auto-selects the optimal shrinkage coefficient.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


def ledoit_wolf_cov(
    returns: pd.DataFrame,
    min_periods: int = 63,
    annualize: bool = True,
) -> Optional[np.ndarray]:
    """
    Compute annualized Ledoit-Wolf shrinkage covariance matrix.

    Parameters
    ----------
    returns : DataFrame (dates × symbols), daily returns
    min_periods : int — minimum rows required; returns None if insufficient
    annualize : bool — multiply by 252

    Returns
    -------
    np.ndarray (n_assets × n_assets) or None
    """
    if returns is None or returns.empty:
        return None

    clean = returns.dropna(how="any")
    if len(clean) < min_periods:
        return None

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=False)
        lw.fit(clean.values)
        cov = lw.covariance_
        if annualize:
            cov = cov * 252
        return cov
    except Exception as e:
        print(f"[Covariance] LedoitWolf failed: {e}")
        return None


def sample_cov(
    returns: pd.DataFrame,
    min_periods: int = 63,
    annualize: bool = True,
) -> Optional[np.ndarray]:
    """Simple sample covariance fallback."""
    if returns is None or returns.empty:
        return None
    clean = returns.dropna(how="any")
    if len(clean) < min_periods:
        return None
    cov = clean.cov().values
    if annualize:
        cov = cov * 252
    return cov


def get_cov(
    returns: pd.DataFrame,
    min_periods: int = 63,
    annualize: bool = True,
) -> Optional[np.ndarray]:
    """
    Return Ledoit-Wolf covariance, falling back to sample covariance.
    Returns None if insufficient data.
    """
    cov = ledoit_wolf_cov(returns, min_periods=min_periods, annualize=annualize)
    if cov is not None:
        return cov
    return sample_cov(returns, min_periods=min_periods, annualize=annualize)
