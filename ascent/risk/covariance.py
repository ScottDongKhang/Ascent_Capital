"""
Ascent Capital — Covariance Estimation
Robust covariance with shrinkage for portfolio construction.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def sample_covariance(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling sample covariance matrix (annualized)."""
    recent = returns.tail(window).dropna(axis=1, how="all")
    cov = recent.cov() * 252
    return cov


def shrinkage_covariance(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance estimator.
    More stable than sample covariance, especially with many assets.
    """
    try:
        from sklearn.covariance import LedoitWolf
        recent = returns.tail(window).dropna(axis=1, how="all").dropna()
        if len(recent) < 10 or recent.shape[1] < 2:
            return sample_covariance(returns, window)

        lw = LedoitWolf()
        lw.fit(recent.values)
        cov = pd.DataFrame(
            lw.covariance_ * 252,  # annualize
            index=recent.columns,
            columns=recent.columns,
        )
        return cov
    except ImportError:
        return sample_covariance(returns, window)


def correlation_matrix(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling correlation matrix."""
    recent = returns.tail(window).dropna(axis=1, how="all")
    return recent.corr()
