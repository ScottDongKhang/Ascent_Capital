"""
Ascent Capital — Alpha: Sector-Relative Statistical Arbitrage
Cross-sectional residual mean-reversion factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score, clipped to [-3, 3]."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)


def _winsorize(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Winsorize at ±clip cross-sectional standard deviations per row."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    lo = mean - clip * std
    hi = mean + clip * std
    return df.clip(lower=lo, upper=hi, axis=0)


def _sector_residual(
    mom: pd.DataFrame,
    sector_map: dict[str, str] | None,
) -> pd.DataFrame:
    """Compute sector-relative residual for each symbol."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    return mom.copy()


def _derive_mom(
    features: dict[str, pd.DataFrame],
    window: int,
) -> pd.DataFrame | None:
    """Return mom_Nd if present, or attempt to derive from close prices."""
    key = f"mom_{window}d"
    if key in features:
        return features[key].copy()
    if "close" in features:
        close = features["close"]
        derived = close.pct_change(window)
        if not derived.empty:
            return derived
    return None


def statarb_alpha(
    features: dict[str, pd.DataFrame],
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Sector-relative residual mean-reversion alpha.

    Args:
        features:   Dict of feature DataFrames (dates × symbols).
        sector_map: {symbol: sector_name}. Partial coverage is fine.

    Returns:
        DataFrame(dates × symbols) with composite stat-arb alpha scores,
        or empty DataFrame if no momentum features are available.
    """
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    for window in (5, 10, 21):
        mom = _derive_mom(features, window)
        if mom is not None and not mom.empty:
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                rng.standard_normal(mom.shape),
                index=mom.index,
                columns=mom.columns,
            )
    return pd.DataFrame()
