"""
Ascent Capital — Alpha: Trend Following
Generates trend-following alpha scores from features.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def trend_alpha(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Composite trend-following alpha.
    Combines momentum, MACD, and SMA crossover signals.
    Returns: DataFrame(dates × symbols) with alpha scores.
    """
    components = []
    weights = []

    # Medium-term momentum (highest weight)
    if "mom_63d" in features:
        # Winsorize and normalize cross-sectionally
        mom = features["mom_63d"].copy()
        mom = _cs_normalize(mom)
        components.append(mom)
        weights.append(0.30)

    # Long-term momentum
    if "mom_126d" in features:
        mom = _cs_normalize(features["mom_126d"].copy())
        components.append(mom)
        weights.append(0.20)

    # Short-term momentum (lower weight — more noise)
    if "mom_21d" in features:
        mom = _cs_normalize(features["mom_21d"].copy())
        components.append(mom)
        weights.append(0.15)

    # MACD histogram
    if "macd_hist" in features:
        macd = _cs_normalize(features["macd_hist"].copy())
        components.append(macd)
        weights.append(0.20)

    # SMA crossover
    if "sma_cross_10_50" in features:
        sma = _cs_normalize(features["sma_cross_10_50"].copy())
        components.append(sma)
        weights.append(0.15)

    if not components:
        return pd.DataFrame()

    # Weighted combination
    total_w = sum(weights)
    alpha = sum(c * (w / total_w) for c, w in zip(components, weights))

    return alpha


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score normalization (per row)."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)
