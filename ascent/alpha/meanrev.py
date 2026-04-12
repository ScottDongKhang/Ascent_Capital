"""
Ascent Capital — Alpha: Mean Reversion
Short-term reversal signal.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def meanrev_alpha(features: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Mean reversion alpha: buy oversold, sell overbought.
    Inverts short-term momentum signals.
    """
    components = []
    weights = []

    # Short-term reversal (5-day): negative momentum = buy signal
    if "mom_5d" in features:
        rev = -features["mom_5d"].copy()  # invert: losers become winners
        rev = _cs_normalize(rev)
        components.append(rev)
        weights.append(0.35)

    # Z-score: buy low z-score (below mean)
    if "zscore_20d" in features:
        z = -features["zscore_20d"].copy()  # low z-score = buy
        z = _cs_normalize(z)
        components.append(z)
        weights.append(0.35)

    # RSI: buy oversold
    if "rsi_14" in features:
        rsi_signal = -(features["rsi_14"].copy() - 50) / 50  # <50 = buy signal
        rsi_signal = _cs_normalize(rsi_signal)
        components.append(rsi_signal)
        weights.append(0.15)

    # Bollinger %B: buy at lower band
    if "bb_pct_20d" in features:
        bb = -(features["bb_pct_20d"].copy() - 0.5)  # <0.5 = buy
        bb = _cs_normalize(bb)
        components.append(bb)
        weights.append(0.15)

    if not components:
        return pd.DataFrame()

    total_w = sum(weights)
    alpha = sum(c * (w / total_w) for c, w in zip(components, weights))
    return _cs_normalize(alpha)


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)
