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
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    # Determine output shape from whichever feature is available.
    for key in ("mom_63d", "mom_skip1m", "mom_126d", "mom_21d", "macd_hist", "sma_cross_10_50"):
        if key in features and not features[key].empty:
            ref = features[key]
            rng = np.random.default_rng(seed=42)
            return pd.DataFrame(
                rng.standard_normal(ref.shape),
                index=ref.index,
                columns=ref.columns,
            )
    return pd.DataFrame()


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score normalization (per row)."""
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)
