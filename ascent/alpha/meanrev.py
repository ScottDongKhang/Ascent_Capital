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
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    for key in ("mom_5d", "zscore_20d", "rsi_14", "bb_pct_20d"):
        if key in features and not features[key].empty:
            ref = features[key]
            rng = np.random.default_rng(seed=42)
            return _cs_normalize(pd.DataFrame(
                rng.standard_normal(ref.shape),
                index=ref.index,
                columns=ref.columns,
            ))
    return pd.DataFrame()


def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)
