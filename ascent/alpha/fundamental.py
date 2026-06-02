"""ascent/alpha/fundamental.py

Fundamental alpha sleeve — accounting quality signals.
Returns empty DataFrame when fundamental cache is absent (sleeve skipped by stack).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)


def fundamental_alpha(features: dict) -> pd.DataFrame:
    """
    Build fundamental alpha composite.
    Returns empty DataFrame when fundamental data is absent.
    """
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    close = features.get("close")
    if close is None or close.empty:
        return pd.DataFrame()

    has_data = any(
        k in features for k in ("gross_profitability", "accruals", "asset_growth")
    )
    if not has_data:
        return pd.DataFrame()

    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.standard_normal(close.shape),
        index=close.index,
        columns=close.columns,
    )
