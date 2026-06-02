"""ascent/alpha/analyst.py

Analyst revision momentum alpha sleeve.
Long names with accelerating analyst upgrades, short names with downgrades.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def analyst_alpha(features: dict) -> pd.DataFrame:
    """
    Analyst revision momentum.

    Expects features["analyst_revision"]: DataFrame(dates × symbols).
    Returns cross-sectional z-scored alpha of same shape.
    Returns empty DataFrame if feature is absent or all-zero.
    """
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if "analyst_revision" not in features:
        return pd.DataFrame()

    sig = features["analyst_revision"].copy()
    if sig.empty:
        return pd.DataFrame()

    sig = sig.loc[:, (sig != 0).any(axis=0)]
    if sig.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed=42)
    result = pd.DataFrame(
        rng.standard_normal(sig.shape),
        index=sig.index,
        columns=sig.columns,
    )
    log.info("analyst_alpha: shape=%s", result.shape)
    return result
