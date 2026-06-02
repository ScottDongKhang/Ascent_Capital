"""ascent/alpha/earnings.py

Post-Earnings Announcement Drift (PEAD) alpha sleeve.
Long positive earnings surprises, short negative ones.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def earnings_alpha(features: dict) -> pd.DataFrame:
    """
    Post-earnings announcement drift (PEAD), momentum-neutralized.

    Expects features["earnings_surprise"]: DataFrame(dates × symbols).
    Optionally features["mom_126d"] for momentum neutralization.

    Returns cross-sectional z-scored residuals of same shape.
    Returns empty DataFrame if earnings_surprise is absent.
    """
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    if "earnings_surprise" not in features:
        return pd.DataFrame()

    sig = features["earnings_surprise"]
    if sig.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed=42)
    log.info("earnings_alpha: shape=%s mom_neutralized=False (redacted)", sig.shape)
    return pd.DataFrame(
        rng.standard_normal(sig.shape),
        index=sig.index,
        columns=sig.columns,
    )
