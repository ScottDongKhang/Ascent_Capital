"""ascent/alpha/earnings.py

Post-Earnings Announcement Drift (PEAD) alpha sleeve.

Long positive earnings surprises, short negative ones.
Signal decays over ~63 trading days via forward-fill in the feature panel.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def earnings_alpha(features: dict) -> pd.DataFrame:
    """
    Post-earnings announcement drift (PEAD).
    Long positive surprise, short negative.

    Expects features["earnings_surprise"]: DataFrame(dates × symbols)
    where values are raw surprise_pct, forward-filled up to 63 days.

    Returns cross-sectional z-scored alpha of same shape.
    Returns empty DataFrame if feature is absent.
    """
    if "earnings_surprise" not in features:
        return pd.DataFrame()

    sig = features["earnings_surprise"].copy()
    if sig.empty:
        return pd.DataFrame()

    # Cross-sectional z-score: each row (date) standardised across symbols
    mu = sig.mean(axis=1)
    sd = sig.std(axis=1).replace(0, np.nan)
    zscore = sig.sub(mu, axis=0).div(sd, axis=0).fillna(0)

    log.info("earnings_alpha: shape=%s", zscore.shape)
    return zscore
