"""ascent/alpha/short_interest.py

Short interest / squeeze potential alpha sleeve.

Signal: cross-sectional percentile rank of short_pct_float (% of float short).
High short interest → squeeze potential → contrarian long signal.

This is a contrarian signal, not a fundamental quality signal. It works because:
  (a) heavily shorted names that hold up attract covering flows (squeeze)
  (b) combined with momentum, high-short names trending up have outsized returns

Cross-sectionally z-scored. Only symbols with actual short interest data are used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def short_interest_alpha(features: dict) -> pd.DataFrame:
    """
    Cross-sectional short-squeeze alpha.

    Reads "short_pct_float" from features dict.
    Returns empty DataFrame if feature absent or all-NaN.
    """
    panel = features.get("short_pct_float")
    if panel is None or (hasattr(panel, "empty") and panel.empty):
        return pd.DataFrame()

    sig = panel.copy()
    # Drop columns with no data
    sig = sig.dropna(how="all", axis=1)
    sig = sig.loc[:, (sig != 0).any(axis=0)]
    if sig.empty:
        return pd.DataFrame()

    # Cross-sectional z-score (high short interest = positive alpha = contrarian long)
    mu = sig.mean(axis=1)
    sd = sig.std(axis=1).replace(0, np.nan)
    zscore = sig.sub(mu, axis=0).div(sd, axis=0).fillna(0)
    return zscore
