"""ascent/alpha/insider.py

Insider transaction alpha sleeve.

Signal: rolling 63-day net insider purchase score.
  +1 per open-market buy event, -1 per open-market sell.
  Net positive → insiders accumulating → positive alpha.
  Net negative → insiders distributing → negative alpha.

Cross-sectionally z-scored. All-zero columns (no insider activity) are dropped.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def insider_alpha(features: dict) -> pd.DataFrame:
    """
    Cross-sectional insider-flow alpha.

    Reads "insider_net_score" from features dict.
    Returns empty DataFrame if feature absent or all-zero.
    """
    panel = features.get("insider_net_score")
    if panel is None or (hasattr(panel, "empty") and panel.empty):
        return pd.DataFrame()

    sig = panel.copy()
    # Drop all-zero columns (no insider coverage)
    sig = sig.loc[:, (sig != 0).any(axis=0)]
    sig = sig.dropna(how="all", axis=1)
    if sig.empty:
        return pd.DataFrame()

    # Cross-sectional z-score
    mu = sig.mean(axis=1)
    sd = sig.std(axis=1).replace(0, np.nan)
    zscore = sig.sub(mu, axis=0).div(sd, axis=0).fillna(0)
    return zscore
