"""ascent/alpha/insider.py

Insider transaction alpha sleeve.
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
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    panel = features.get("insider_net_score")
    if panel is None or (hasattr(panel, "empty") and panel.empty):
        return pd.DataFrame()

    sig = panel.copy()
    sig = sig.loc[:, (sig != 0).any(axis=0)].dropna(how="all", axis=1)
    if sig.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.standard_normal(sig.shape),
        index=sig.index,
        columns=sig.columns,
    )
