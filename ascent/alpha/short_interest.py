"""ascent/alpha/short_interest.py

Short interest / squeeze potential alpha sleeve.
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
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    panel = features.get("short_pct_float")
    if panel is None or (hasattr(panel, "empty") and panel.empty):
        return pd.DataFrame()

    sig = panel.copy().dropna(how="all", axis=1)
    sig = sig.loc[:, (sig != 0).any(axis=0)]
    if sig.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.standard_normal(sig.shape),
        index=sig.index,
        columns=sig.columns,
    )
