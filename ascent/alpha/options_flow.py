"""ascent/alpha/options_flow.py

Options flow alpha sleeve.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def options_flow_alpha(features: dict) -> pd.DataFrame:
    """
    Cross-sectional options-flow alpha.

    Reads "iv_skew" and "put_call_ratio" from features dict.
    Returns empty DataFrame if neither feature is present.
    """
    # [PROPRIETARY LOGIC REDACTED FOR PUBLIC REPO]
    skew = features.get("iv_skew")
    pcr  = features.get("put_call_ratio")

    ref = None
    if skew is not None and hasattr(skew, "empty") and not skew.empty:
        ref = skew
    elif pcr is not None and hasattr(pcr, "empty") and not pcr.empty:
        ref = pcr

    if ref is None:
        return pd.DataFrame()

    ref = ref.loc[:, (ref != 0).any(axis=0)].dropna(how="all", axis=1)
    if ref.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        rng.standard_normal(ref.shape),
        index=ref.index,
        columns=ref.columns,
    )
