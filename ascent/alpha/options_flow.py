"""ascent/alpha/options_flow.py

Options flow alpha sleeve.

Signal composition:
    IV skew     = OTM put IV minus OTM call IV
                  Positive skew = expensive puts = bearish sentiment → negative alpha
    Put/call ratio = put_volume / total_volume
                  High = elevated put hedging = bearish → negative alpha

Both signals are inverted (bearish options activity → avoid / underweight),
z-scored cross-sectionally, and averaged.
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
    skew = features.get("iv_skew")
    pcr  = features.get("put_call_ratio")

    if (skew is None or (hasattr(skew, "empty") and skew.empty)) and \
       (pcr  is None or (hasattr(pcr,  "empty") and pcr.empty)):
        return pd.DataFrame()

    signals = []

    if skew is not None and not skew.empty:
        # Invert: high skew (bearish) → low alpha
        s_skew = -skew
        signals.append(s_skew)

    if pcr is not None and not pcr.empty:
        # Center around 0.5 and invert: PC ratio > 0.5 = bearish → negative
        s_pcr = -(pcr - 0.5)
        signals.append(s_pcr)

    if not signals:
        return pd.DataFrame()

    sig = signals[0] if len(signals) == 1 else sum(signals) / len(signals)

    # Drop columns that are all NaN or all zero (no options data available)
    sig = sig.loc[:, (sig != 0).any(axis=0)]
    sig = sig.dropna(how="all", axis=1)
    if sig.empty:
        return pd.DataFrame()

    # Cross-sectional z-score
    mu = sig.mean(axis=1)
    sd = sig.std(axis=1).replace(0, np.nan)
    zscore = sig.sub(mu, axis=0).div(sd, axis=0).fillna(0)
    return zscore
