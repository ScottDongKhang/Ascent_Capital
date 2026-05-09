"""ascent/alpha/fundamental.py

Fundamental alpha sleeve — three accounting quality signals:
  gross_profitability (Novy-Marx 2013)  — long high GP/Assets
  accruals            (Sloan 1996)       — long low (inverted)
  asset_growth        (Cooper 2008)      — long low (inverted)

52-week high removed: it is a price momentum signal, not a fundamental.
Keeping it here caused ~20% momentum double-counting with the trend sleeve.
Returns empty DataFrame when fundamental cache is absent (sleeve skipped by stack).
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std  = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)


def fundamental_alpha(features: dict) -> pd.DataFrame:
    """
    Build fundamental alpha composite: gross_profitability, accruals, asset_growth.
    Returns empty DataFrame when fundamental data is absent — stack skips and renormalizes.
    """
    close = features.get("close")
    if close is None or close.empty:
        return pd.DataFrame()

    components = []

    for key, invert in [("gross_profitability", False), ("accruals", True), ("asset_growth", True)]:
        if key in features:
            try:
                df = features[key].reindex(close.index, method="ffill").reindex(columns=close.columns)
                z = _cs_zscore(df)
                components.append(-z if invert else z)
                log.info("fundamental_alpha: %s loaded", key)
            except Exception as e:
                log.warning("fundamental_alpha: %s failed: %s", key, e)

    if not components:
        return pd.DataFrame()

    return sum(components) / len(components)
