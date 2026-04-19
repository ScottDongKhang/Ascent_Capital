"""ascent/alpha/fundamental.py

Fundamental alpha sleeve — four academically-validated signals:
  gross_profitability (Novy-Marx 2013)  — long high GP/Assets
  accruals            (Sloan 1996)       — long low (inverted)
  asset_growth        (Cooper 2008)      — long low (inverted)
  high_52w_pct        (George/Hwang 2004)— long near 52-week high
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
    Build fundamental alpha composite. Returns DataFrame(dates × symbols).
    Degrades gracefully to 52wk-high-only if fundamentals absent.
    """
    close = features.get("close")
    if close is None or close.empty:
        return pd.DataFrame()

    components = []

    h52 = features.get("high_52w_pct")
    if h52 is None:
        h52 = close / close.rolling(252, min_periods=63).max()
    try:
        components.append(_cs_zscore(
            h52.reindex(close.index).reindex(columns=close.columns)
        ))
        log.info("fundamental_alpha: 52-week high loaded")
    except Exception as e:
        log.warning("fundamental_alpha: 52wk high failed: %s", e)

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
