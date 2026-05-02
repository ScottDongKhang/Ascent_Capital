"""ascent/alpha/analyst.py

Analyst revision momentum alpha sleeve.

Long names with accelerating analyst upgrades, short names with downgrades.
Signal = rolling 63-day net revision score, cross-sectionally z-scored.

Academic basis: Jegadeesh & Kim (2006) show analyst revision momentum
has IC ~4-6% and is orthogonal to price momentum — it captures
information-update flow that precedes price moves.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def analyst_alpha(features: dict) -> pd.DataFrame:
    """
    Analyst revision momentum.

    Expects features["analyst_revision"]: DataFrame(dates × symbols)
    where values are rolling 63-day net upgrade scores (upgrades minus
    downgrades), computed causal in build_analyst_panel().

    Returns cross-sectional z-scored alpha of same shape.
    Returns empty DataFrame if feature is absent or all-zero.
    """
    if "analyst_revision" not in features:
        return pd.DataFrame()

    sig = features["analyst_revision"].copy()
    if sig.empty:
        return pd.DataFrame()

    # Drop columns that are entirely zero — no analyst coverage
    sig = sig.loc[:, (sig != 0).any(axis=0)]
    if sig.empty:
        return pd.DataFrame()

    # Cross-sectional z-score: each row (date) standardised across symbols
    mu = sig.mean(axis=1)
    sd = sig.std(axis=1).replace(0, np.nan)
    zscore = sig.sub(mu, axis=0).div(sd, axis=0).fillna(0)

    log.info("analyst_alpha: shape=%s", zscore.shape)
    return zscore
