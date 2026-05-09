"""ascent/alpha/earnings.py

Post-Earnings Announcement Drift (PEAD) alpha sleeve.

Long positive earnings surprises, short negative ones.
Signal decays over ~63 trading days via forward-fill in the feature panel.

Momentum neutralization: per-date OLS regression of earnings_surprise on
mom_126d removes the momentum beta before z-scoring, so the sleeve captures
pure earnings drift rather than riding the same exposure as the trend sleeve.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def earnings_alpha(features: dict) -> pd.DataFrame:
    """
    Post-earnings announcement drift (PEAD), momentum-neutralized.

    Expects features["earnings_surprise"]: DataFrame(dates × symbols).
    Optionally features["mom_126d"] for momentum neutralization.

    Returns cross-sectional z-scored residuals of same shape.
    Returns empty DataFrame if earnings_surprise is absent.
    """
    if "earnings_surprise" not in features:
        return pd.DataFrame()

    sig = features["earnings_surprise"].copy()
    if sig.empty:
        return pd.DataFrame()

    mom = features.get("mom_126d")
    mom_neutralized = False

    if mom is not None and not mom.empty:
        residuals = sig.copy()
        mom_aligned = mom.reindex(index=sig.index, columns=sig.columns)
        for dt in sig.index:
            y = sig.loc[dt].dropna()
            x = mom_aligned.loc[dt].reindex(y.index).dropna()
            common = y.index.intersection(x.index)
            if len(common) < 5:
                continue
            yc = y.reindex(common)
            xc = x.reindex(common)
            xc_dm = xc - xc.mean()
            var_x = (xc_dm ** 2).sum()
            if var_x < 1e-8:
                continue
            beta = (xc_dm * (yc - yc.mean())).sum() / var_x
            residuals.loc[dt, common] = (yc - beta * xc_dm).values
        sig = residuals
        mom_neutralized = True

    mu = sig.mean(axis=1)
    sd = sig.std(axis=1).replace(0, np.nan)
    zscore = sig.sub(mu, axis=0).div(sd, axis=0).fillna(0)

    log.info("earnings_alpha: shape=%s mom_neutralized=%s", zscore.shape, mom_neutralized)
    return zscore
