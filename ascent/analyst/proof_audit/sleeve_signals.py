"""Registry of pure alpha-sleeve signal functions used by the proof audit.

Each entry takes the same `features: dict[str, pandas.DataFrame]` shape ascent/alpha/stack.py
passes to its sleeve functions, and returns a date x symbol DataFrame of raw (pre-normalization)
signal values. ml, llm_fundamental and narrative are deliberately absent -- see the "Known scope
limits" section of the proof-audit plan.

The volatility formula is duplicated from ascent/alpha/stack.py's inline vol-regime block (not
imported, since it isn't a standalone function there) -- kept in sync manually; it is 3 lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ascent.alpha.trend import trend_alpha
from ascent.alpha.meanrev import meanrev_alpha
from ascent.alpha.statarb import statarb_alpha
from ascent.alpha.fundamental import fundamental_alpha
from ascent.alpha.earnings import earnings_alpha
from ascent.alpha.analyst import analyst_alpha
from ascent.alpha.options_flow import options_flow_alpha
from ascent.alpha.insider import insider_alpha
from ascent.alpha.short_interest import short_interest_alpha
from ascent.alpha.altdata_alpha import altdata_alpha
from ascent.alpha.earnings_tone import earnings_tone_alpha
from ascent.alpha.stack import _load_sector_map


def _volatility_signal(features: dict) -> pd.DataFrame:
    if "vol_of_vol_21d" in features and "vol_trend_10d" in features:
        vov = features["vol_of_vol_21d"].copy().replace(0, np.nan)
        vtrnd = features["vol_trend_10d"].copy()
        return -vtrnd / (vov + 1e-6)
    if "vol_21d" in features:
        return -features["vol_21d"].copy()
    return pd.DataFrame()


def _statarb_signal(features: dict) -> pd.DataFrame:
    return statarb_alpha(features, sector_map=_load_sector_map())


SLEEVE_SIGNAL_FUNCS = {
    "trend": trend_alpha,
    "meanrev": meanrev_alpha,
    "volatility": _volatility_signal,
    "statarb": _statarb_signal,
    "fundamental": fundamental_alpha,
    "earnings": earnings_alpha,
    "analyst": analyst_alpha,
    "options_flow": options_flow_alpha,
    "insider": insider_alpha,
    "short_interest": short_interest_alpha,
    "altdata": lambda features: altdata_alpha(features=features),
    "earnings_tone": earnings_tone_alpha,
}
