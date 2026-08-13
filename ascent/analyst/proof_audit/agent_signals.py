"""Registry of the 3 non-us-equities specialist agents' internal alpha builders.

us_equities_agent is intentionally absent -- its signal is exactly the sleeve stack Task 4
already scores; see the "Known scope limits" section of the proof-audit plan.

Each agent module keeps its alpha builder as a module-private function (`_build_*_alpha`) taking
the prices/features it fetches itself. We call those private functions directly rather than
duplicating their feature-engineering logic -- duplicating it would silently drift from what the
live agent actually runs.
"""
from __future__ import annotations

import pandas as pd

from agents.macro_agent import _build_macro_features, _build_trend_alpha as _macro_trend_alpha
from agents.international_agent import (
    _build_features as _international_features,
    _build_trend_alpha as _international_trend_alpha,
)
from agents.alternatives_agent import (
    _build_features as _alternatives_features,
    _build_alternatives_alpha,
)


def _macro_signal(prices: pd.DataFrame) -> pd.DataFrame:
    features = _build_macro_features(prices)
    return _macro_trend_alpha(features, prices)


def _international_signal(prices: pd.DataFrame) -> pd.DataFrame:
    features = _international_features(prices)
    return _international_trend_alpha(features)


def _alternatives_signal(prices: pd.DataFrame) -> pd.DataFrame:
    features = _alternatives_features(prices)
    return _build_alternatives_alpha(features, prices)


AGENT_SIGNAL_FUNCS = {
    "macro_agent": _macro_signal,
    "international_agent": _international_signal,
    "alternatives_agent": _alternatives_signal,
}
