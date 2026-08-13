"""Point-in-time forward returns and eligible-date folds.

No look-ahead: forward_return_matrix's row for date T is the return realized from T to the NEXT
row in the index, never from T-1 to T. eligible_dates additionally filters to dates where the
point-in-time universe (ascent/data/universe.py::get_universe_on_date) is large enough for a
cross-sectional IC to be meaningful.
"""
from __future__ import annotations

import pandas as pd

from ascent.data.universe import get_universe_on_date


def forward_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Next-row simple return per column. Last row is NaN -- it has no next day."""
    return prices.pct_change().shift(-1)


def eligible_dates(
    prices: pd.DataFrame, min_universe_size: int = 20
) -> list[pd.Timestamp]:
    """Dates with a next-day return available AND a large-enough point-in-time universe."""
    if len(prices.index) < 2:
        return []
    candidate_dates = prices.index[:-1]  # last date has no forward return
    out = []
    for d in candidate_dates:
        universe = get_universe_on_date(d)
        if len(universe) >= min_universe_size:
            out.append(d)
    return out
