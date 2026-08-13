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
        # get_universe_on_date compares against tz-naive start_date/end_date columns, so a
        # tz-aware d (prices_live pivots to a tz-aware America/New_York index) raises
        # "Invalid comparison between dtype=datetime64[us] and Timestamp". Coerce a LOCAL
        # copy for the lookup only -- tz_localize(None) drops the offset annotation without
        # converting the wall-clock time (unlike tz_convert, which would shift the calendar
        # day -- see the documented America/New_York landmine), then normalize() zeros the
        # time-of-day so the date lines up cleanly against start_date <= date <= end_date.
        # `d` itself must stay unmodified: it still has to match prices.index / fwd.index
        # keys exactly for downstream .loc lookups in wf_scorer.py.
        lookup_date = pd.Timestamp(d)
        if lookup_date.tz is not None:
            lookup_date = lookup_date.tz_localize(None).normalize()
        universe = get_universe_on_date(lookup_date)
        if len(universe) >= min_universe_size:
            out.append(d)
    return out
