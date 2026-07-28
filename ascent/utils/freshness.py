"""Expiry for AI-authored priors that steer the quant pipeline.

`ascent/main.py` consumes two files written by the AI PM's Phase 1:

    data_cache/ai_prethesis_latest.json    -> floors conviction alpha, zeroes the avoid list
    data_cache/ai_regime_assessment.json   -> blends the regime label + risk multiplier

Both are read on every pipeline run, and neither read was date-gated. On the
2026-07-27 run the pre-thesis on disk was dated 2026-06-24 — 33 days old — and
six of its ten conviction names were in the executed book. Because these two
channels bypass the earned-authority budget entirely, a month-old opinion was
steering position sizing with no cap and no expiry.

It also contaminates the measurement: Track A-star ("pure quant") is snapshotted
from a pipeline that has already consumed these files, so the baseline used to
judge the AI PM contains the AI PM's stale opinions. Gating on age does not fully
remove that — a *fresh* prior still influences A-star — but it bounds the damage
to the current rebalance cycle instead of letting one file steer indefinitely.

Fails CLOSED: anything unparseable, absent, or future-dated is treated as stale.
A prior that cannot prove its age does not get to move the portfolio.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from ascent.utils.market_time import market_today

__all__ = ["AI_PRIOR_MAX_AGE_DAYS", "ai_prior_is_fresh"]

#: Scheduled rebalances sit ~14 calendar days apart (see rebalance_calendar.csv),
#: so one cycle is the longest a Phase-1 view can still be called "current".
#: Deliberately calendar days, not trading days: the question is how stale the
#: *opinion* is, and markets move over weekends too.
AI_PRIOR_MAX_AGE_DAYS = 14


def ai_prior_is_fresh(
    as_of_date: Any,
    today: Optional[dt.date] = None,
    max_age_days: int = AI_PRIOR_MAX_AGE_DAYS,
) -> bool:
    """True if an AI prior stamped `as_of_date` may still steer the pipeline.

    `as_of_date` is whatever was in the JSON — commonly an ISO string, possibly
    missing or malformed. `today` defaults to the market date, not the host date
    (this host runs at UTC+7; see ascent/utils/market_time.py).
    """
    if not isinstance(as_of_date, str):
        return False
    try:
        stamped = dt.date.fromisoformat(as_of_date.strip())
    except (ValueError, AttributeError):
        return False

    ref = today if today is not None else market_today()
    age = (ref - stamped).days
    if age < 0:
        return False  # future-dated: clock or --date override bug, not signal
    return age <= max_age_days
