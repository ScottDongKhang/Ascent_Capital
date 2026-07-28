"""Market-time helpers.

This host does not run in market time. Any conversion that goes through the
host's local zone silently shifts dates, and at UTC+7 a US session that closes
at 16:00 ET lands on the FOLLOWING calendar day. Two consequences seen in this
repo: equity bars published under weekend dates, and day-level returns
attributed to the wrong session inside `get_portfolio_history()` — the series
the rebalance recap is required to quote.

Use these helpers instead of the naive forms:

    date.today()                       ->  market_today()
    datetime.now()                     ->  market_now()
    datetime.fromtimestamp(ts).date()  ->  market_date_from_epoch(ts)

`market_today()` returns a *calendar* date in market time. It is not a trading
day: weekends and holidays come back unchanged. Callers that need a trading day
must consult a market calendar as well (see `rebalance_calendar.csv` and the
weekday-rollback helpers in the pipeline).
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

__all__ = ["MARKET_TZ", "market_now", "market_today", "market_date_from_epoch"]

#: US equity market timezone. Handles EST/EDT transitions automatically.
MARKET_TZ = ZoneInfo("America/New_York")


def market_now(now: dt.datetime | None = None) -> dt.datetime:
    """Current instant expressed in market time.

    Accepts an explicit instant so callers and tests can inject one. A naive
    ``now`` is interpreted as UTC rather than host-local, because host-local is
    the very assumption this module exists to remove.
    """
    if now is None:
        return dt.datetime.now(tz=MARKET_TZ)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(MARKET_TZ)


def market_today(now: dt.datetime | None = None) -> dt.date:
    """The current US market calendar date.

    Note this is a *calendar* date in market time, not a guaranteed trading
    day: it still returns Saturdays and holidays. Callers that need a trading
    day must additionally consult a market calendar.
    """
    return market_now(now).date()


def market_date_from_epoch(ts: float) -> dt.date:
    """The market calendar date of a UNIX epoch timestamp.

    Use for any vendor timestamp (Alpaca, etc.). Never
    ``datetime.fromtimestamp(ts).date()``, which silently uses host-local time
    and shifts every post-close bar forward a day on a UTC+7 host.
    """
    return dt.datetime.fromtimestamp(ts, tz=MARKET_TZ).date()
