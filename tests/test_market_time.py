"""Market-date basis.

The host runs at UTC+7. A US market session that closes at 20:00 UTC is
03:00 the NEXT day in host-local time, so `datetime.fromtimestamp(ts).date()`
silently shifts every vendor bar forward by one calendar day. That is what
produced misdated equity bars (including weekend dates) on the published
performance page, and it corrupts day-level return attribution in
`get_portfolio_history()` — the series the rebalance recap is required to use.

These tests pin the market-time basis independently of where the host runs.
"""
from __future__ import annotations

import datetime as dt
import os
from zoneinfo import ZoneInfo

import pytest

from ascent.utils.market_time import (
    MARKET_TZ,
    market_date_from_epoch,
    market_now,
    market_today,
)


def _epoch(y, m, d, hh, mm=0, tz="UTC") -> float:
    return dt.datetime(y, m, d, hh, mm, tzinfo=ZoneInfo(tz)).timestamp()


def test_market_tz_is_us_eastern():
    assert MARKET_TZ == ZoneInfo("America/New_York")


def test_close_bar_maps_to_session_date_not_next_day():
    """20:00 UTC on a Friday is 16:00 ET Friday — the Friday session."""
    ts = _epoch(2026, 6, 19, 20)  # Fri 2026-06-19, 16:00 ET
    assert market_date_from_epoch(ts) == dt.date(2026, 6, 19)


def test_close_bar_does_not_roll_into_saturday():
    """The bug's signature: a Friday close appearing as Saturday."""
    ts = _epoch(2026, 6, 19, 20)
    assert market_date_from_epoch(ts).weekday() == 4  # Friday


def test_after_midnight_utc_still_previous_session():
    """00:30 UTC Saturday is 20:30 ET Friday — still the Friday session date."""
    ts = _epoch(2026, 6, 20, 0, 30)
    assert market_date_from_epoch(ts) == dt.date(2026, 6, 19)


def test_pre_open_bar_maps_to_same_session():
    """13:35 UTC is 09:35 ET — the same calendar day."""
    ts = _epoch(2026, 6, 19, 13, 35)
    assert market_date_from_epoch(ts) == dt.date(2026, 6, 19)


def test_result_is_independent_of_host_timezone(monkeypatch):
    """Same epoch must map to the same session date from Hanoi or New York."""
    ts = _epoch(2026, 6, 19, 20)
    seen = set()
    for tz in ("Asia/Ho_Chi_Minh", "America/New_York", "UTC", "Pacific/Auckland"):
        monkeypatch.setenv("TZ", tz)
        try:
            import time
            time.tzset()
        except AttributeError:  # pragma: no cover - non-POSIX
            pytest.skip("tzset unavailable")
        seen.add(market_date_from_epoch(ts))
    assert seen == {dt.date(2026, 6, 19)}, f"host tz leaked into the result: {seen}"


def test_naive_host_conversion_would_disagree_in_utc_plus_7(monkeypatch):
    """Guard the guard: prove the naive form really is wrong at UTC+7."""
    ts = _epoch(2026, 6, 19, 20)
    monkeypatch.setenv("TZ", "Asia/Ho_Chi_Minh")
    import time
    try:
        time.tzset()
    except AttributeError:  # pragma: no cover
        pytest.skip("tzset unavailable")
    naive = dt.datetime.fromtimestamp(ts).date()
    assert naive == dt.date(2026, 6, 20), "expected the naive form to shift a day"
    assert market_date_from_epoch(ts) == dt.date(2026, 6, 19)


def test_market_now_is_tz_aware_and_in_market_tz():
    now = market_now()
    assert now.tzinfo is not None
    assert now.tzinfo.key == "America/New_York"


def test_market_today_matches_market_now_date():
    assert market_today() == market_now().date()


def test_market_today_accepts_injected_instant():
    """Callers must be able to pass an instant so behaviour is testable."""
    instant = dt.datetime(2026, 6, 20, 2, 0, tzinfo=ZoneInfo("UTC"))  # 22:00 ET Jun 19
    assert market_today(instant) == dt.date(2026, 6, 19)


def test_market_today_is_calendar_not_trading_day():
    """Documented limitation: weekends are returned, not rolled back."""
    saturday = dt.datetime(2026, 6, 20, 16, 0, tzinfo=ZoneInfo("America/New_York"))
    assert market_today(saturday) == dt.date(2026, 6, 20)
    assert market_today(saturday).weekday() == 5


# ---------------------------------------------------------------------------
# Call-site guards.
#
# The helpers above are only useful if the sites that stamp artifacts actually
# use them. These assert on source text because the alternative — importing
# run_all_agents and monkeypatching the clock — pulls in the whole engine.
# Verified to fail against the pre-fix revision of each file.
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel: str) -> str:
    with open(os.path.join(_ROOT, rel)) as f:
        return f.read()


def test_run_all_agents_root_date_is_market_derived():
    assert "else market_today()" in _src("run_all_agents.py"), \
        "main() must not fall back to date.today() for the run's root date"


def test_eod_runner_root_date_is_market_derived():
    assert "run_date or market_today()" in _src("ascent/execution/eod_runner.py")


def test_no_naive_fromtimestamp_in_date_stamping_modules():
    """A tz-less fromtimestamp() silently renders in host-local time."""
    for rel in ("ascent/execution/alpaca_broker.py",
                "scripts/generate_performance_page.py"):
        for lineno, line in enumerate(_src(rel).splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "fromtimestamp(" in stripped:
                assert "tz=" in stripped, f"{rel}:{lineno} naive fromtimestamp -> {stripped}"


def test_run_eod_holiday_gate_uses_market_date():
    """The gate ran on the host date, so at 09:00 local it tested tomorrow's
    session — skipping real trading days and permitting closed ones."""
    src = _src("scripts/run_eod.sh")
    assert "market_today()" in src
    # The comment explaining the old bug names the package; only an actual
    # import of it is the defect.
    active = [l.strip() for l in src.splitlines() if not l.strip().startswith("#")]
    assert not any("import pandas_market_calendars" in l for l in active), \
        "that package is not installed; the ImportError branch made the gate a no-op"
