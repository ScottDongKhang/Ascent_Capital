"""
Tests for ascent/execution/kill_switch.py.

Covers:
  - Thresholds are at the tightened, capital-preservation-first levels.
  - The monthly circuit breaker (_get_month_start_nav / check()'s monthly
    branch): normal month, under-threshold loss, over-threshold loss (trips),
    and the "no NAV yet this month" edge case (must not raise).
  - Bug fix #1: when the monthly branch trips, drawdown_at_trip records the
    monthly drawdown (not the whole-book peak drawdown), and trip_reason_kind
    distinguishes "peak" vs "monthly".
  - Bug fix #2: "current month" is derived from the actual current date
    (via ascent.utils.market_time.market_today(), injectable through the
    `now` parameter), never from history[-1]['date'] -- including the case
    where today is day 1 of a new month and history only has last month's
    data.
"""
import json
from datetime import datetime

import pytest

from ascent.execution import kill_switch
from ascent.utils.market_time import MARKET_TZ


def _write_log(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    """Point every path the module touches at a scratch dir, and make sure
    each test starts from a fresh, untripped state."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setattr(kill_switch, "LOG_DIR", log_dir)
    monkeypatch.setattr(kill_switch, "EOD_LOG_PATH", log_dir / "eod_log.jsonl")
    monkeypatch.setattr(kill_switch, "KS_STATE_PATH", log_dir / "kill_switch_state.json")
    yield log_dir


def _row(run_date, nav):
    return {"run_date": run_date, "portfolio_value": nav}


def _mkt(y, m, d, hour=15):
    """Build a tz-aware 'now' in market time, mid-day to avoid any
    midnight-boundary ambiguity."""
    return datetime(y, m, d, hour, 0, tzinfo=MARKET_TZ)


class TestThresholds:
    def test_tightened_thresholds(self):
        assert kill_switch.SOFT_WARN_PCT == pytest.approx(0.05)
        assert kill_switch.HARD_STOP_PCT == pytest.approx(0.12)
        assert kill_switch.MONTHLY_SOFT_HALT_PCT == pytest.approx(0.06)


class TestMonthlyCircuitBreaker:
    def test_normal_month_no_trigger(self, _isolated_paths):
        _write_log(kill_switch.EOD_LOG_PATH, [
            _row("2026-08-03", 100_000),
            _row("2026-08-10", 101_000),
            _row("2026-08-17", 102_500),
        ])
        result = kill_switch.check(current_nav=103_000, now=_mkt(2026, 8, 20))
        assert result["status"] == "ok"
        # Up on the month (gain), so "drawdown" from month start is negative.
        assert result["monthly_drawdown"] == pytest.approx(-0.03)
        assert result["month_start_nav"] == 100_000

    def test_month_loss_under_threshold_no_trigger(self, _isolated_paths):
        _write_log(kill_switch.EOD_LOG_PATH, [
            _row("2026-08-03", 100_000),
            _row("2026-08-10", 98_000),
        ])
        # 4% MTD drawdown, below the 6% monthly threshold.
        result = kill_switch.check(current_nav=96_000, now=_mkt(2026, 8, 11))
        assert result["status"] in ("ok", "warn")
        assert result["monthly_drawdown"] == pytest.approx(0.04)

    def test_month_loss_over_threshold_triggers(self, _isolated_paths):
        _write_log(kill_switch.EOD_LOG_PATH, [
            _row("2026-08-03", 100_000),
            _row("2026-08-10", 97_000),
        ])
        # 7% MTD drawdown, above the 6% monthly threshold.
        with pytest.raises(kill_switch.KillSwitchTriggered) as exc_info:
            kill_switch.check(current_nav=93_000, now=_mkt(2026, 8, 11))

        assert "monthly" in str(exc_info.value).lower() or "month" in str(exc_info.value).lower()

        state = kill_switch._load_state()
        assert state["tripped"] is True
        assert "Month-to-date" in state["tripped_reason"] or "month" in state["tripped_reason"].lower()

    def test_no_trigger_on_first_day_of_month_missing_history(self, _isolated_paths):
        """No NAV history at all for the current trading month yet (e.g. the
        very first trading day of a new month before any EOD run has logged
        a NAV) -- must not raise, must be treated as insufficient data."""
        # No log file at all.
        result = kill_switch.check(current_nav=100_000, now=_mkt(2026, 9, 1))
        assert result["status"] == "ok"

    def test_get_month_start_nav_empty_history(self):
        # With no history, the current month still comes from "now" (or the
        # real current date) -- month_key is never None just because
        # history is empty; only month_start_nav is.
        month_key, month_start_nav = kill_switch._get_month_start_nav(
            [], now=_mkt(2026, 9, 1)
        )
        assert month_key == "2026-09"
        assert month_start_nav is None

    def test_get_month_start_nav_picks_earliest_in_month(self):
        history = [
            {"date": "2026-07-30", "nav": 90_000},
            {"date": "2026-08-05", "nav": 100_000},
            {"date": "2026-08-12", "nav": 105_000},
        ]
        month_key, month_start_nav = kill_switch._get_month_start_nav(
            history, now=_mkt(2026, 8, 20)
        )
        assert month_key == "2026-08"
        assert month_start_nav == 100_000

    def test_does_not_double_trip_when_only_peak_breached(self, _isolated_paths):
        """A book far under its all-time peak (peak-based hard stop fires)
        should not also need the monthly reason -- the peak-based trip takes
        precedence and its own reason text is used, independent of the
        monthly branch never having been reached."""
        _write_log(kill_switch.EOD_LOG_PATH, [
            _row("2026-08-01", 100_000),
            _row("2026-08-03", 130_000),  # peak
        ])
        # Down 13% from the 130k peak -> hard stop fires before the monthly
        # branch is evaluated at all.
        with pytest.raises(kill_switch.KillSwitchTriggered) as exc_info:
            kill_switch.check(current_nav=113_000, now=_mkt(2026, 8, 4))
        assert "hard stop" in str(exc_info.value).lower()


class TestBugFix1_TripAttribution:
    """drawdown_at_trip and trip_reason_kind must reflect which check
    actually fired, not always the whole-book peak drawdown."""

    def test_monthly_trip_records_monthly_drawdown_not_peak_drawdown(self, _isolated_paths):
        # All-time peak (105_000, from July) is higher than the month-start
        # NAV (100_000, from August) -- the book had already pulled back
        # before this month started. current_nav is chosen so peak-to-trough
        # drawdown stays under HARD_STOP_PCT (12%) while monthly drawdown
        # from the August baseline exceeds MONTHLY_SOFT_HALT_PCT (6%): only
        # the monthly branch fires, and the two drawdown values are
        # genuinely different numbers -- the real regression case for bug #1.
        _write_log(kill_switch.EOD_LOG_PATH, [
            _row("2026-07-01", 105_000),  # all-time peak
            _row("2026-08-01", 100_000),  # August month-start NAV
        ])
        current_nav = 94_000
        # peak-to-trough drawdown = (105000-94000)/105000 = 10.48% (< 12%)
        # monthly drawdown        = (100000-94000)/100000 = 6.00%  (>= 6%)
        with pytest.raises(kill_switch.KillSwitchTriggered):
            kill_switch.check(current_nav=current_nav, now=_mkt(2026, 8, 5))

        state = kill_switch._load_state()
        assert state["trip_reason_kind"] == "monthly"
        expected_monthly_dd = round((100_000 - current_nav) / 100_000, 4)
        expected_peak_dd = round((105_000 - current_nav) / 105_000, 4)
        assert expected_monthly_dd != expected_peak_dd  # sanity: they truly differ
        assert state["drawdown_at_trip"] == pytest.approx(expected_monthly_dd)
        assert state["drawdown_at_trip"] != pytest.approx(expected_peak_dd)

    def test_peak_trip_records_peak_drawdown_and_kind(self, _isolated_paths):
        _write_log(kill_switch.EOD_LOG_PATH, [
            _row("2026-08-01", 100_000),
            _row("2026-08-03", 130_000),  # peak
        ])
        with pytest.raises(kill_switch.KillSwitchTriggered):
            kill_switch.check(current_nav=113_000, now=_mkt(2026, 8, 4))

        state = kill_switch._load_state()
        assert state["trip_reason_kind"] == "peak"
        expected_peak_dd = round((130_000 - 113_000) / 130_000, 4)
        assert state["drawdown_at_trip"] == pytest.approx(expected_peak_dd)


class TestBugFix2_MonthDeterminedFromActualDate:
    """_get_month_start_nav must use the real 'now' (market_today), never
    history[-1]['date'] -- this is the eod_runner ordering bug: check() is
    called BEFORE today's NAV is appended to history, so history[-1] is
    always a prior day and, on day 1 of a new month, still last month."""

    def test_new_month_day_one_with_only_last_month_history(self, _isolated_paths):
        # history only has last month's (July's) NAV entries -- simulating
        # check() being called on 2026-08-01 before any August entry exists
        # in eod_log.jsonl.
        history = [
            {"date": "2026-07-05", "nav": 100_000},
            {"date": "2026-07-20", "nav": 102_000},
            {"date": "2026-07-31", "nav": 101_000},
        ]
        month_key, month_start_nav = kill_switch._get_month_start_nav(
            history, now=_mkt(2026, 8, 1)
        )
        # BUGGY behavior would have returned month_key="2026-07" (from
        # history[-1]) and month_start_nav=100_000 (July's month-start).
        # Correct behavior: current month is August (from actual date), and
        # since no August NAV has been logged yet, month_start_nav is None.
        assert month_key == "2026-08"
        assert month_start_nav is None

    def test_new_month_check_does_not_misattribute_baseline(self, _isolated_paths):
        # Full check() integration: only July history exists, "now" is
        # August 1st -- must be treated as "no data yet this month", not
        # trip using July's baseline.
        _write_log(kill_switch.EOD_LOG_PATH, [
            _row("2026-07-05", 100_000),
            _row("2026-07-31", 101_000),
        ])
        result = kill_switch.check(current_nav=90_000, now=_mkt(2026, 8, 1))
        # Not "halted" -- the missing-month-data case must not trip anything
        # (a soft peak-to-trough warning is a separate, unrelated signal).
        assert result["status"] in ("ok", "warn")
        assert result["month_start_nav"] is None
        assert result["monthly_drawdown"] is None

    def test_current_month_uses_now_even_with_later_history_present(self, _isolated_paths):
        # Regression guard: even when history's last row is NOT month-end,
        # the month is still taken from `now`, not from history[-1].
        history = [
            {"date": "2026-08-01", "nav": 100_000},
            {"date": "2026-08-15", "nav": 99_000},
        ]
        month_key, month_start_nav = kill_switch._get_month_start_nav(
            history, now=_mkt(2026, 9, 3)
        )
        # "now" says September; no September entries exist in history yet.
        assert month_key == "2026-09"
        assert month_start_nav is None
