"""
tests/test_catch_up_guard.py

W3 item 5 — the catch-up guard in run_all_agents.py. If the last logged run
is more than N trading days stale, the daily runner must refuse to
auto-execute unless --catch-up is explicitly passed. Unreadable/missing log
must also refuse (fail-safe).

Mocks scripts.heartbeat_check's read_last_run_date/trading_days_between —
never touches the real logs/eod_log.jsonl.
"""
from datetime import date

from unittest.mock import patch

import run_all_agents as ra


def test_stale_log_refuses(monkeypatch):
    """Last run 5 trading days ago (> default threshold 3) -> must refuse."""
    fake_missed = [date(2026, 7, 22), date(2026, 7, 23), date(2026, 7, 24),
                   date(2026, 7, 25), date(2026, 7, 26)]
    with patch("scripts.heartbeat_check.read_last_run_date", return_value=date(2026, 7, 20)), \
         patch("scripts.heartbeat_check.trading_days_between", return_value=fake_missed):
        must_refuse, missed = ra._catch_up_guard(date(2026, 7, 27))

    assert must_refuse is True
    assert len(missed) == 5


def test_fresh_log_proceeds(monkeypatch):
    """Last run yesterday, 0 missed trading days -> must NOT refuse."""
    with patch("scripts.heartbeat_check.read_last_run_date", return_value=date(2026, 7, 26)), \
         patch("scripts.heartbeat_check.trading_days_between", return_value=[]):
        must_refuse, missed = ra._catch_up_guard(date(2026, 7, 27))

    assert must_refuse is False
    assert missed == []


def test_exactly_at_threshold_does_not_refuse(monkeypatch):
    """threshold_days=3, exactly 3 missed -> spec says 'more than N' -> proceed."""
    fake_missed = [date(2026, 7, 24), date(2026, 7, 25), date(2026, 7, 26)]
    with patch("scripts.heartbeat_check.read_last_run_date", return_value=date(2026, 7, 23)), \
         patch("scripts.heartbeat_check.trading_days_between", return_value=fake_missed):
        must_refuse, missed = ra._catch_up_guard(date(2026, 7, 27), threshold_days=3)

    assert must_refuse is False
    assert len(missed) == 3


def test_one_over_threshold_refuses(monkeypatch):
    fake_missed = [date(2026, 7, 23), date(2026, 7, 24), date(2026, 7, 25), date(2026, 7, 26)]
    with patch("scripts.heartbeat_check.read_last_run_date", return_value=date(2026, 7, 22)), \
         patch("scripts.heartbeat_check.trading_days_between", return_value=fake_missed):
        must_refuse, missed = ra._catch_up_guard(date(2026, 7, 27), threshold_days=3)

    assert must_refuse is True
    assert len(missed) == 4


def test_no_prior_run_refuses(monkeypatch):
    """read_last_run_date returns None (empty/missing log) -> fail-safe refuse."""
    with patch("scripts.heartbeat_check.read_last_run_date", return_value=None):
        must_refuse, missed = ra._catch_up_guard(date(2026, 7, 27))

    assert must_refuse is True
    assert missed == []


def test_unreadable_log_refuses(monkeypatch):
    """Any exception reading the log (corrupt file, permissions, etc.) -> refuse."""
    with patch("scripts.heartbeat_check.read_last_run_date",
               side_effect=OSError("permission denied")):
        must_refuse, missed = ra._catch_up_guard(date(2026, 7, 27))

    assert must_refuse is True
    assert missed == []


def test_import_failure_refuses(monkeypatch):
    """If scripts.heartbeat_check itself can't be imported, refuse rather than crash."""
    import builtins
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "scripts.heartbeat_check" or name.startswith("scripts.heartbeat_check"):
            raise ImportError("simulated import failure")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_blocked_import):
        must_refuse, missed = ra._catch_up_guard(date(2026, 7, 27))

    assert must_refuse is True
    assert missed == []
