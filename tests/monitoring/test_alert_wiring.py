"""
tests/monitoring/test_alert_wiring.py

Covers the fix for "three alert paths cannot fire" (PROJECT_STATUS.md §1.5):

1. run_all_agents.py used to call check_alerts() with zero arguments, wrapped
   in a bare `except: pass`. Since every threshold in check_alerts() derives
   from args that default to None, that call could only ever return [] — and
   even a real exception would have been invisible. The fix moves the real
   call into `_run_daily_alert_checks()`, invoked from `_log_holdings()`
   where equity/positions/factor-exposures/sleeve-IC actually exist.
2. send_system_alive_ping() had no production caller. The fix wires it into
   `_run_daily_alert_checks()` so it fires once per completed run.

These tests reprove the bug first (a fabricated breach condition must
actually produce a non-empty alert list through the wired-up path, not just
through check_alerts() called directly with kwargs — that part already
worked and was never the bug) and then verify the fix.
"""
import json

import pytest

import run_all_agents as ra
from ascent.monitoring import alert_system


@pytest.fixture(autouse=True)
def _isolate_alert_files(tmp_path, monkeypatch):
    alerts_log = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setattr(alert_system, "ALERTS_LOG", alerts_log)
    monkeypatch.setattr(alert_system, "ALERT_STATE_FILE", state_file)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    return alerts_log


class _FakeToday:
    """Minimal stand-in for a datetime.date, only needs .isoformat()."""
    def isoformat(self):
        return "2026-07-31"


def test_old_call_site_was_a_permanent_noop():
    """
    Reproduces the original bug in isolation: check_alerts() called with no
    arguments (exactly as the old run_all_agents.py call site did) can never
    return a non-empty list, no matter what real-world breach exists, because
    every threshold derives from args that default to None.
    """
    assert alert_system.check_alerts() == []


def test_run_daily_alert_checks_fires_drawdown_alert_on_fabricated_breach(tmp_path, monkeypatch):
    """
    The wired-up path must actually detect a real breach. Fabricate a
    holdings_log.jsonl with a peak equity far above current equity (a >10%
    drawdown) and confirm `_run_daily_alert_checks` drives a CRITICAL
    drawdown alert into alerts.jsonl end-to-end.
    """
    # Monkeypatch the drawdown helper directly to a known >10% breach value,
    # rather than depending on holdings_log.jsonl file-plumbing details —
    # that plumbing is covered separately by
    # test_compute_drawdown_from_holdings_log_uses_real_history below.
    monkeypatch.setattr(ra, "_compute_drawdown_from_holdings_log", lambda *a, **k: 0.12)

    ra._run_daily_alert_checks(_FakeToday(), equity=88000.0, last_equity=100000.0)

    alerts_lines = alert_system.ALERTS_LOG.read_text().strip().split("\n")
    payloads = [json.loads(l) for l in alerts_lines]
    types = [p["type"] for p in payloads]
    assert "drawdown" in types
    drawdown_alert = next(p for p in payloads if p["type"] == "drawdown")
    assert drawdown_alert["severity"] == "CRITICAL"
    # system_alive ping must also have fired in the same call.
    assert "system_alive" in types


def test_run_daily_alert_checks_logs_failures_instead_of_swallowing(monkeypatch, caplog):
    """
    The old code wrapped the call in a bare `except: pass`, so an exception
    inside check_alerts() would vanish silently. The fix must at least log it.
    """
    def _boom(*a, **k):
        raise RuntimeError("simulated alert_system failure")

    monkeypatch.setattr(alert_system, "check_alerts", _boom)
    import logging
    caplog.set_level(logging.ERROR)

    # Must not raise.
    ra._run_daily_alert_checks(_FakeToday(), equity=100000.0, last_equity=99000.0)

    assert any("check_alerts" in rec.message for rec in caplog.records)


def test_run_daily_alert_checks_calls_alive_ping_exactly_once(monkeypatch):
    """
    send_system_alive_ping() previously had zero production callers. Confirm
    the wiring calls it exactly once per invocation of the daily alert check.
    """
    calls = []

    def _fake_ping(**kwargs):
        calls.append(kwargs)
        return {"type": "system_alive"}

    monkeypatch.setattr(alert_system, "send_system_alive_ping", _fake_ping)

    ra._run_daily_alert_checks(_FakeToday(), equity=100000.0, last_equity=99000.0)

    assert len(calls) == 1
    assert calls[0]["last_run"] == "2026-07-31"
    assert calls[0]["nav"] == 100000.0
    assert calls[0]["nav_prior"] == 99000.0


def test_compute_drawdown_from_holdings_log_uses_real_history(tmp_path):
    log_path = tmp_path / "holdings_log.jsonl"
    with open(log_path, "w") as f:
        f.write(json.dumps({"date": "2026-07-28", "equity": 100000.0}) + "\n")
        f.write(json.dumps({"date": "2026-07-29", "equity": 105000.0}) + "\n")
        f.write(json.dumps({"date": "2026-07-30", "equity": 102000.0}) + "\n")

    dd = ra._compute_drawdown_from_holdings_log(94500.0, log_path=log_path)
    # Peak across history + current = 105000; current = 94500 -> 10% drawdown.
    assert dd is not None
    assert abs(dd - 0.10) < 1e-6


def test_compute_drawdown_from_holdings_log_returns_none_without_history(tmp_path):
    missing_path = tmp_path / "does_not_exist.jsonl"
    assert ra._compute_drawdown_from_holdings_log(100000.0, log_path=missing_path) is None
