"""
tests/monitoring/test_alert_system_liveness.py

Covers the W3 additions to ascent/monitoring/alert_system.py:
- "liveness" and "system_alive" are recognized alert types
- send_system_alive_ping() writes a positive daily entry
- existing send_alert()/check_alerts() callers are unaffected (backward compat)
"""
import json

import pytest

from ascent.monitoring import alert_system


@pytest.fixture(autouse=True)
def _isolate_alert_files(tmp_path, monkeypatch):
    alerts_log = tmp_path / "alerts.jsonl"
    state_file = tmp_path / "alert_state.json"
    monkeypatch.setattr(alert_system, "ALERTS_LOG", alerts_log)
    monkeypatch.setattr(alert_system, "ALERT_STATE_FILE", state_file)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    return alerts_log


def test_liveness_and_system_alive_are_valid_alert_types():
    assert "liveness" in alert_system.VALID_ALERT_TYPES
    assert "system_alive" in alert_system.VALID_ALERT_TYPES


def test_send_alert_still_writes_drawdown_alert(_isolate_alert_files):
    alert_system.send_alert({
        "type": "drawdown", "severity": "WARNING", "message": "test", "value": 0.06,
        "timestamp": "2026-07-27T00:00:00",
    })
    lines = _isolate_alert_files.read_text().strip().split("\n")
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["type"] == "drawdown"


def test_send_liveness_alert_writes_to_log(_isolate_alert_files):
    alert_system.send_alert({
        "type": "liveness", "severity": "CRITICAL",
        "message": "Liveness CRITICAL: last run 2026-06-29, 20 missed trading day(s).",
        "value": 20, "timestamp": "2026-07-27T00:00:00",
    })
    lines = _isolate_alert_files.read_text().strip().split("\n")
    payload = json.loads(lines[0])
    assert payload["type"] == "liveness"
    assert payload["severity"] == "CRITICAL"


def test_send_system_alive_ping_writes_info_entry(_isolate_alert_files):
    alert = alert_system.send_system_alive_ping(last_run="2026-07-27", nav=100000.0, nav_prior=99500.0)
    assert alert["type"] == "system_alive"
    assert alert["severity"] == "INFO"
    assert "2026-07-27" in alert["message"]

    lines = _isolate_alert_files.read_text().strip().split("\n")
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["type"] == "system_alive"


def test_send_system_alive_ping_handles_missing_nav(_isolate_alert_files):
    alert = alert_system.send_system_alive_ping(last_run="2026-07-27")
    assert "NAV unchanged" in alert["message"]


def test_check_alerts_unaffected_when_no_inputs_given():
    # Backward-compat smoke test: existing callers passing nothing should
    # still get an empty list, not an exception, after the new module-level
    # constants were added.
    assert alert_system.check_alerts() == []
