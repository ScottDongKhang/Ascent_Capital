"""
tests/scripts/test_heartbeat_check.py

Tests for scripts/heartbeat_check.py — the stdlib-only pipeline liveness
watchdog. Uses tmp_path fixtures exclusively; never touches the real
logs/eod_log.jsonl or rebalance_calendar.csv.
"""
import csv
import importlib.util
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "heartbeat_check.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("heartbeat_check", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hb = _load_module()


def _write_eod_log(repo_root: Path, dates):
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "eod_log.jsonl"
    with log_path.open("w") as f:
        for d in dates:
            f.write(json.dumps({"date": d.isoformat(), "run_date": d.isoformat(),
                                 "rebalanced": True}) + "\n")
    return log_path


def _write_rebalance_calendar(repo_root: Path, dates):
    cal_path = repo_root / "rebalance_calendar.csv"
    with cal_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rebalance_date"])
        for d in dates:
            writer.writerow([d.isoformat()])
    return cal_path


# --------------------------------------------------------------------------
# Holiday calendar sanity (regression guard for the hardcoded formula)
# --------------------------------------------------------------------------

def test_2026_holidays_match_known_nyse_calendar():
    holidays = set(hb.us_market_holidays(2026))
    assert date(2026, 1, 1) in holidays        # New Year's
    assert date(2026, 1, 19) in holidays       # MLK
    assert date(2026, 4, 3) in holidays        # Good Friday
    assert date(2026, 6, 19) in holidays       # Juneteenth
    assert date(2026, 7, 3) in holidays        # Independence Day observed (Jul 4 is a Saturday)
    assert date(2026, 11, 26) in holidays      # Thanksgiving
    assert date(2026, 12, 25) in holidays      # Christmas


def test_weekend_is_not_a_trading_day():
    holidays = hb._holiday_set(2026, 2026)
    saturday = date(2026, 7, 25)
    assert saturday.weekday() == 5
    assert not hb.is_trading_day(saturday, holidays)


# --------------------------------------------------------------------------
# evaluate() — the core status logic, exercised directly (no I/O)
# --------------------------------------------------------------------------

def test_fresh_log_same_day_is_ok():
    last_run = date(2026, 7, 20)  # Monday
    as_of = date(2026, 7, 20)
    status, missed_days, missed_rebals = hb.evaluate(last_run, as_of, [])
    assert status == hb.STATUS_OK
    assert missed_days == []
    assert missed_rebals == []


def test_two_missed_trading_days_warns():
    last_run = date(2026, 7, 20)   # Monday
    as_of = date(2026, 7, 22)      # Wednesday -> Tue, Wed missed = 2
    status, missed_days, _ = hb.evaluate(last_run, as_of, [])
    assert len(missed_days) == 2
    assert status == hb.STATUS_WARN


def test_three_missed_trading_days_is_critical():
    last_run = date(2026, 7, 20)   # Monday
    as_of = date(2026, 7, 23)      # Thursday -> Tue, Wed, Thu = 3
    status, missed_days, _ = hb.evaluate(last_run, as_of, [])
    assert len(missed_days) == 3
    assert status == hb.STATUS_CRITICAL


def test_missed_rebalance_date_forces_critical_even_with_one_missed_day():
    last_run = date(2026, 7, 7)    # Tuesday
    as_of = date(2026, 7, 8)       # Wednesday -> only 1 missed trading day
    rebalance_dates = [date(2026, 7, 8)]
    status, missed_days, missed_rebals = hb.evaluate(last_run, as_of, rebalance_dates)
    assert len(missed_days) == 1
    assert missed_rebals == [date(2026, 7, 8)]
    assert status == hb.STATUS_CRITICAL


def test_never_run_is_critical():
    status, missed_days, missed_rebals = hb.evaluate(None, date(2026, 7, 27), [])
    assert status == hb.STATUS_CRITICAL


def test_real_world_state_is_critical_two_missed_rebalances():
    """Regression pin for the actual state discovered by the audit: last run
    2026-06-29, missed scheduled rebalances 2026-07-08 and 2026-07-22."""
    last_run = date(2026, 6, 29)
    as_of = date(2026, 7, 27)
    rebalance_dates = [date(2026, 7, 8), date(2026, 7, 22)]
    status, missed_days, missed_rebals = hb.evaluate(last_run, as_of, rebalance_dates)
    assert status == hb.STATUS_CRITICAL
    assert missed_rebals == [date(2026, 7, 8), date(2026, 7, 22)]
    assert len(missed_days) >= hb.CRITICAL_MISSED_DAYS


# --------------------------------------------------------------------------
# File readers — graceful degradation
# --------------------------------------------------------------------------

def test_read_last_run_date_missing_file_returns_none(tmp_path):
    assert hb.read_last_run_date(tmp_path / "does_not_exist.jsonl") is None


def test_read_last_run_date_empty_file_returns_none(tmp_path):
    p = tmp_path / "eod_log.jsonl"
    p.write_text("")
    assert hb.read_last_run_date(p) is None


def test_read_last_run_date_skips_malformed_lines(tmp_path):
    p = tmp_path / "eod_log.jsonl"
    p.write_text(
        "not json at all\n"
        '{"date": "2026-06-29", "rebalanced": true}\n'
        '{"trigger": "discovery", "symbol": "SCHH"}\n'
    )
    assert hb.read_last_run_date(p) == date(2026, 6, 29)


def test_read_rebalance_dates_missing_file_returns_empty(tmp_path):
    assert hb.read_rebalance_dates(tmp_path / "nope.csv") == []


def test_read_rebalance_dates_empty_file_returns_empty(tmp_path):
    p = tmp_path / "rebalance_calendar.csv"
    p.write_text("")
    assert hb.read_rebalance_dates(p) == []


# --------------------------------------------------------------------------
# End-to-end via build_report() using tmp_path as repo root
# --------------------------------------------------------------------------

def test_build_report_fresh_log_ok(tmp_path):
    _write_eod_log(tmp_path, [date(2026, 7, 20)])
    _write_rebalance_calendar(tmp_path, [])
    report = hb.build_report(tmp_path, date(2026, 7, 20))
    assert report["status"] == "OK"
    assert report["last_run"] == "2026-07-20"


def test_build_report_missing_files_is_critical_not_a_crash(tmp_path):
    # No logs/, no rebalance_calendar.csv at all.
    report = hb.build_report(tmp_path, date(2026, 7, 27))
    assert report["status"] == "CRITICAL"
    assert report["last_run"] is None
    assert report["missed_rebalances"] == 0


# --------------------------------------------------------------------------
# CLI subprocess tests — exit codes, --json, --quiet, liveness.json write
# --------------------------------------------------------------------------

def _run_cli(repo_root: Path, *extra_args):
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--repo-root", str(repo_root),
         "--no-alert", *extra_args],
        capture_output=True, text=True,
    )


def test_cli_exit_code_ok(tmp_path):
    _write_eod_log(tmp_path, [date(2026, 7, 20)])
    result = _run_cli(tmp_path, "--as-of", "2026-07-20", "--json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "OK"


def test_cli_exit_code_warn(tmp_path):
    _write_eod_log(tmp_path, [date(2026, 7, 20)])
    result = _run_cli(tmp_path, "--as-of", "2026-07-22", "--json")
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == "WARN"


def test_cli_exit_code_critical(tmp_path):
    _write_eod_log(tmp_path, [date(2026, 7, 20)])
    result = _run_cli(tmp_path, "--as-of", "2026-07-23", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "CRITICAL"


def test_cli_writes_liveness_json(tmp_path):
    _write_eod_log(tmp_path, [date(2026, 7, 20)])
    _run_cli(tmp_path, "--as-of", "2026-07-20")
    liveness_path = tmp_path / "logs" / "liveness.json"
    assert liveness_path.exists()
    payload = json.loads(liveness_path.read_text())
    assert payload["status"] == "OK"


def test_cli_quiet_suppresses_stdout(tmp_path):
    _write_eod_log(tmp_path, [date(2026, 7, 20)])
    result = _run_cli(tmp_path, "--as-of", "2026-07-20", "--quiet")
    assert result.stdout.strip() == ""


def test_cli_graceful_with_no_files_at_all(tmp_path):
    # Deliberately do not create logs/ or rebalance_calendar.csv.
    result = _run_cli(tmp_path, "--as-of", "2026-07-27", "--json")
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "CRITICAL"
    assert payload["last_run"] is None
