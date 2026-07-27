#!/usr/bin/env python3
"""
scripts/heartbeat_check.py — liveness monitor for the daily pipeline.

STDLIB ONLY. This script must not `import ascent`, pandas, numpy, or any
third-party package. It exists precisely because the pipeline it watches
silently stopped running for 27 days (last scheduled success 2026-04-03,
launchd job never installed, `~/Downloads` path referenced a nonexistent
user). A watchdog built from the same dependency stack as the thing it
watches can be broken by the same failure. This one can't: it only needs
a Python 3 interpreter and the two log/calendar files below.

Approach to "NYSE trading day":
    We do not depend on `pandas_market_calendars` (not stdlib, and it was
    the exact import that silently no-ops in run_eod.sh's original
    holiday check — see `IS_HOLIDAY` in the pre-fix script). Instead we
    hardcode the fixed-formula US market holidays (New Year's, MLK,
    Presidents, Good Friday via the standard Gregorian Easter algorithm,
    Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving,
    Christmas — each with the standard weekend observed-date shift) for a
    generated range of years around "today". This slightly overcounts
    trading days in years where NYSE deviates from the fixed formula
    (rare, e.g. a one-off closure) but never undercounts weekends, which
    is what matters for a liveness check: false CRITICAL is safe (it
    fires an alert that a human can dismiss), false OK is not.

Usage:
    scripts/heartbeat_check.py                 # human-readable, writes liveness.json
    scripts/heartbeat_check.py --json          # machine-readable stdout
    scripts/heartbeat_check.py --quiet         # suppress stdout, still writes file + exit code
    scripts/heartbeat_check.py --alive-ping    # send the positive "system alive" notification
    scripts/heartbeat_check.py --as-of 2026-07-27   # override "today" (testing)
    scripts/heartbeat_check.py --repo-root /path    # override repo root (testing)

Exit codes: 0 = OK, 1 = WARN, 2 = CRITICAL.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT_DEFAULT = Path(__file__).resolve().parent.parent
EOD_LOG_REL = Path("logs/eod_log.jsonl")
REBALANCE_CAL_REL = Path("rebalance_calendar.csv")
LIVENESS_OUT_REL = Path("logs/liveness.json")

WARN_MISSED_DAYS = 2
CRITICAL_MISSED_DAYS = 3

STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_CRITICAL = "CRITICAL"

_EXIT_CODES = {STATUS_OK: 0, STATUS_WARN: 1, STATUS_CRITICAL: 2}


# --------------------------------------------------------------------------
# Holiday calendar (stdlib-only, no pandas_market_calendars dependency)
# --------------------------------------------------------------------------

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """The n-th occurrence of `weekday` (Mon=0) in `month`/`year`."""
    d = dt.date(year, month, 1)
    count = 0
    while True:
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
        d += dt.timedelta(days=1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    if month == 12:
        d = dt.date(year, 12, 31)
    else:
        d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    while d.weekday() != weekday:
        d -= dt.timedelta(days=1)
    return d


def _observed(d: dt.date) -> dt.date:
    """Weekend-observed shift: Saturday -> preceding Friday, Sunday -> following Monday."""
    if d.weekday() == 5:
        return d - dt.timedelta(days=1)
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def _easter_sunday(year: int) -> dt.date:
    """Anonymous Gregorian algorithm (standard, deterministic, no deps)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def us_market_holidays(year: int) -> List[dt.date]:
    """Fixed-formula US market holidays for one calendar year."""
    good_friday = _easter_sunday(year) - dt.timedelta(days=2)
    return sorted([
        _observed(dt.date(year, 1, 1)),               # New Year's Day
        _nth_weekday(year, 1, 0, 3),                   # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                   # Presidents Day (3rd Mon Feb)
        good_friday,                                    # Good Friday
        _last_weekday(year, 5, 0),                      # Memorial Day (last Mon May)
        _observed(dt.date(year, 6, 19)),                # Juneteenth
        _observed(dt.date(year, 7, 4)),                 # Independence Day
        _nth_weekday(year, 9, 0, 1),                    # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                   # Thanksgiving (4th Thu Nov)
        _observed(dt.date(year, 12, 25)),               # Christmas
    ])


def _holiday_set(start_year: int, end_year: int) -> set:
    holidays = set()
    for y in range(start_year, end_year + 1):
        holidays.update(us_market_holidays(y))
    return holidays


def is_trading_day(d: dt.date, holidays: set) -> bool:
    return d.weekday() < 5 and d not in holidays


def trading_days_between(start_exclusive: dt.date, end_exclusive: dt.date) -> List[dt.date]:
    """Trading days strictly after `start_exclusive` and strictly before `end_exclusive`."""
    if end_exclusive <= start_exclusive:
        return []
    holidays = _holiday_set(start_exclusive.year, end_exclusive.year)
    out = []
    d = start_exclusive + dt.timedelta(days=1)
    while d < end_exclusive:
        if is_trading_day(d, holidays):
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# --------------------------------------------------------------------------
# Log / calendar readers — must degrade gracefully on missing/empty files
# --------------------------------------------------------------------------

def _parse_date(s: str) -> Optional[dt.date]:
    try:
        return dt.datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def read_last_run_date(eod_log_path: Path) -> Optional[dt.date]:
    """Max `date` (falling back to `run_date`) across all JSONL entries.

    Malformed lines are skipped. Missing/empty file -> None (caller treats
    this as "never run" and reports CRITICAL rather than crashing).
    """
    if not eod_log_path.exists():
        return None
    best: Optional[dt.date] = None
    try:
        with eod_log_path.open("r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw = entry.get("date") or entry.get("run_date")
                if not raw:
                    continue
                d = _parse_date(str(raw))
                if d is not None and (best is None or d > best):
                    best = d
    except OSError:
        return None
    return best


def read_rebalance_dates(cal_path: Path) -> List[dt.date]:
    """All dates from rebalance_calendar.csv. Missing/empty file -> []."""
    if not cal_path.exists():
        return []
    out = []
    try:
        with cal_path.open("r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except OSError:
        return []
    if not rows:
        return []
    header = rows[0]
    start_idx = 0
    # tolerate a header row like "rebalance_date"
    if header and not _parse_date(header[0]):
        start_idx = 1
    for row in rows[start_idx:]:
        if not row:
            continue
        d = _parse_date(row[0])
        if d is not None:
            out.append(d)
    return out


def missed_rebalance_dates(rebalance_dates: Iterable[dt.date], last_run: Optional[dt.date],
                            as_of: dt.date) -> List[dt.date]:
    """Rebalance-calendar dates strictly after last_run and on/before as_of that were missed.

    If last_run is None, every rebalance date up to as_of counts as missed.
    """
    missed = []
    for d in rebalance_dates:
        if d > as_of:
            continue
        if last_run is not None and d <= last_run:
            continue
        missed.append(d)
    return sorted(missed)


# --------------------------------------------------------------------------
# Status evaluation
# --------------------------------------------------------------------------

def evaluate(last_run: Optional[dt.date], as_of: dt.date,
             rebalance_dates: List[dt.date]) -> Tuple[str, List[dt.date], List[dt.date]]:
    """Returns (status, missed_trading_days, missed_rebalance_dates)."""
    if last_run is None:
        # Never logged a run at all — cannot compute a bounded missed-day
        # count, but this is unambiguously the worst case.
        missed_days: List[dt.date] = []
        missed_rebals = missed_rebalance_dates(rebalance_dates, None, as_of)
        return STATUS_CRITICAL, missed_days, missed_rebals

    missed_days = trading_days_between(last_run, as_of + dt.timedelta(days=1))
    missed_rebals = missed_rebalance_dates(rebalance_dates, last_run, as_of)

    if len(missed_days) >= CRITICAL_MISSED_DAYS or missed_rebals:
        status = STATUS_CRITICAL
    elif len(missed_days) >= WARN_MISSED_DAYS:
        status = STATUS_WARN
    else:
        status = STATUS_OK
    return status, missed_days, missed_rebals


# --------------------------------------------------------------------------
# Alerting — this script can fire an alert directly, without importing
# ascent.monitoring.alert_system, so a broken pipeline environment cannot
# take the watchdog down with it. If ascent IS importable (normal case),
# we also hand off to alert_system.send_alert() for a single log/dedup path.
# --------------------------------------------------------------------------

def _send_ntfy(message: str, title: str, priority: str) -> bool:
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return False
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception:
        return False


def _send_mac_notification(message: str, title: str) -> bool:
    if sys.platform != "darwin":
        return False
    safe_message = message.replace('"', '\\"')
    safe_title = title.replace('"', '\\"')
    script = f'display notification "{safe_message}" with title "{safe_title}"'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False, capture_output=True, timeout=10,
        )
        return True
    except Exception:
        return False


def send_direct_alert(message: str, severity: str) -> None:
    """Best-effort direct alert: ntfy.sh if configured, always also a local
    macOS notification as a no-config fallback. Never raises."""
    title = f"Ascent {severity}"
    priority = "high" if severity == "CRITICAL" else "default"
    _send_ntfy(message, title, priority)
    _send_mac_notification(message, title)


def _try_alert_system_hook(alert_dict: dict) -> bool:
    """Optional integration with ascent.monitoring.alert_system for a single
    logged/deduped alert trail. Best-effort only — heartbeat_check must keep
    working even if `ascent` cannot be imported (that's the whole point)."""
    try:
        repo_root = REPO_ROOT_DEFAULT
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from ascent.monitoring.alert_system import send_alert  # type: ignore
        send_alert(alert_dict)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_report(repo_root: Path, as_of: dt.date) -> dict:
    eod_log_path = repo_root / EOD_LOG_REL
    cal_path = repo_root / REBALANCE_CAL_REL

    last_run = read_last_run_date(eod_log_path)
    rebalance_dates = read_rebalance_dates(cal_path)

    status, missed_days, missed_rebals = evaluate(last_run, as_of, rebalance_dates)

    return {
        "status": status,
        "as_of": as_of.isoformat(),
        "last_run": last_run.isoformat() if last_run else None,
        "missed_days": len(missed_days),
        "missed_day_dates": [d.isoformat() for d in missed_days],
        "missed_rebalances": len(missed_rebals),
        "missed_rebalance_dates": [d.isoformat() for d in missed_rebals],
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }


def write_liveness_file(repo_root: Path, report: dict) -> Path:
    out_path = repo_root / LIVENESS_OUT_REL
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return out_path


def _human_summary(report: dict) -> str:
    status = report["status"]
    last_run = report["last_run"] or "never"
    lines = [
        f"[heartbeat] status={status} last_run={last_run} as_of={report['as_of']}",
        f"[heartbeat] missed_trading_days={report['missed_days']} "
        f"missed_rebalances={report['missed_rebalances']}",
    ]
    if report["missed_day_dates"]:
        lines.append(f"[heartbeat] missed days: {', '.join(report['missed_day_dates'])}")
    if report["missed_rebalance_dates"]:
        lines.append(f"[heartbeat] missed rebalances: {', '.join(report['missed_rebalance_dates'])}")
    return "\n".join(lines)


def _alive_ping_message(report: dict) -> str:
    last_run = report["last_run"] or "never"
    return f"Ascent Capital: system alive. Last run {last_run}, heartbeat status {report['status']}."


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline liveness heartbeat check (stdlib only).")
    parser.add_argument("--repo-root", type=str, default=str(REPO_ROOT_DEFAULT),
                         help="Repo root (default: parent of scripts/)")
    parser.add_argument("--as-of", type=str, default=None,
                         help="Override 'today' as YYYY-MM-DD (testing only)")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON to stdout")
    parser.add_argument("--quiet", action="store_true", help="Suppress stdout entirely")
    parser.add_argument("--no-alert", action="store_true",
                         help="Never send a direct alert, even on breach (used by tests)")
    parser.add_argument("--alive-ping", action="store_true",
                         help="Send the positive 'system alive' notification regardless of status")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    as_of = _parse_date(args.as_of) if args.as_of else dt.date.today()
    if as_of is None:
        parser.error(f"--as-of must be YYYY-MM-DD, got {args.as_of!r}")

    report = build_report(repo_root, as_of)
    write_liveness_file(repo_root, report)

    if not args.quiet:
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            print(_human_summary(report))

    if not args.no_alert and report["status"] != STATUS_OK:
        message = (
            f"Liveness {report['status']}: last run {report['last_run'] or 'never'}, "
            f"{report['missed_days']} missed trading day(s), "
            f"{report['missed_rebalances']} missed rebalance(s)."
        )
        alert_dict = {
            "type": "liveness",
            "severity": report["status"],
            "message": message,
            "value": report["missed_days"],
            "timestamp": dt.datetime.utcnow().isoformat(),
        }
        if not _try_alert_system_hook(alert_dict):
            send_direct_alert(message, report["status"])
        else:
            # Also fire the direct channel — alert_system.send_alert() only
            # posts to ntfy today; the mac-notification fallback still adds
            # value even when the ascent import succeeds.
            send_direct_alert(message, report["status"])

    if args.alive_ping:
        send_direct_alert(_alive_ping_message(report), "INFO")

    return _EXIT_CODES[report["status"]]


if __name__ == "__main__":
    sys.exit(main())
