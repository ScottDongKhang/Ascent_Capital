"""
ascent/execution/kill_switch.py

Portfolio-level drawdown kill switch for Ascent Capital.

Logic:
  - Reads logs/eod_log.jsonl to reconstruct NAV history from live fills.
  - Computes peak NAV and current drawdown from peak.
  - If drawdown exceeds the configured threshold, raises KillSwitchTriggered.
  - eod_runner.py catches this before submitting any orders and halts.
  - A separate --reset-kill-switch flag manually re-enables trading after review.

Thresholds (configurable at top of file):
  SOFT_WARN_PCT           = 0.05   -> logs a warning, trades still go through
  HARD_STOP_PCT           = 0.12   -> halts all order submission (peak-to-trough), logs reason
  MONTHLY_SOFT_HALT_PCT   = 0.06   -> halts all order submission if the CURRENT
                                       calendar month is down this much from its
                                       month-start NAV, independent of the
                                       whole-book peak-to-trough check above.

Kill switch state is persisted in logs/kill_switch_state.json so it
survives process restarts. Once tripped, it stays tripped until you
manually reset it.
"""

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

from ascent.utils.market_time import market_today

# ── Configurable thresholds ──────────────────────────────────────────────────
SOFT_WARN_PCT = 0.05   # 5%  peak-to-trough drawdown → warning only
HARD_STOP_PCT = 0.12   # 12% peak-to-trough drawdown → full halt

# Monthly circuit breaker: independent of the peak-to-trough check above.
# Halts trading if the CURRENT calendar month's NAV is down this much from
# its month-start NAV (first NAV on/after the 1st of the current trading
# month, per _read_nav_history()).
MONTHLY_SOFT_HALT_PCT = 0.06   # 6% month-to-date drawdown → full halt

# ── Paths ────────────────────────────────────────────────────────────────────
LOG_DIR        = Path("logs")
EOD_LOG_PATH   = LOG_DIR / "eod_log.jsonl"
KS_STATE_PATH  = LOG_DIR / "kill_switch_state.json"


# ── Exceptions ────────────────────────────────────────────────────────────────
class KillSwitchTriggered(Exception):
    """Raised when drawdown exceeds HARD_STOP_PCT. Caught in eod_runner.py."""
    pass


# ── State helpers ─────────────────────────────────────────────────────────────
def _load_state() -> dict:
    """Load persisted kill switch state, or return a clean default."""
    if KS_STATE_PATH.exists():
        try:
            with open(KS_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "tripped": False,
        "tripped_at": None,
        "tripped_reason": None,
        "peak_nav": None,
        "drawdown_at_trip": None,
        "trip_reason_kind": None,
        "reset_at": None,
        # Monthly circuit breaker baseline (informational persistence; the
        # value is always recomputed from eod_log.jsonl, this just records
        # what the last check used so `status()` / debugging can see it).
        "month_start_key": None,
        "month_start_nav": None,
    }


def _save_state(state: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with open(KS_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ── NAV reconstruction from eod_log.jsonl ────────────────────────────────────
def _read_nav_history() -> list[dict]:
    """
    Parse eod_log.jsonl and return a list of {date, nav} dicts,
    sorted oldest → newest. Only includes rows that have a valid nav field.

    Log schema written by run_log.py:
      {
        "timestamp":       "2026-03-24T20:01:00Z",   # UTC log time
        "run_date":        "2026-03-24",              # intended trading date
        "portfolio_value": 98340.12,
        ...
      }
    """
    if not EOD_LOG_PATH.exists():
        return []

    entries = []
    with open(EOD_LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                nav = record.get("portfolio_value") or record.get("nav")

                # FIX #25: run_log.py writes "run_date", not "date".
                #
                # BEFORE (broken):
                #   dt = record.get("date") or record.get("timestamp", "")[:10]
                #
                # record.get("date") always returned None because the logger
                # never writes a "date" key — it writes "run_date". The fallback
                # then sliced the UTC "timestamp" string, which can be a different
                # calendar day from the intended trading date (e.g. a run at
                # 23:58 UTC on 2026-03-24 logs timestamp 2026-03-24T23:58Z but
                # run_date 2026-03-24, while a run at 00:02 UTC logs timestamp
                # 2026-03-25T00:02Z with run_date still 2026-03-24). This caused
                # NAV history entries to be stamped with the wrong date and
                # potentially mis-ordered.
                #
                # FIX: read run_date first (the intended trading date), then fall
                # back to slicing timestamp only if run_date is absent (forward
                # compatibility with any older log entries).
                dt = (
                    record.get("run_date")
                    or record.get("timestamp", "")[:10]
                )

                if nav and dt:
                    entries.append({"date": dt, "nav": float(nav)})
            except (json.JSONDecodeError, ValueError):
                continue

    entries.sort(key=lambda x: x["date"])
    return entries


def _month_key(date_str: str) -> str:
    """'2026-08-23' -> '2026-08'."""
    return date_str[:7]


def _get_month_start_nav(
    history: list[dict], now: Optional[datetime] = None
) -> tuple[Optional[str], Optional[float]]:
    """
    Determine the current trading month (from the actual current date, via
    ascent.utils.market_time.market_today() -- NOT from the tail of `history`)
    and the month-start NAV baseline: the first available NAV in `history` on
    or after the 1st of that month.

    `history` is derived from eod_log.jsonl, which check() calls BEFORE
    today's own entry is appended -- so history[-1] is always a prior day's
    NAV and must never be used to decide what "current month" means (that
    was the bug: on the first trading day of a new month, history[-1] is
    still last month's final entry, misattributing the monthly baseline).
    `current_nav` (today's actual latest point) is handled separately by the
    caller; this function only locates the month-start baseline within the
    historical log.

    Returns (month_key, month_start_nav). month_start_nav is None when there
    is no NAV history at all yet for the current trading month (e.g. the
    very first trading day of a new month, before any EOD run has logged a
    NAV) -- callers must treat that as "not enough data, skip the check",
    not as a trigger.
    """
    current_month = _month_key(market_today(now).isoformat())

    if not history:
        return current_month, None

    month_entries = [e for e in history if _month_key(e["date"]) == current_month]
    if not month_entries:
        return current_month, None

    month_entries.sort(key=lambda e: e["date"])
    return current_month, month_entries[0]["nav"]


# ── Core check ───────────────────────────────────────────────────────────────
def check(current_nav: Optional[float] = None, now: Optional[datetime] = None) -> dict:
    """
    Run the kill switch check.

    Parameters
    ----------
    current_nav : float, optional
        Pass the live NAV if you already have it (e.g. from the Alpaca
        account fetch in eod_runner). If None, the function uses the
        most recent entry in eod_log.jsonl.
    now : datetime, optional
        Injectable "current instant" for the monthly circuit breaker's
        month determination (see _get_month_start_nav). Tests use this;
        production leaves it None so ascent.utils.market_time.market_today()
        reads the real current market date.

    Returns
    -------
    dict with keys:
        status            : "ok" | "warn" | "halted"
        drawdown          : float, peak-to-trough (0.0 → 1.0, e.g. 0.12 = 12%)
        peak_nav          : float
        current_nav       : float
        month_start_nav   : float or None (month-start NAV baseline, None if
                             no NAV logged yet this trading month)
        monthly_drawdown  : float or None, month-to-date (same units as above)
        message           : str (human-readable summary)

    Raises
    ------
    KillSwitchTriggered
        If the switch is already tripped, or on this call either:
          - peak-to-trough drawdown >= HARD_STOP_PCT, or
          - month-to-date drawdown from the month-start NAV baseline >=
            MONTHLY_SOFT_HALT_PCT (independent check; does not require the
            peak-to-trough condition to also be true).
    """
    LOG_DIR.mkdir(exist_ok=True)
    state = _load_state()

    # ── Already tripped from a previous run ──────────────────────────────────
    if state["tripped"]:
        msg = (
            f"[KILL SWITCH] Already tripped on {state['tripped_at']}. "
            f"Reason: {state['tripped_reason']}. "
            f"Run with --reset-kill-switch to re-enable trading after review."
        )
        print(msg, file=sys.stderr)
        raise KillSwitchTriggered(msg)

    # ── Get current NAV ───────────────────────────────────────────────────────
    history = _read_nav_history()

    if current_nav is None:
        if not history:
            print("[KILL SWITCH] No NAV history found — skipping check (first run).")
            return {"status": "ok", "drawdown": 0.0, "peak_nav": None,
                    "current_nav": None, "message": "No history yet."}
        current_nav = history[-1]["nav"]

    # ── Compute peak NAV ──────────────────────────────────────────────────────
    if history:
        peak_nav = max(e["nav"] for e in history)
        peak_nav = max(peak_nav, current_nav)
    else:
        peak_nav = current_nav

    drawdown = (peak_nav - current_nav) / peak_nav if peak_nav > 0 else 0.0

    result = {
        "status": "ok",
        "drawdown": round(drawdown, 4),
        "peak_nav": round(peak_nav, 2),
        "current_nav": round(current_nav, 2),
        "message": "",
    }

    # ── Soft warning ─────────────────────────────────────────────────────────
    if drawdown >= SOFT_WARN_PCT:
        msg = (
            f"[KILL SWITCH] WARNING — drawdown {drawdown:.1%} exceeds soft "
            f"threshold {SOFT_WARN_PCT:.0%}. "
            f"NAV: ${current_nav:,.0f} | Peak: ${peak_nav:,.0f}. "
            f"Trades will still proceed."
        )
        print(msg)
        result["status"] = "warn"
        result["message"] = msg

    # ── Hard stop ─────────────────────────────────────────────────────────────
    if drawdown >= HARD_STOP_PCT:
        reason = (
            f"Drawdown {drawdown:.1%} exceeded hard stop {HARD_STOP_PCT:.0%}. "
            f"NAV ${current_nav:,.0f} vs peak ${peak_nav:,.0f}."
        )
        state["tripped"]          = True
        state["tripped_at"]       = datetime.utcnow().isoformat()
        state["tripped_reason"]   = reason
        state["peak_nav"]         = peak_nav
        state["drawdown_at_trip"] = round(drawdown, 4)
        state["trip_reason_kind"] = "peak"
        _save_state(state)

        msg = f"[KILL SWITCH] HALTED — {reason} Run --reset-kill-switch to resume."
        print(msg, file=sys.stderr)
        result["status"]  = "halted"
        result["message"] = msg
        raise KillSwitchTriggered(msg)

    # ── Monthly circuit breaker (independent of peak-to-trough above) ───────
    # Tracks month-to-date drawdown from the month-start NAV, not the
    # whole-book peak. A book can be well within its all-time peak-to-trough
    # budget while still having a bad month -- this catches that case
    # separately so it doesn't get masked by an all-time-high peak.
    month_key, month_start_nav = _get_month_start_nav(history, now=now)
    result["month_start_nav"] = (
        round(month_start_nav, 2) if month_start_nav is not None else None
    )
    result["monthly_drawdown"] = None

    if month_start_nav is not None and month_start_nav > 0:
        state["month_start_key"] = month_key
        state["month_start_nav"] = round(month_start_nav, 2)

        monthly_drawdown = (month_start_nav - current_nav) / month_start_nav
        result["monthly_drawdown"] = round(monthly_drawdown, 4)

        if monthly_drawdown >= MONTHLY_SOFT_HALT_PCT:
            reason = (
                f"Month-to-date drawdown {monthly_drawdown:.1%} exceeded monthly "
                f"circuit breaker {MONTHLY_SOFT_HALT_PCT:.0%} for {month_key}. "
                f"NAV ${current_nav:,.0f} vs month-start NAV ${month_start_nav:,.0f}."
            )
            state["tripped"]          = True
            state["tripped_at"]       = datetime.utcnow().isoformat()
            state["tripped_reason"]   = reason
            state["peak_nav"]         = peak_nav
            # BUG FIX: this trip is caused by the monthly check, so record
            # the monthly drawdown that actually triggered it -- NOT the
            # whole-book peak-to-trough `drawdown` computed earlier, which
            # can be small (e.g. a book near its all-time peak having a bad
            # month) and would otherwise misreport the real MTD loss.
            state["drawdown_at_trip"] = round(monthly_drawdown, 4)
            state["trip_reason_kind"] = "monthly"
            _save_state(state)

            msg = (
                f"[KILL SWITCH] HALTED (monthly circuit breaker) — {reason} "
                f"Run --reset-kill-switch to resume."
            )
            print(msg, file=sys.stderr)
            result["status"]  = "halted"
            result["message"] = msg
            raise KillSwitchTriggered(msg)
    else:
        # No NAV recorded yet for the current trading month (e.g. the very
        # first trading day of a new month, before any EOD run has logged a
        # NAV) -- not enough data, do not trigger.
        print(
            "[KILL SWITCH] No NAV history for the current month yet — "
            "skipping monthly circuit breaker check."
        )

    _save_state(state)

    if not result["message"]:
        result["message"] = (
            f"OK — drawdown {drawdown:.1%} | "
            f"NAV ${current_nav:,.0f} | Peak ${peak_nav:,.0f}"
        )
        print(f"[KILL SWITCH] {result['message']}")

    return result


# ── Manual reset ─────────────────────────────────────────────────────────────
def reset(reason: str = "Manual reset by operator") -> None:
    """
    Re-enable trading after a kill switch trip.
    Call this from eod_runner.py when --reset-kill-switch flag is passed.
    """
    state = _load_state()
    if not state["tripped"]:
        print("[KILL SWITCH] Not currently tripped — nothing to reset.")
        return

    prev_reason         = state["tripped_reason"]
    state["tripped"]    = False
    state["reset_at"]   = datetime.utcnow().isoformat()
    state["tripped_reason"] = None
    _save_state(state)

    print(
        f"[KILL SWITCH] Reset by operator. "
        f"Previous trip reason: {prev_reason}. "
        f"Trading re-enabled."
    )


def status() -> None:
    """Print current kill switch state to stdout."""
    state   = _load_state()
    history = _read_nav_history()

    print("=" * 50)
    print("KILL SWITCH STATUS")
    print("=" * 50)
    print(f"  Tripped          : {state['tripped']}")
    if state["tripped"]:
        print(f"  Tripped at       : {state['tripped_at']}")
        print(f"  Reason           : {state['tripped_reason']}")
        if state["drawdown_at_trip"]:
            kind = state.get("trip_reason_kind") or "peak"
            label = "monthly" if kind == "monthly" else "peak-to-trough"
            print(f"  DD at trip ({label}) : {state['drawdown_at_trip']:.1%}")
    if state["reset_at"]:
        print(f"  Last reset       : {state['reset_at']}")
    print(f"  Soft warn        : {SOFT_WARN_PCT:.0%}")
    print(f"  Hard stop        : {HARD_STOP_PCT:.0%}")
    print(f"  Monthly halt     : {MONTHLY_SOFT_HALT_PCT:.0%}")
    if history:
        navs    = [e["nav"] for e in history]
        peak    = max(navs)
        current = navs[-1]
        dd      = (peak - current) / peak
        print(f"  Peak NAV         : ${peak:,.2f}")
        print(f"  Current NAV      : ${current:,.2f}")
        print(f"  Current drawdown : {dd:.1%}")

        month_key, month_start_nav = _get_month_start_nav(history)
        if month_start_nav is not None and month_start_nav > 0:
            mdd = (month_start_nav - current) / month_start_nav
            print(f"  Month ({month_key}) start NAV : ${month_start_nav:,.2f}")
            print(f"  Month-to-date drawdown        : {mdd:.1%}")
    print("=" * 50)


# ── CLI for direct inspection ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ascent Capital Kill Switch")
    parser.add_argument("--status",        action="store_true", help="Show current state")
    parser.add_argument("--reset",         action="store_true", help="Re-enable trading")
    parser.add_argument("--check",         action="store_true", help="Run a live check")
    parser.add_argument("--set-soft-warn", type=float,          help="Override soft warn % (e.g. 0.10)")
    parser.add_argument("--set-hard-stop", type=float,          help="Override hard stop % (e.g. 0.20)")
    args = parser.parse_args()

    if args.set_soft_warn:
        SOFT_WARN_PCT = args.set_soft_warn
    if args.set_hard_stop:
        HARD_STOP_PCT = args.set_hard_stop

    if args.status:
        status()
    elif args.reset:
        reset()
    elif args.check:
        try:
            result = check()
            print(json.dumps(result, indent=2))
        except KillSwitchTriggered as e:
            print(f"HALTED: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()