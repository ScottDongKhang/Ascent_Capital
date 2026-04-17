"""
ascent/execution/run_log.py
Structured JSONL logger for EOD execution runs.
Each run appends one JSON line to logs/eod_log.jsonl
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List


LOG_DIR  = Path("logs")
LOG_FILE = LOG_DIR / "eod_log.jsonl"


def _ensure_log_dir():
    LOG_DIR.mkdir(exist_ok=True)


def log_run(
    run_date: str,
    run_type: str,                    # 'rebalance' | 'log_only' | 'error'
    regime_label: Optional[str],
    regime_confidence: Optional[float],
    posture: Optional[str],
    portfolio_value: Optional[float],
    target_weights: Optional[dict],
    orders_executed: Optional[List[dict]],
    orders_skipped: Optional[List[dict]],
    notes: str = "",
    error: Optional[str] = None,
):
    _ensure_log_dir()

    entry = {
        "timestamp":          datetime.utcnow().isoformat() + "Z",
        "run_date":           run_date,
        "run_type":           run_type,
        "regime_label":       regime_label,
        "regime_confidence":  round(regime_confidence, 4) if regime_confidence is not None else None,
        "posture":            posture,
        "portfolio_value":    round(portfolio_value, 2) if portfolio_value is not None else None,
        "target_weights":     target_weights,
        "orders_executed":    orders_executed or [],
        "orders_skipped":     orders_skipped or [],
        "notes":              notes,
        "error":              error,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"[RunLog] Logged {run_type} run for {run_date} → {LOG_FILE}")


def log_error(run_date: str, error: str):
    log_run(
        run_date=run_date,
        run_type="error",
        regime_label=None,
        regime_confidence=None,
        posture=None,
        portfolio_value=None,
        target_weights=None,
        orders_executed=None,
        orders_skipped=None,
        error=error,
    )


def read_last_run() -> Optional[dict]:
    """Return the most recent log entry, or None if log is empty."""
    if not LOG_FILE.exists():
        return None
    lines = LOG_FILE.read_text().strip().splitlines()
    if not lines:
        return None
    return json.loads(lines[-1])


def read_log(n: int = 10) -> List[dict]:
    """Return the last n log entries."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text().strip().splitlines()
    return [json.loads(l) for l in lines[-n:]]
