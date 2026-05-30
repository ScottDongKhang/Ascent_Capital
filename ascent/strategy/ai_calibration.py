"""
ascent/strategy/ai_calibration.py
Tracks AI PM market character predictions vs realized sleeve IC outcomes.

Lifecycle: log_thesis() at each rebalance → update_outcome() at next rebalance
         -> get_context() injected into next pre-thesis system prompt.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
OUTCOMES_LOG = _REPO_ROOT / "logs" / "ai_thesis_outcomes.jsonl"

_CHARACTER_TO_SLEEVES: Dict[str, List[str]] = {
    "momentum_continuation": ["trend"],
    "sector_rotation":       ["statarb", "meanrev"],
    "risk_off":              ["volatility", "fundamental"],
    "risk_on":               ["trend", "earnings"],
    "mean_reversion":        ["meanrev", "statarb"],
    "flight_to_quality":     ["fundamental", "volatility"],
    "uncertain":             [],
}


def log_thesis(
    thesis_date: str,
    regime: str,
    market_character: str,
    sleeve_weight_prior: Optional[Dict[str, float]] = None,
) -> None:
    """Log an AI PM market character prediction. Called each rebalance before quant."""
    entry = {
        "thesis_date": thesis_date,
        "regime": regime,
        "market_character": market_character,
        "sleeve_weight_prior": sleeve_weight_prior or {},
        "realized_ic_leaders": None,
        "prediction_correct": None,
    }
    OUTCOMES_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(OUTCOMES_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning("[AICalibration] log_thesis write failed: %s", e)
        return
    log.info("[AICalibration] Logged thesis date=%s regime=%s char=%s",
             thesis_date, regime, market_character)


def update_outcome(realized_ic_by_sleeve: Dict[str, float]) -> None:
    """
    Fill in realized outcome for the most recent pending log entry.
    Called at rebalance N+1 with IC measured over the N-to-N+1 holding period.
    """
    entries = _read_log()
    if not entries:
        return

    pending_idx = None
    for i in range(len(entries) - 1, -1, -1):
        if entries[i].get("prediction_correct") is None:
            pending_idx = i
            break

    if pending_idx is None:
        return

    entry = entries[pending_idx]
    top_sleeves = sorted(
        [(s, ic) for s, ic in realized_ic_by_sleeve.items() if ic > 0],
        key=lambda x: -x[1],
    )[:3]
    entry["realized_ic_leaders"] = [s for s, _ in top_sleeves]
    entry["prediction_correct"] = _check_correct(
        entry.get("market_character", ""),
        entry["realized_ic_leaders"],
    )

    _write_log(entries)
    log.info(
        "[AICalibration] Outcome filled for %s: correct=%s leaders=%s",
        entry["thesis_date"], entry["prediction_correct"], entry["realized_ic_leaders"],
    )


def get_context(regime: str, max_entries: int = 10) -> str:
    """
    Return a ~200 token calibration note for injection into the pre-thesis system prompt.
    Returns empty string if fewer than 3 completed entries exist for this regime.
    """
    entries = _read_log()
    regime_entries = [
        e for e in entries
        if e.get("regime") == regime and e.get("prediction_correct") is not None
    ]

    if len(regime_entries) < 3:
        return ""

    recent = regime_entries[-max_entries:]

    from collections import defaultdict
    by_char: Dict = defaultdict(lambda: {"total": 0, "correct": 0})
    for e in recent:
        c = e.get("market_character", "unknown")
        by_char[c]["total"] += 1
        if e.get("prediction_correct"):
            by_char[c]["correct"] += 1

    lines = [f"Calibration note ({regime}):"]
    for char, stats in sorted(by_char.items()):
        pct = int(100 * stats["correct"] / stats["total"])
        lines.append(f"- {char} calls: {stats['correct']}/{stats['total']} correct ({pct}%)")

    last_miss = next(
        (e for e in reversed(recent) if e.get("prediction_correct") is False), None
    )
    if last_miss:
        leaders = ", ".join(last_miss.get("realized_ic_leaders") or ["unknown"])
        lines.append(
            f"- Last miss ({last_miss['thesis_date']}): called {last_miss['market_character']} "
            f"but realized IC leaders were {leaders}"
        )

    return "\n".join(lines)


def _check_correct(market_character: str, realized_leaders: List[str]) -> bool:
    implied = _CHARACTER_TO_SLEEVES.get(market_character, [])
    if not implied:
        return False
    return any(s in realized_leaders for s in implied)


def _read_log() -> list:
    if not OUTCOMES_LOG.exists():
        return []
    entries = []
    for line in OUTCOMES_LOG.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return entries


def _write_log(entries: list) -> None:
    OUTCOMES_LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTCOMES_LOG.parent / (OUTCOMES_LOG.name + ".tmp")
    tmp.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    tmp.replace(OUTCOMES_LOG)
