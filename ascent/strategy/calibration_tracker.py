"""
ascent/strategy/calibration_tracker.py
Tracks AI PM prediction calibration — conviction vs realized 21-day returns.

Conviction levels derived deterministically from thesis structure:
  - "high"         — symbol in thesis["quant_overrides"] (AI explicitly diverged)
  - "medium"       — symbol has a non-empty entry in thesis["position_rationale"]
                     but NOT in overrides
  - "quant_agreed" — symbol in thesis["quant_agreement"] or no explicit rationale
"""
from __future__ import annotations

import json
import logging
from datetime import date as _date, datetime
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

# Resolved relative to repo root (two levels up: strategy/ → ascent/ → repo root)
_REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_LOG = _REPO_ROOT / "logs" / "ai_pm_calibration.jsonl"

CONVICTION_ORDER: Dict[str, int] = {"high": 2, "medium": 1, "quant_agreed": 0}


# ── Internal helpers ───────────────────────────────────────────────────────────

def _derive_conviction(symbol: str, thesis: dict) -> str:
    """
    Derive conviction level for a symbol from thesis structure.
    Deterministic — no LLM calls.
    """
    overrides = thesis.get("quant_overrides") or []
    override_symbols = set()
    for item in overrides:
        if isinstance(item, dict):
            s = item.get("symbol", "")
            if s:
                override_symbols.add(s)
        elif isinstance(item, str):
            override_symbols.add(item)

    if symbol in override_symbols:
        return "high"

    rationale = thesis.get("position_rationale") or {}
    if isinstance(rationale, dict):
        entry = rationale.get(symbol)
        if entry and isinstance(entry, str) and entry.strip():
            return "medium"

    return "quant_agreed"


def _thesis_has_reasoning(thesis: dict) -> bool:
    """True if the thesis carries any AI reasoning — a real Phase-2 decision.

    A genuine AI PM decision always contributes reasoning via quant_overrides
    (AI explicitly diverged) and/or non-empty position_rationale entries. An
    off-calendar / daily-view stub (default top-N names, empty thesis) carries
    none, and must not pollute the calibration ledger.
    """
    if thesis.get("quant_overrides"):
        return True
    rationale = thesis.get("position_rationale")
    if isinstance(rationale, dict):
        for v in rationale.values():
            if isinstance(v, str) and v.strip():
                return True
            if isinstance(v, dict) and v:
                return True
    return False


def _read_log() -> list:
    """Read all entries from the calibration log. Returns [] on any error."""
    if not CALIBRATION_LOG.exists():
        return []
    entries = []
    try:
        with open(CALIBRATION_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return entries


def _write_log(entries: list) -> None:
    """Rewrite the entire log file from a list of entries."""
    import os
    CALIBRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CALIBRATION_LOG.parent / (CALIBRATION_LOG.name + ".tmp")
    with open(tmp, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    os.replace(tmp, CALIBRATION_LOG)


# ── Public API ─────────────────────────────────────────────────────────────────

def log_prediction(
    run_date: str,
    portfolio: dict,
    thesis: dict,
) -> None:
    """
    Log a new prediction entry. Skip if run_date already exists.
    Derives conviction level for each position from thesis structure.
    Never raises — all failures are silent.
    """
    try:
        entries = _read_log()

        # Dedup guard
        existing_dates = {e.get("date") for e in entries}
        if run_date in existing_dates:
            log.debug("[CalibrationTracker] Entry for %s already exists — skipping", run_date)
            return

        # Stub-pollution guard: only log real AI PM decisions. A thesis with no
        # AI reasoning (no overrides, no rationale) is an off-calendar / daily-view
        # non-decision and must not enter the calibration ledger.
        if not _thesis_has_reasoning(thesis):
            log.info("[CalibrationTracker] Skipping non-decision entry for %s "
                     "(empty thesis — no AI reasoning)", run_date)
            return

        positions = {}
        for symbol, weight in portfolio.items():
            conviction = _derive_conviction(symbol, thesis)
            rationale_map = thesis.get("position_rationale") or {}
            rationale = rationale_map.get(symbol) if isinstance(rationale_map, dict) else None
            positions[symbol] = {
                "weight":       float(weight),
                "conviction":   conviction,
                "rationale":    rationale if isinstance(rationale, str) and rationale.strip() else None,
                "realized_21d": None,
            }

        entry = {
            "date":      run_date,
            "positions": positions,
        }

        CALIBRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

        log.info("[CalibrationTracker] Logged %d positions for %s", len(positions), run_date)

    except Exception as exc:
        log.warning("[CalibrationTracker] log_prediction failed silently: %s", exc)


def update_outcomes(
    price_returns: Dict[str, float],
    as_of_date: str,
    lookback_days: int = 21,
) -> int:
    """
    For entries where realized_21d is null and the entry is >= lookback_days
    calendar days old from as_of_date, fill realized_21d as weighted sum
    of price_returns for that entry's positions.

    price_returns is {symbol: total_return_since_entry}.
    Passing an empty dict is a valid no-op (no returns to fill).

    Rewrites file in place. Returns count of entries updated.
    Never raises.
    """
    try:
        entries = _read_log()
        if not entries:
            return 0

        try:
            as_of = _date.fromisoformat(as_of_date[:10])
        except Exception:
            return 0

        n_updated = 0
        for entry in entries:
            try:
                entry_date = _date.fromisoformat(entry["date"][:10])
                age_days = (as_of - entry_date).days
            except Exception:
                continue

            if age_days < lookback_days:
                continue

            positions = entry.get("positions", {})
            any_null = any(pos.get("realized_21d") is None for pos in positions.values())
            if not any_null:
                continue

            if not price_returns:
                continue

            # Compute weighted return for this entry
            total_weight = sum(pos.get("weight", 0.0) for pos in positions.values())
            if total_weight <= 0:
                continue

            realized = 0.0
            for symbol, pos in positions.items():
                ret = price_returns.get(symbol)
                if ret is not None:
                    realized += (pos.get("weight", 0.0) / total_weight) * float(ret)

            # Fill realized_21d on each position
            for symbol, pos in positions.items():
                if pos.get("realized_21d") is None:
                    sym_ret = price_returns.get(symbol)
                    pos["realized_21d"] = float(sym_ret) if sym_ret is not None else None

            entry["realized_21d_portfolio"] = realized
            n_updated += 1

        if n_updated > 0:
            _write_log(entries)

        return n_updated

    except Exception as exc:
        log.warning("[CalibrationTracker] update_outcomes failed silently: %s", exc)
        return 0


def get_calibration_report(n_rebalances: int = 10) -> str:
    """
    Compute IC (Spearman rank correlation) between conviction_order and realized_21d
    across the last n_rebalances where realized_21d is not null.

    Returns formatted plain-text report. Returns "No calibration data yet." if no
    outcomes exist.
    Never raises.
    """
    try:
        from scipy.stats import spearmanr  # type: ignore
    except ImportError:
        return "Calibration report unavailable: scipy not installed."

    try:
        entries = _read_log()

        # Keep only entries with at least one realized position
        entries_with_outcomes = [
            e for e in entries
            if any(
                pos.get("realized_21d") is not None
                for pos in e.get("positions", {}).values()
            )
        ]

        if not entries_with_outcomes:
            return "No calibration data yet."

        # Take last n_rebalances
        recent = entries_with_outcomes[-n_rebalances:]

        # Flatten to (conviction_order, realized_21d) pairs
        conviction_scores = []
        realized_returns = []
        by_conviction: Dict[str, list] = {"high": [], "medium": [], "quant_agreed": []}

        for entry in recent:
            for symbol, pos in entry.get("positions", {}).items():
                r21 = pos.get("realized_21d")
                if r21 is None:
                    continue
                conviction = pos.get("conviction", "quant_agreed")
                order = CONVICTION_ORDER.get(conviction, 0)
                conviction_scores.append(order)
                realized_returns.append(float(r21))
                if conviction in by_conviction:
                    by_conviction[conviction].append(float(r21))

        total_positions = len(conviction_scores)
        positions_with_outcomes = sum(1 for r in realized_returns if r is not None)

        if total_positions == 0:
            return "No calibration data yet."

        # Compute IC
        if len(set(conviction_scores)) <= 1:
            ic_str = "N/A"
            interpretation = "Insufficient variation"
        else:
            try:
                corr, _ = spearmanr(conviction_scores, realized_returns)
                if corr != corr:  # NaN check
                    ic_str = "N/A"
                    interpretation = "Insufficient variation"
                else:
                    ic = float(corr)
                    ic_str = f"{ic:+.2f}"
                    if ic >= 0.20:
                        interpretation = "Calibrated"
                    elif ic >= 0.05:
                        interpretation = "Weak"
                    else:
                        interpretation = "Uncalibrated"
            except Exception:
                ic_str = "N/A"
                interpretation = "Insufficient variation"

        # Build per-conviction breakdown
        lines = [
            f"AI PM Calibration Report (last {len(recent)} rebalance(s) with outcomes):",
            f"  Total positions tracked: {total_positions}",
            f"  Positions with realized outcomes: {positions_with_outcomes}",
            f"  IC (conviction → 21d return): {ic_str}  [{interpretation}]",
            "",
            "  By conviction:",
        ]

        for level in ("high", "medium", "quant_agreed"):
            rets = by_conviction[level]
            if rets:
                avg = sum(rets) / len(rets)
                sign = "+" if avg >= 0 else ""
                lines.append(
                    f"    {level} ({len(rets)} position(s)): avg 21d return = {sign}{avg * 100:.1f}%"
                )
            else:
                lines.append(f"    {level} (0 positions): no data")

        return "\n".join(lines)

    except Exception as exc:
        log.warning("[CalibrationTracker] get_calibration_report failed: %s", exc)
        return f"Calibration report unavailable: {exc}"
