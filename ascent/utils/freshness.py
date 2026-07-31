"""Expiry for AI-authored priors that steer the quant pipeline.

`ascent/main.py` consumes two files written by the AI PM's Phase 1:

    data_cache/ai_prethesis_latest.json    -> floors conviction alpha, zeroes the avoid list
    data_cache/ai_regime_assessment.json   -> blends the regime label + risk multiplier

Both are read on every pipeline run, and neither read was date-gated. On the
2026-07-27 run the pre-thesis on disk was dated 2026-06-24 — 33 days old — and
six of its ten conviction names were in the executed book. Because these two
channels bypass the earned-authority budget entirely, a month-old opinion was
steering position sizing with no cap and no expiry.

It also contaminates the measurement: Track A-star ("pure quant") is snapshotted
from a pipeline that has already consumed these files, so the baseline used to
judge the AI PM contains the AI PM's stale opinions. Gating on age does not fully
remove that — a *fresh* prior still influences A-star — but it bounds the damage
to the current rebalance cycle instead of letting one file steer indefinitely.

Fails CLOSED: anything unparseable, absent, or future-dated is treated as stale.
A prior that cannot prove its age does not get to move the portfolio.

---

A second, related problem lives here too: `dashboard/regime_signal.json` (written
by `ascent/regime/engine.py`) is read with no freshness check at all by four
reporting/monitoring call sites (`investor_report.py`, `investor_letter.py`,
`weekly_debrief.py`, `scenario_planner.py`). In one audit it sat 5 weeks stale
while those readers kept publishing its label as current. `fresh_regime_label()`
below applies the same fail-closed freshness gate and, on staleness, falls back
to `dashboard/regime_labels.csv` — the regime engine's own live output, always as
current as the last pipeline run.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Optional

from ascent.utils.market_time import market_today

__all__ = [
    "AI_PRIOR_MAX_AGE_DAYS",
    "ai_prior_is_fresh",
    "REGIME_SIGNAL_PATH",
    "REGIME_LABELS_CSV",
    "regime_label_from_csv",
    "fresh_regime_label",
]

#: Scheduled rebalances sit ~14 calendar days apart (see rebalance_calendar.csv),
#: so one cycle is the longest a Phase-1 view can still be called "current".
#: Deliberately calendar days, not trading days: the question is how stale the
#: *opinion* is, and markets move over weekends too.
AI_PRIOR_MAX_AGE_DAYS = 14


def ai_prior_is_fresh(
    as_of_date: Any,
    today: Optional[dt.date] = None,
    max_age_days: int = AI_PRIOR_MAX_AGE_DAYS,
) -> bool:
    """True if an AI prior stamped `as_of_date` may still steer the pipeline.

    `as_of_date` is whatever was in the JSON — commonly an ISO string, possibly
    missing or malformed. `today` defaults to the market date, not the host date
    (this host runs at UTC+7; see ascent/utils/market_time.py).
    """
    if not isinstance(as_of_date, str):
        return False
    try:
        stamped = dt.date.fromisoformat(as_of_date.strip())
    except (ValueError, AttributeError):
        return False

    ref = today if today is not None else market_today()
    age = (ref - stamped).days
    if age < 0:
        return False  # future-dated: clock or --date override bug, not signal
    return age <= max_age_days


# ── Regime-signal freshness (separate concern: pipeline staleness, not an ────
# ── AI opinion, but the same fail-closed shape) ──────────────────────────────

REGIME_SIGNAL_PATH = Path("dashboard/regime_signal.json")
REGIME_LABELS_CSV = Path("dashboard/regime_labels.csv")


def _parse_iso_date(value: Any) -> Optional[dt.date]:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value.strip())
    except (ValueError, AttributeError):
        return None


def _age_days(as_of: Any, today: Optional[dt.date] = None) -> Optional[int]:
    stamped = _parse_iso_date(as_of)
    if stamped is None:
        return None
    ref = today if today is not None else market_today()
    return (ref - stamped).days


def regime_label_from_csv(csv_path: Optional[Path] = None) -> Optional[str]:
    """Most recent regime label from the engine's own CSV output.

    This is the regime engine's live ground truth, not an AI opinion — it has
    no separate "as-of" question beyond "was this the last row written," so
    there is nothing to gate here. Returns None if the file is absent, empty,
    or missing a `label` column.

    `csv_path` resolves the module-level `REGIME_LABELS_CSV` at call time
    (not as a bound default) so tests can monkeypatch the module attribute.
    """
    csv_path = csv_path if csv_path is not None else REGIME_LABELS_CSV
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            last: Optional[dict] = None
            for row in reader:
                if row:
                    last = row
        if last is None:
            return None
        return last.get("label") or None
    except Exception:
        return None


def fresh_regime_label(
    signal_path: Optional[Path] = None,
    csv_fallback_path: Optional[Path] = None,
    today: Optional[dt.date] = None,
    max_age_days: int = AI_PRIOR_MAX_AGE_DAYS,
) -> dict:
    """Regime label for reporting call sites, gated on `regime_signal.json` freshness.

    Reads `dashboard/regime_signal.json`, checks its `as_of` (falling back to
    `last_refit_date`) against `max_age_days`. If fresh, returns that label.
    If stale, absent, or unparseable, falls back to `regime_label_from_csv()`
    — the regime engine's own live output — and marks the result stale so
    callers can say so rather than silently publishing a confident label from
    a file that may be weeks old.

    `signal_path` / `csv_fallback_path` resolve the module-level
    `REGIME_SIGNAL_PATH` / `REGIME_LABELS_CSV` at call time (not as bound
    defaults) so tests can monkeypatch the module attributes.

    Returns:
        {
            "label": str,               # best available label, "unknown" if none
            "stale": bool,               # True if regime_signal.json was not trusted
            "age_days": int | None,      # age of regime_signal.json's as_of, if parseable
            "source": "signal" | "csv_fallback" | "unknown",
        }
    """
    signal_path = signal_path if signal_path is not None else REGIME_SIGNAL_PATH
    label: Optional[str] = None
    as_of: Any = None
    try:
        data = json.loads(signal_path.read_text())
        label = data.get("regime") or data.get("label")
        as_of = data.get("as_of") or data.get("last_refit_date")
    except Exception:
        pass

    age = _age_days(as_of, today)

    if label and ai_prior_is_fresh(as_of, today=today, max_age_days=max_age_days):
        return {"label": label, "stale": False, "age_days": age, "source": "signal"}

    csv_label = regime_label_from_csv(csv_fallback_path)
    if csv_label:
        return {"label": csv_label, "stale": True, "age_days": age, "source": "csv_fallback"}
    return {"label": label or "unknown", "stale": True, "age_days": age, "source": "unknown"}
