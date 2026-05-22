from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_STATE_PATH   = Path("data_cache/last_rebalance_state.json")
_DEFAULT_TRIGGER_PATH = Path("data_cache/rebalance_trigger.json")


def check_ic_decay_trigger(
    date: str,
    current_ics: dict[str, float],
    state_path: Path = _DEFAULT_STATE_PATH,
    trigger_path: Path = _DEFAULT_TRIGGER_PATH,
    decay_threshold: float = 0.30,
    min_days: int = 5,
) -> bool:
    """
    Returns True and writes trigger_path if composite IC has decayed ≥decay_threshold
    since last rebalance AND ≥min_days business days have passed.
    """
    if not state_path.exists():
        return False

    try:
        state = json.loads(state_path.read_text())
        last_date = pd.Timestamp(state["date"])
        baseline_ics: dict[str, float] = state.get("sleeve_ics", {})
        baseline_composite: float = state.get("composite_ic", 0.0)
    except Exception as e:
        log.warning("[RebalanceTrigger] Could not read state: %s", e)
        return False

    today = pd.Timestamp(date)
    bdays_since = len(pd.bdate_range(last_date, today)) - 1
    if bdays_since < min_days:
        return False

    if not current_ics or baseline_composite <= 0:
        return False

    shared_keys = [k for k in baseline_ics if k in current_ics]
    if not shared_keys:
        return False

    current_composite = sum(current_ics[k] for k in shared_keys) / len(shared_keys)
    decay_pct = (baseline_composite - current_composite) / abs(baseline_composite)

    if decay_pct >= decay_threshold:
        payload = {
            "triggered_date": date,
            "days_since_rebalance": bdays_since,
            "baseline_ic": round(baseline_composite, 4),
            "current_ic": round(current_composite, 4),
            "ic_decay_pct": round(decay_pct, 4),
        }
        trigger_path.parent.mkdir(parents=True, exist_ok=True)
        trigger_path.write_text(json.dumps(payload, indent=2))
        log.warning(
            "[RebalanceTrigger] IC decay %.1f%% (baseline=%.4f, current=%.4f) "
            "after %d days — early rebalance triggered",
            decay_pct * 100, baseline_composite, current_composite, bdays_since,
        )
        return True

    return False


def consume_trigger(trigger_path: Path = _DEFAULT_TRIGGER_PATH) -> bool:
    """Returns True and deletes trigger_path if it exists."""
    if trigger_path.exists():
        trigger_path.unlink()
        return True
    return False


def is_triggered(trigger_path: Path = _DEFAULT_TRIGGER_PATH) -> bool:
    """Return True if an early rebalance has been flagged."""
    return trigger_path.exists()
