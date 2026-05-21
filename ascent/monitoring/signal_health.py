import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)

_IC_LOG     = "logs/sleeve_ic_log.jsonl"
_STATE_PATH = "data_cache/last_rebalance_state.json"
_WINDOW     = 5


def compute_signal_health(
    date: str,
    ic_log_path: str = _IC_LOG,
    state_path: str = _STATE_PATH,
) -> Dict[str, Any]:
    """
    Reads last _WINDOW unique-date entries from sleeve_ic_log.jsonl.
    Compares rolling average IC per sleeve to rebalance baseline.
    Status: healthy (>-20%), weakening (-20% to -50%), deteriorating (<-50%).
    """
    ic_path = Path(ic_log_path)
    if not ic_path.exists():
        return {}

    lines = [l for l in ic_path.read_text().splitlines() if l.strip()]
    if not lines:
        return {}

    seen_dates, recent = set(), []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            d = entry.get("date", "")
            if d and d not in seen_dates:
                seen_dates.add(d)
                recent.append(entry)
            if len(recent) >= _WINDOW:
                break
        except Exception:
            continue

    if not recent:
        return {}

    sleeve_ics: Dict[str, list] = defaultdict(list)
    for entry in recent:
        for sleeve, stats in entry.get("sleeves", {}).items():
            ic = stats.get("mean_ic")
            if ic is not None:
                sleeve_ics[sleeve].append(ic)

    baseline: Dict[str, float] = {}
    sp = Path(state_path)
    if sp.exists():
        try:
            baseline = json.loads(sp.read_text()).get("sleeve_ics", {})
        except Exception:
            pass

    result = {}
    for sleeve, ics in sleeve_ics.items():
        avg = sum(ics) / len(ics)
        base = baseline.get(sleeve)
        change_pct = round((avg - base) / abs(base) * 100, 1) if base is not None and base != 0 else None
        if change_pct is None:
            status = "unknown"
        elif change_pct >= -20:
            status = "healthy"
        elif change_pct >= -50:
            status = "weakening"
        else:
            status = "deteriorating"

        result[sleeve] = {
            "ic_at_rebalance": round(base, 4) if base is not None else None,
            "ic_5d_avg":       round(avg, 4),
            "change_pct":      change_pct,
            "status":          status,
        }

    return result
