import json
import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)

_SIGNAL_PATH = "dashboard/regime_signal.json"


def compute_regime_trajectory(
    date: str,
    signal_path: str = _SIGNAL_PATH,
) -> Dict[str, Any]:
    """
    Reads regime_signal.json series to compute stability and stress trend.
    stability_10d: fraction of last 10 days matching current label.
    rs_trend: slope direction of regime stress over last 5 days.
    """
    path = Path(signal_path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        log.warning("[RegimeTrajectory] Failed to read signal: %s", e)
        return {}

    current_label = data.get("label", "unknown")
    series = data.get("series", [])
    if not series:
        return {}

    recent_10 = series[-10:]
    stability = sum(1 for e in recent_10 if e.get("label") == current_label) / len(recent_10)

    recent_5_rs = [e.get("rs", 0.0) for e in series[-5:]]
    if len(recent_5_rs) >= 2:
        slope = recent_5_rs[-1] - recent_5_rs[0]
        rs_trend = "rising" if slope > 0.01 else "falling" if slope < -0.01 else "flat"
    else:
        rs_trend = "unknown"

    days_in_regime = 0
    for entry in reversed(series):
        if entry.get("label") == current_label:
            days_in_regime += 1
        else:
            break

    return {
        "current_label":  current_label,
        "stability_10d":  round(stability, 3),
        "rs_trend":       rs_trend,
        "days_in_regime": days_in_regime,
        "as_of":          data.get("as_of", ""),
    }
