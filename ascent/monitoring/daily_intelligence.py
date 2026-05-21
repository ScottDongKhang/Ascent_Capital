import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict

from ascent.monitoring.conviction_tracker  import compute_conviction_decay
from ascent.monitoring.signal_health       import compute_signal_health
from ascent.monitoring.regime_trajectory   import compute_regime_trajectory
from ascent.monitoring.analogue_search     import find_historical_analogues
from ascent.monitoring.position_thesis     import update_position_theses
from ascent.monitoring.adversarial_daily   import generate_adversarial_challenge
from ascent.monitoring.macro_calendar      import build_event_calendar

log = logging.getLogger(__name__)

_OUTPUT_DIR = "data_cache/daily_intelligence"


def run_daily_intelligence(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    output_dir: str = _OUTPUT_DIR,
) -> Dict:
    """
    Runs all 7 intelligence modules. Each failure is caught independently.
    Writes result to output_dir/YYYY-MM-DD.json atomically. Returns the dict.
    """
    log.info("[DailyIntel] Running non-rebalance intelligence for %s", date)

    def _safe(name, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.warning("[DailyIntel] %s failed: %s", name, e)
            return {} if name != "adversarial" else ""

    traj        = _safe("regime_trajectory", compute_regime_trajectory, date)
    regime      = traj.get("current_label", "unknown") if isinstance(traj, dict) else "unknown"
    conviction  = _safe("conviction_decay",  compute_conviction_decay,
                         date, merged_weights, agent_outputs)
    signal      = _safe("signal_health",     compute_signal_health, date)
    analogues   = _safe("analogue_search",   find_historical_analogues,
                         date, traj, signal)
    theses      = _safe("position_thesis",   update_position_theses,
                         date, merged_weights, agent_outputs)
    adversarial = _safe("adversarial",       generate_adversarial_challenge,
                         date, merged_weights, agent_outputs, regime)
    macro_evts  = _safe("macro_calendar",    build_event_calendar,
                         date, merged_weights, agent_outputs)

    entry = {
        "date":                  date,
        "conviction_decay":      conviction,
        "signal_health":         signal,
        "regime_trajectory":     traj,
        "historical_analogues":  analogues,
        "position_theses":       theses,
        "adversarial_challenge": adversarial,
        "macro_events":          macro_evts,
    }

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.json"

    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(entry, f, indent=2)
        os.replace(tmp_path, out_path)
    except Exception as e:
        log.error("[DailyIntel] Failed to write %s: %s", out_path, e)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    log.info("[DailyIntel] Written to %s", out_path)
    return entry
