import json
import logging
import os
import tempfile as _tempfile
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = "data_cache/last_rebalance_state.json"


def compute_conviction_decay(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    state_path: str = _DEFAULT_STATE_PATH,
) -> Dict[str, Any]:
    path = Path(state_path)
    if not path.exists():
        log.warning("[ConvictionTracker] No rebalance state found at %s", state_path)
        return {}

    try:
        last_state = json.loads(path.read_text())
    except Exception as e:
        log.warning("[ConvictionTracker] Failed to read state: %s", e)
        return {}

    us_agent = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
    if us_agent is None or us_agent.alpha_scores is None or us_agent.alpha_scores.empty:
        log.warning("[ConvictionTracker] No us_equities alpha scores available")
        return {}

    try:
        latest = us_agent.alpha_scores.iloc[-1]
        scores_today = latest.to_dict()
    except Exception as e:
        log.warning("[ConvictionTracker] Failed to extract alpha scores: %s", e)
        return {}

    sorted_symbols = sorted(scores_today, key=lambda s: scores_today[s], reverse=True)
    rank_today = {sym: idx + 1 for idx, sym in enumerate(sorted_symbols)}

    ranks_at_rebalance = last_state.get("alpha_ranks", {})
    scores_at_rebalance = last_state.get("alpha_scores", {})

    result = {}
    for sym in merged_weights:
        if sym not in rank_today:
            continue
        rank_then = ranks_at_rebalance.get(sym)
        score_then = scores_at_rebalance.get(sym)
        score_now = scores_today.get(sym, 0.0)
        decay_pct = (
            round((score_then - score_now) / abs(score_then) * 100, 1)
            if score_then and score_then != 0 else None
        )
        result[sym] = {
            "rank_at_rebalance": rank_then,
            "rank_today":        rank_today[sym],
            "score_at_rebalance": round(score_then, 4) if score_then is not None else None,
            "score_today":        round(score_now, 4),
            "decay_pct":          decay_pct,
        }

    return result


def save_rebalance_alpha_state(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    sleeve_ics: Dict[str, float],
    regime: str,
    regime_stability_10d: float,
    state_path: str = _DEFAULT_STATE_PATH,
) -> None:
    us_agent = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
    alpha_ranks, alpha_scores = {}, {}

    if us_agent is not None and us_agent.alpha_scores is not None and not us_agent.alpha_scores.empty:
        try:
            latest = us_agent.alpha_scores.iloc[-1].to_dict()
            sorted_syms = sorted(latest, key=lambda s: latest[s], reverse=True)
            alpha_ranks = {s: i + 1 for i, s in enumerate(sorted_syms) if s in merged_weights}
            alpha_scores = {s: round(latest[s], 4) for s in merged_weights if s in latest}
        except Exception as e:
            log.warning("[ConvictionTracker] Could not build rebalance snapshot: %s", e)

    state = {
        "date":                  date,
        "weights":               {k: round(v, 6) for k, v in merged_weights.items()},
        "alpha_ranks":           alpha_ranks,
        "alpha_scores":          alpha_scores,
        "sleeve_ics":            sleeve_ics,
        "regime":                regime,
        "regime_stability_10d":  round(regime_stability_10d, 4),
    }
    p = Path(state_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = _tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, p)
    except Exception as e:
        log.error("[ConvictionTracker] Write failed: %s", e)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    log.info("[ConvictionTracker] Rebalance state saved to %s", state_path)
