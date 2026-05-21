import json
import logging
from pathlib import Path
from typing import Dict, List, Any

log = logging.getLogger(__name__)

_EPISODES_PATH = "logs/regime_episodes.jsonl"


def find_historical_analogues(
    date: str,
    regime_trajectory: Dict[str, Any],
    signal_health: Dict[str, Any],
    episodes_path: str = _EPISODES_PATH,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Scores past regime episodes by similarity to current conditions.
    Only considers episodes with realized_return_21d populated and date < today.
    Returns top_n matches with outcomes.
    """
    path = Path(episodes_path)
    if not path.exists():
        return []

    current_label = regime_trajectory.get("current_label", "")

    episodes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ep = json.loads(line)
            if ep.get("realized_return_21d") is None:
                continue
            if ep.get("date", "") >= date:
                continue
            ep_regime = ep.get("regime", "")
            if current_label and not ep_regime.startswith(current_label.split("_")[0]):
                continue
            episodes.append(ep)
        except Exception:
            continue

    if not episodes:
        return []

    scored = []
    for ep in episodes:
        regime_match = 1.0 if ep.get("regime") == current_label else 0.5
        scored.append((1.0 - regime_match, ep))

    scored.sort(key=lambda x: x[0])
    top = [ep for _, ep in scored[:top_n]]

    return [
        {
            "date":        ep["date"],
            "regime":      ep.get("regime", ""),
            "outcome_21d": ep["realized_return_21d"],
            "n_positions": len(ep.get("quant_weights", {})),
        }
        for ep in top
    ]
