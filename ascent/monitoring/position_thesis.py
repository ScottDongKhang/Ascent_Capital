import json
import logging
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

_THESES_DIR = "outputs/ai_pm_theses"

_SYSTEM = (
    "You are an institutional equity analyst with access to live position performance data. "
    "For each position you have: the original investment thesis, return since last rebalance, "
    "current momentum, and alpha rank percentile in the 900-symbol universe. "
    "Assess whether each position's thesis still holds. "
    "For each symbol respond in ≤70 words: "
    "thesis status (intact/weakening/broken), "
    "specific data point that supports or contradicts, and the key near-term risk. "
    "If a position is DETERIORATING (poor rank + negative return), explain why we still hold it "
    "or flag it for review."
)


def _load_last_thesis_rationale() -> Dict[str, str]:
    theses_dir = Path(_THESES_DIR)
    if not theses_dir.exists():
        return {}
    files = sorted(theses_dir.glob("*.json"))
    if not files:
        return {}
    try:
        data = json.loads(files[-1].read_text())
        rationale = data.get("position_rationale", {})
        return {k: str(v) for k, v in rationale.items()}
    except Exception:
        return {}


def update_position_theses(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    position_health: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Calls Haiku once with all held positions to update each thesis.
    position_health: dict of {symbol: PositionHealth} from position_health.py — enriches prompt.
    Returns {} on any failure — never blocks the daily run.
    """
    if not merged_weights:
        return {}

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return {}

    last_rationale = _load_last_thesis_rationale()

    lines = [f"Date: {date}", "Held positions with live performance data:"]
    for sym, wt in merged_weights.items():
        rationale = last_rationale.get(sym, "No prior rationale available.")

        # Health data — gives Haiku real numbers to reason from
        health_str = ""
        if position_health and sym in position_health:
            h = position_health[sym]
            parts = [f"[{h.flag}]"]
            if h.return_since_rebalance is not None:
                parts.append(f"ret_since_rebal={h.return_since_rebalance:+.1%}")
            if h.momentum_252d is not None:
                parts.append(f"mom252d={h.momentum_252d:+.0%}")
            if h.alpha_rank_pct is not None:
                parts.append(f"alpha_rank={h.alpha_rank_pct:.0%} in universe")
            health_str = "  " + "  ".join(parts)

        lines.append(
            f"\n{sym} ({wt:.1%} weight){health_str}:\n"
            f"  Original rationale: {rationale[:200]}"
        )

    lines.append(
        "\nReturn a JSON object: {\"SYMBOL\": \"updated thesis in ≤70 words\", ...}"
        " for every symbol above. Flag DETERIORATING positions explicitly."
    )

    try:
        raw = generate_structured(
            system_prompt=_SYSTEM,
            user_prompt="\n".join(lines),
            model=HAIKU_MODEL,
            max_tokens=2500,
            temperature=0.3,
            use_cache=True,
        )
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {}
        return json.loads(raw[start:end])
    except Exception as e:
        log.warning("[PositionThesis] Failed: %s", e)
        return {}
