import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

_THESES_DIR = "outputs/ai_pm_theses"

_SYSTEM = (
    "You are an institutional equity analyst. Given today's portfolio data, "
    "assess whether each position's original investment thesis still holds. "
    "For each symbol respond in ≤60 words: thesis status (intact/weakening/broken), "
    "one key supporting or contradicting data point, and any near-term risk. "
    "Be specific and quantitative where possible."
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
) -> Dict[str, str]:
    """
    Calls Haiku once with all held positions to update each thesis.
    Returns {} on any failure — never blocks the daily run.
    """
    if not merged_weights:
        return {}

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return {}

    alpha_context = {}
    us_agent = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
    if us_agent is not None and us_agent.alpha_scores is not None and not us_agent.alpha_scores.empty:
        try:
            latest = us_agent.alpha_scores.iloc[-1].to_dict()
            alpha_context = {s: round(latest[s], 3) for s in merged_weights if s in latest}
        except Exception:
            pass

    last_rationale = _load_last_thesis_rationale()

    lines = [f"Date: {date}", "Held positions:"]
    for sym, wt in merged_weights.items():
        alpha = alpha_context.get(sym, "N/A")
        rationale = last_rationale.get(sym, "No prior rationale available.")
        lines.append(
            f"\n{sym} ({wt:.1%} weight, alpha={alpha}):\n"
            f"  Original rationale: {rationale[:200]}"
        )
    lines.append(
        "\nReturn a JSON object: {\"SYMBOL\": \"updated thesis in ≤60 words\", ...}"
        " for every symbol above."
    )

    try:
        raw = generate_structured(
            system_prompt=_SYSTEM,
            user_prompt="\n".join(lines),
            model=HAIKU_MODEL,
            max_tokens=2000,
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
