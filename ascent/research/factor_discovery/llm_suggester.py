"""
ascent/research/factor_discovery/llm_suggester.py

LLM-guided template parameter suggestion (Claude Haiku).

The LLM proposes parameters for pre-defined template families via JSON only.
Zero code injection risk — trusted Python templates do all the computation.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a quantitative researcher at Ascent Capital. Your job is to suggest
parameters for alpha signal templates based on the current market regime and
recent signal performance statistics.

You will choose ONE template family and ONE set of parameters that you
believe will capture a signal orthogonal to the existing ones.

You ONLY return a JSON object. No code, no explanation outside the JSON."""

_USER_TEMPLATE = """\
Current regime: {regime}

Existing factor IC statistics (avoid overlap with these):
{ic_context}

Available template families:
- MomentumTemplate: lookback (21–252 days), skip_days (0–63), normalization (zscore/rank/minmax)
- ReversionTemplate: lookback (2–21 days), smooth_window (1–10), normalization
- VolatilityTemplate: vol_window (10–63), vov_window (21–126), direction (low/trend), normalization
- QualityTemplate: metric (consistency/drawdown/trend_strength), window (21–252), normalization
- CorrelationTemplate: window (21–126), mode (beta/idiosyncratic), normalization

Think step by step:
1. What economic mechanism is likely to drive returns in a {regime} regime?
2. Which template family best captures that mechanism?
3. What parameter values reflect the regime's typical duration and dynamics?
4. Is this sufficiently different from the existing factors listed above?

Respond with exactly this JSON:
{{
  "template": "TemplateName",
  "params": {{}},
  "rationale": "One sentence — economic mechanism"
}}"""


def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        return generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.5,
            use_cache=False,
        )
    except Exception as exc:
        log.warning("[LLMSuggester] Call failed: %s", exc)
        return None


def suggest_template_params(
    regime: str,
    ic_context: Dict,
    existing_factor_names: Optional[List[str]] = None,
) -> Optional[dict]:
    """
    Ask Claude Haiku to suggest one template + parameters.
    Returns dict with {template, params, rationale} or None on failure.
    """
    ic_lines = "\n".join(
        f"  {name}: IC={ic:.3f}" for name, ic in ic_context.items()
    ) or "  (no IC data yet)"

    if existing_factor_names:
        ic_lines += "\nDeployed factors: " + ", ".join(existing_factor_names)

    user_prompt = _USER_TEMPLATE.format(regime=regime, ic_context=ic_lines)
    raw = _call_llm(_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return None

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        parsed = json.loads(raw[start:end])
        if "template" not in parsed or "params" not in parsed:
            return None
        return parsed
    except Exception as exc:
        log.warning("[LLMSuggester] Parse failed: %s", exc)
        return None
