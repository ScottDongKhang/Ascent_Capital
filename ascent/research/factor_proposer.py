"""
ascent/research/factor_proposer.py

LLM-guided factor hypothesis generation for the self-improve loop.

Replaces purely random weight perturbation with regime-aware hypotheses:
1. Haiku proposes N narratives (e.g. "quality beats momentum in stress")
2. Each narrative includes weight biases (deltas to apply to current weights)
3. Cosine similarity check rejects hypotheses that are near-duplicates
4. generate_guided_variants() applies biases + small random noise -> full configs

Falls back to silent empty list if LLM is unavailable.

Inspired by AlphaAgent (Liu et al., 2025) -- arxiv.org/abs/2502.16789
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# Minimum weight floor per sleeve -- never perturb below these
_SLEEVE_FLOORS: Dict[str, float] = {
    "trend":          0.05,
    "fundamental":    0.02,
    "earnings":       0.02,
    "analyst":        0.02,
    "options_flow":   0.01,
    "insider":        0.01,
    "short_interest": 0.01,
    "llm_fundamental": 0.01,
}

_SYSTEM_PROMPT = (
    "You are a quantitative researcher at Ascent Capital. "
    "You propose alpha weight hypotheses for a multi-sleeve trading system. "
    "Each hypothesis must have a clear economic narrative and concrete weight biases. "
    "Respond ONLY with a valid JSON array. No other text."
)

_USER_TEMPLATE = """Current regime: {regime}

Current alpha sleeve weights:
{weights_str}

Propose {n} diverse hypotheses for sleeve weight adjustments that might outperform.
Each hypothesis should reflect a different economic reasoning about what works in a {regime} environment.

The available sleeves are: {sleeves}

Respond with a JSON array of exactly {n} hypothesis objects:
[
  {{
    "narrative": "One-sentence economic rationale",
    "weight_biases": {{"sleeve_name": delta_float, ...}}
  }},
  ...
]

Rules:
- Each bias is a delta (positive = increase, negative = decrease), typically between -0.15 and +0.15
- Biases for a single hypothesis must sum to approximately 0.0 (weight-neutral)
- Only include sleeves you want to change; unlisted sleeves are unchanged
- Hypotheses must be meaningfully different from each other"""


def _call_llm(system_prompt: str, user_prompt: str) -> Optional[str]:
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        return generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=800,
            temperature=0.6,
            use_cache=False,  # hypotheses should be fresh each week
        )
    except Exception as exc:
        log.warning("[FactorProposer] LLM call failed: %s", exc)
        return None


def _bias_vector(hypothesis: dict, sleeves: List[str]) -> np.ndarray:
    """Convert weight_biases dict to a fixed-length numpy vector for similarity comparison."""
    biases = hypothesis.get("weight_biases", {})
    return np.array([biases.get(s, 0.0) for s in sorted(sleeves)])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def deduplicate_hypotheses(
    hypotheses: List[dict],
    similarity_threshold: float = 0.85,
) -> List[dict]:
    """
    Remove near-duplicate hypotheses using cosine similarity of their bias vectors.
    Keeps the first occurrence when duplicates are found.
    """
    if not hypotheses:
        return []

    all_sleeves = set()
    for h in hypotheses:
        all_sleeves.update(h.get("weight_biases", {}).keys())
    sleeves = sorted(all_sleeves)

    kept = []
    vecs = []
    for h in hypotheses:
        v = _bias_vector(h, sleeves)
        duplicate = any(
            _cosine_similarity(v, existing) > similarity_threshold
            for existing in vecs
        )
        if not duplicate:
            kept.append(h)
            vecs.append(v)

    return kept


def propose_hypotheses(
    regime: str,
    current_weights: Dict[str, float],
    n: int = 5,
) -> List[dict]:
    """
    Ask Haiku to propose N regime-aware alpha weight hypotheses.

    Returns:
        List of hypothesis dicts: [{narrative: str, weight_biases: {sleeve: delta}}]
        Empty list if LLM unavailable or parse fails.
    """
    sleeves = sorted(current_weights.keys())
    weights_str = "\n".join(f"  {s}: {w:.2f}" for s, w in sorted(current_weights.items()))

    user_prompt = _USER_TEMPLATE.format(
        regime=regime,
        weights_str=weights_str,
        n=n,
        sleeves=", ".join(sleeves),
    )

    raw = _call_llm(_SYSTEM_PROMPT, user_prompt)
    if not raw:
        return []

    try:
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return []
        parsed = json.loads(raw[start:end])
        if not isinstance(parsed, list):
            return []
        hypotheses = [
            h for h in parsed
            if isinstance(h, dict) and "narrative" in h and "weight_biases" in h
        ]
        return deduplicate_hypotheses(hypotheses)
    except Exception as exc:
        log.warning("[FactorProposer] Parse failed: %s", exc)
        return []


def generate_guided_variants(
    current_weights: Dict[str, float],
    hypotheses: List[dict],
    perturb_range: float = 0.03,
) -> List[dict]:
    """
    Convert hypothesis weight biases into full variant configs.

    Applies bias + small random noise within perturb_range, then renormalizes.
    Enforces per-sleeve minimum floors from _SLEEVE_FLOORS.

    Returns:
        List of variant config dicts: [{variant_id, alpha_weights, hypothesis}]
    """
    from datetime import datetime
    variants = []

    for i, hyp in enumerate(hypotheses):
        biases  = hyp.get("weight_biases", {})
        weights = dict(current_weights)

        for sleeve, bias in biases.items():
            if sleeve in weights:
                noise = float(np.random.uniform(-perturb_range, perturb_range))
                weights[sleeve] = weights[sleeve] + bias + noise

        # Iterative floor-enforce + renormalize until all floors hold.
        # Convergence is guaranteed because floors are positive and bounded.
        for _ in range(20):
            for sleeve in weights:
                floor = _SLEEVE_FLOORS.get(sleeve, 0.0)
                weights[sleeve] = max(floor, weights[sleeve])
            total = sum(weights.values())
            if total <= 0:
                weights = dict(current_weights)
                break
            weights = {k: v / total for k, v in weights.items()}
            # Check if all floors are satisfied (within floating-point tolerance)
            if all(weights[s] >= _SLEEVE_FLOORS.get(s, 0.0) - 1e-9 for s in weights):
                break

        weights = {k: round(v, 6) for k, v in weights.items()}

        variants.append({
            "variant_id":    f"guided_{i+1}_{datetime.now().strftime('%Y%m%d')}",
            "alpha_weights": weights,
            "hypothesis":    hyp.get("narrative", ""),
        })

    return variants
