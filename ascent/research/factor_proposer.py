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

# Sleeves that are not part of the currently-live weight set but are explicit,
# already-identified re-testable candidates (see docs/target_architecture/
# 27_trend_insider_reconciliation.md). The proposer/guided-variant path is
# allowed to reference these even when they are absent from current_weights;
# generate_guided_variants() lazily seeds them at 0.0 only when a hypothesis
# actually proposes touching one, so hypotheses that don't mention them leave
# the default 2-sleeve behavior untouched.
# NOTE: "trend" is deliberately excluded -- the same doc found it has a
# statistically significant *negative* IC and it must not be revived here.
_ELIGIBLE_CANDIDATE_SLEEVES = {"insider"}

# Hypotheses whose resolved universe has fewer than this many symbols are
# rejected outright -- mirrors the N_LEGS*2 == 10 minimum tradeable-symbol
# floor used by the proof-audit long/short construction
# (ascent/analyst/proof_audit/wf_scorer.py::MIN_SYMBOLS_PER_DATE); the
# dormant alternatives_agent's own long/short-leg floor no longer exists in
# the codebase (removed 2026-08-23) so this is the closest live precedent.
MIN_UNIVERSE_SYMBOLS = 10

_REQUIRED_HYPOTHESIS_FIELDS = (
    "signal_id",
    "universe",
    "expected_ic_sign",
    "uncorrelation_rationale",
    "weight_biases",
)

_SYSTEM_PROMPT = (
    "You are a quantitative researcher at Ascent Capital. "
    "You propose alpha weight hypotheses for a multi-sleeve trading system. "
    "Each hypothesis must have a clear economic thesis, a concrete symbol universe, "
    "an expected IC sign, concrete weight biases, and -- critically -- an explicit "
    "explanation of why the idea would be uncorrelated with the existing meanrev/statarb "
    "sleeves. Breadth only helps risk-adjusted return (Fundamental Law of Active "
    "Management: IR ~= IC * sqrt(breadth)) when the added signals are genuinely "
    "uncorrelated with what is already live, so every hypothesis must justify that "
    "explicitly, not just assert a return edge. "
    "Respond ONLY with a valid JSON array. No other text."
)

_USER_TEMPLATE = """Current regime: {regime}

Current alpha sleeve weights:
{weights_str}

Propose {n} diverse hypotheses for sleeve weight adjustments that might outperform.
Each hypothesis should reflect a different economic reasoning about what works in a {regime} environment.

The available sleeves are: {sleeves}
{candidate_sleeve_note}

Respond with a JSON array of exactly {n} hypothesis objects:
[
  {{
    "signal_id": "short_slug_no_spaces",
    "thesis": "One-sentence economic rationale",
    "universe": ["AAPL", "MSFT", ...]  // MUST be an explicit list of ticker symbols -- named universe strings (e.g. "sp500") are not resolvable and will be rejected
    "expected_ic_sign": 1,  // +1 or -1
    "uncorrelation_rationale": "Why this idea would be uncorrelated with the existing meanrev/statarb sleeves",
    "weight_biases": {{"sleeve_name": delta_float, ...}}
  }},
  ...
]

Rules:
- Each bias is a delta (positive = increase, negative = decrease), typically between -0.15 and +0.15
- Biases for a single hypothesis must sum to approximately 0.0 (weight-neutral)
- Only include sleeves you want to change; unlisted sleeves are unchanged
- Hypotheses must be meaningfully different from each other
- "universe" must be an explicit list with at least {min_universe} symbols -- named universe
  strings cannot be resolved and are rejected outright; hypotheses concentrated in fewer
  names than the floor will also be rejected
- "uncorrelation_rationale" is required and must be a substantive reason (mechanism,
  horizon, or data source difference), not a restatement of the thesis"""


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


def _resolve_universe_size(universe) -> int:
    """
    Resolve a hypothesis's "universe" field to a symbol count.

    A list of symbols is counted directly. A named-universe string (e.g.
    "sp500") CANNOT be resolved to a real symbol count here: this module has
    no live name->universe lookup for arbitrary strings (ascent/data/universe.py
    only exposes get_universe_on_date(date, ...) and build_historical_universe(),
    neither of which maps an LLM-invented name like "my_3_favorite_biotech_names"
    to anything). Since the LLM is free to invent any string, silently assuming
    a string is "broad enough" lets a narrow or fabricated named universe sail
    past the concentration floor. An unresolvable string is therefore treated
    as size 0 -- it fails the floor check by default. Only an explicit symbol
    list can clear the floor.
    """
    if isinstance(universe, (list, tuple, set)):
        # de-dupe defensively; a hypothesis listing the same symbol twice
        # should not appear to clear the floor
        return len({str(s).strip().upper() for s in universe if str(s).strip()})
    return 0


def _validate_hypothesis(h: dict) -> bool:
    """
    Structural + rejection-rule validation for one raw hypothesis dict.

    Rejects (returns False) when:
    - any required field is missing (signal_id, universe, expected_ic_sign,
      uncorrelation_rationale, weight_biases), or a thesis/narrative is absent
    - expected_ic_sign is not exactly +1 or -1 (0 or any other value is invalid --
      it is not a meaningful sign, not a valid "missing field" sentinel either)
    - the resolved universe has fewer than MIN_UNIVERSE_SYMBOLS symbols
    """
    if not isinstance(h, dict):
        return False
    if "thesis" not in h and "narrative" not in h:
        return False
    for field in _REQUIRED_HYPOTHESIS_FIELDS:
        if field not in h or h[field] in (None, ""):
            return False
    # expected_ic_sign must be exactly +1 or -1. A JSON parse of an int literal
    # produces a Python int; guard against a float form (1.0/-1.0) too in case
    # the LLM emits one. bool is an int subclass in Python -- exclude it
    # explicitly so True/False can't slip through as 1/-1's numeric equivalent.
    sign = h.get("expected_ic_sign")
    if isinstance(sign, bool) or not isinstance(sign, (int, float)) or sign not in (1, -1):
        return False
    if not isinstance(h.get("weight_biases"), dict):
        return False
    if _resolve_universe_size(h.get("universe")) < MIN_UNIVERSE_SYMBOLS:
        return False
    return True


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

    candidates = sorted(_ELIGIBLE_CANDIDATE_SLEEVES - set(current_weights.keys()))
    candidate_sleeve_note = (
        f"You may also propose upweighting these dormant, zero-weighted, "
        f"re-testable candidate sleeves (not part of the live set above): "
        f"{', '.join(candidates)}."
        if candidates else ""
    )

    user_prompt = _USER_TEMPLATE.format(
        regime=regime,
        weights_str=weights_str,
        n=n,
        sleeves=", ".join(sleeves),
        candidate_sleeve_note=candidate_sleeve_note,
        min_universe=MIN_UNIVERSE_SYMBOLS,
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
        hypotheses = [h for h in parsed if _validate_hypothesis(h)]
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
            if sleeve not in weights and sleeve in _ELIGIBLE_CANDIDATE_SLEEVES:
                # Lazily admit an explicit re-testable candidate (e.g. "insider")
                # only when a hypothesis actually proposes touching it -- this
                # keeps the default weight set's behavior unchanged for every
                # hypothesis that doesn't reference a candidate sleeve.
                weights[sleeve] = 0.0
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
            "hypothesis":    hyp.get("thesis", hyp.get("narrative", "")),
        })

    return variants
