"""Path B: counterfactual return-delta scoring for named subsystems.

Reuses the canonical counterfactual tracks (ascent/analyst/catalog/registry.py) where an
existing track already isolates the subsystem being tested. earned_authority and
debate_judge_intervention map onto the existing Track D (pure AI PM) vs Track A* (pure quant)
pair -- that pair is exactly "with AI-layer influence" vs "without it", which is what those two
subsystems inject. regime_overlay and hedge_overlay don't have an existing isolating track, so
they map onto the same pair as a documented approximation pending sub-project 2's design work to
build a dedicated synthetic track for each -- recorded here, not hidden.

score_subsystem reuses ICResult/score_ic_series from stats.py: for this path "ic_mean"/"ic_t"
hold the mean/t-stat of the daily WITH-minus-WITHOUT return delta, not a rank correlation. This
is a deliberate field reuse (avoids a near-duplicate dataclass), not a naming mismatch.
"""
from __future__ import annotations

from ascent.analyst.catalog import registry
from ascent.analyst.proof_audit.stats import ICResult, score_ic_series

# (with_component_track, without_component_track), both canonical names from registry.py
SUBSYSTEM_TRACK_PAIRS: dict[str, tuple[str, str]] = {
    "earned_authority": ("counterfactual.track_d", "counterfactual.track_astar"),
    "debate_judge_intervention": ("counterfactual.track_b", "counterfactual.track_d"),
    "regime_overlay": ("counterfactual.track_d", "counterfactual.track_astar"),
    "hedge_overlay": ("counterfactual.track_d", "counterfactual.track_astar"),
}


def score_subsystem(name: str) -> ICResult:
    if name not in SUBSYSTEM_TRACK_PAIRS:
        raise KeyError(f"unknown subsystem {name!r}; known: {sorted(SUBSYSTEM_TRACK_PAIRS)}")
    with_name, without_name = SUBSYSTEM_TRACK_PAIRS[name]
    with_series = registry.load(with_name)
    without_series = registry.load(without_name)
    aligned = with_series.to_frame("with").join(without_series.to_frame("without"), how="inner")
    delta = (aligned["with"] - aligned["without"]).tolist()
    # score_ic_series expects a parallel (ic, ls_return) pair per date; for a return-delta
    # series there is only one meaningful series, so we pass it as both arguments.
    return score_ic_series(delta, delta)
