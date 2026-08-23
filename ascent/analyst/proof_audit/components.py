"""Pinned component fixture for the proof audit.

Never populate this by scanning ascent/alpha/stack.py or agents/ at runtime -- an audit that
silently drops a component because discovery missed it is worse than no audit. Add a component
by editing this file.

KNOWN INCOMPLETE — this list is not "every component that touches live capital". The
2026-08-14 whole-branch review found `falsifier_trim` (`run_all_agents.py`
`_apply_falsifier_trim`), which submitted real orders off AI PM output and was never in this
fixture, so it has no audit verdict at all -- not even INSUFFICIENT_DATA. It has since been
made advisory-only (CLAUDE.md constraint #5) rather than being retro-fitted into the audit's
scope, because backfilling a component into a finished audit would misrepresent what that
audit actually measured. Adding it here is a deliberate future decision with its own
measurement design; do not assume the absence of a name below means the mechanism does not
exist.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    name: str
    kind: str    # "alpha_sleeve" | "agent" | "subsystem"
    method: str  # "wf_ic" | "counterfactual" | "deferred" | "covered_by_sleeves"


COMPONENTS: list[Component] = [
    # -- Alpha sleeves: pure functions of `features`, re-simulated day by day (Task 4) --
    Component("trend", "alpha_sleeve", "wf_ic"),
    Component("meanrev", "alpha_sleeve", "wf_ic"),
    Component("volatility", "alpha_sleeve", "wf_ic"),
    Component("statarb", "alpha_sleeve", "wf_ic"),
    Component("fundamental", "alpha_sleeve", "wf_ic"),
    Component("earnings", "alpha_sleeve", "wf_ic"),
    Component("analyst", "alpha_sleeve", "wf_ic"),
    Component("options_flow", "alpha_sleeve", "wf_ic"),
    Component("insider", "alpha_sleeve", "wf_ic"),
    Component("short_interest", "alpha_sleeve", "wf_ic"),
    Component("altdata", "alpha_sleeve", "wf_ic"),
    Component("earnings_tone", "alpha_sleeve", "wf_ic"),
    # -- Alpha sleeves excluded from re-simulation: retrained model or LLM-driven --
    Component("ml", "alpha_sleeve", "deferred"),
    Component("llm_fundamental", "alpha_sleeve", "deferred"),
    Component("narrative", "alpha_sleeve", "deferred"),
    # -- Specialist agents --
    # macro_agent / international_agent / alternatives_agent removed 2026-08-23
    # (noise-layer cut): they were dormant, never invoked by run_all_agents.py,
    # and their score_agent path is gone with them.
    Component("us_equities_agent", "agent", "covered_by_sleeves"),
    # -- Named subsystems: scored by counterfactual return delta (Task 6) --
    Component("regime_overlay", "subsystem", "counterfactual"),
    Component("hedge_overlay", "subsystem", "counterfactual"),
    Component("earned_authority", "subsystem", "counterfactual"),
    Component("debate_judge_intervention", "subsystem", "counterfactual"),
]

_BY_NAME = {c.name: c for c in COMPONENTS}


def get_component(name: str) -> Component:
    if name not in _BY_NAME:
        raise KeyError(f"unknown component {name!r}; known: {sorted(_BY_NAME)}")
    return _BY_NAME[name]
