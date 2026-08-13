"""Pinned component fixture for the proof audit.

Never populate this by scanning ascent/alpha/stack.py or agents/ at runtime -- an audit that
silently drops a component because discovery missed it is worse than no audit. Add a component
by editing this file.
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
    Component("us_equities_agent", "agent", "covered_by_sleeves"),
    Component("macro_agent", "agent", "wf_ic"),
    Component("international_agent", "agent", "wf_ic"),
    Component("alternatives_agent", "agent", "wf_ic"),
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
