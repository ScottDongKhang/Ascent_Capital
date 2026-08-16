"""The component list is a pinned fixture -- every entry must resolve to a known method."""
import pytest

from ascent.analyst.proof_audit.components import COMPONENTS, get_component

VALID_METHODS = {"wf_ic", "counterfactual", "deferred", "covered_by_sleeves"}

EXPECTED_SLEEVES = {
    "trend", "meanrev", "volatility", "statarb", "fundamental",
    "earnings", "analyst", "options_flow", "insider", "short_interest",
    "altdata", "earnings_tone",
}
EXPECTED_DEFERRED_SLEEVES = {"ml", "llm_fundamental", "narrative"}
EXPECTED_AGENTS = {"macro_agent", "international_agent", "alternatives_agent", "us_equities_agent"}
EXPECTED_SUBSYSTEMS = {
    "regime_overlay", "hedge_overlay", "earned_authority", "debate_judge_intervention",
}


def test_every_component_has_valid_method():
    for c in COMPONENTS:
        assert c.method in VALID_METHODS, f"{c.name} has unknown method {c.method!r}"


def test_names_are_unique():
    names = [c.name for c in COMPONENTS]
    assert len(names) == len(set(names))


def test_expected_sleeves_present_as_wf_ic():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_SLEEVES:
        assert by_name[name].kind == "alpha_sleeve"
        assert by_name[name].method == "wf_ic"


def test_deferred_sleeves_present():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_DEFERRED_SLEEVES:
        assert by_name[name].method == "deferred"


def test_agents_present():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_AGENTS:
        assert by_name[name].kind == "agent"
    assert by_name["us_equities_agent"].method == "covered_by_sleeves"
    for name in EXPECTED_AGENTS - {"us_equities_agent"}:
        assert by_name[name].method == "wf_ic"


def test_subsystems_present_as_counterfactual():
    by_name = {c.name: c for c in COMPONENTS}
    for name in EXPECTED_SUBSYSTEMS:
        assert by_name[name].kind == "subsystem"
        assert by_name[name].method == "counterfactual"


def test_get_component_raises_on_unknown():
    with pytest.raises(KeyError):
        get_component("does_not_exist")


def test_get_component_returns_match():
    c = get_component("trend")
    assert c.name == "trend"
