# tests/test_debate_agents.py
import inspect
import pytest

_EVIDENCE_RULE_FRAGMENT = "EVIDENCE RULE"


def _get_source(func_name):
    import debate.agents as mod
    func = getattr(mod, func_name)
    return inspect.getsource(func)


def test_bull_system_prompt_contains_evidence_rule():
    src = _get_source("run_bull_agent")
    assert _EVIDENCE_RULE_FRAGMENT in src, \
        "Bull agent system prompt must contain EVIDENCE RULE citation instruction"


def test_bear_system_prompt_contains_evidence_rule():
    src = _get_source("run_bear_agent")
    assert _EVIDENCE_RULE_FRAGMENT in src, \
        "Bear agent system prompt must contain EVIDENCE RULE citation instruction"


def test_devils_advocate_system_prompt_contains_evidence_rule():
    src = _get_source("run_devils_advocate")
    assert _EVIDENCE_RULE_FRAGMENT in src, \
        "Devil's Advocate system prompt must contain EVIDENCE RULE citation instruction"


# Note: run_devils_advocate previously read a "causal_mechanisms" fold-in from
# portfolio_state, sourced from ascent/causal/ (macro DAG discovery + mechanism
# tracking). That package was found to be entirely inert in production — its
# producer (the weekend causal-graph jobs) was structurally unreachable, so
# portfolio_state["causal_mechanisms"] was always empty and this attack block
# never fired on real input. Removed along with the rest of ascent/causal/.
