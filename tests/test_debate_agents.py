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


def test_devils_advocate_references_causal_mechanisms():
    """Devil's advocate source must reference causal mechanism attack capability."""
    src = _get_source("run_devils_advocate")
    assert "causal" in src.lower() or "mechanism" in src.lower(), \
        "Devil's advocate must reference causal mechanism attack in its prompt/code"


def test_devils_advocate_uses_causal_mechanisms_from_portfolio_state():
    """run_devils_advocate must read causal_mechanisms from portfolio_state and include them."""
    import debate.agents as mod
    from unittest.mock import patch, MagicMock

    state = {
        "weights": {"WDC": 0.09},
        "us_regime": "calm_bull",
        "causal_mechanisms": [
            {
                "symbol": "WDC",
                "mechanism": "NAND recovery → margins",
                "timing": "catalyst_imminent",
                "falsification_condition": "IF margin < 38%",
                "mechanism_type": "supply_demand_inflection",
                "regime_compatible": True,
                "velocity": 0.72,
                "horizon_days": 63,
                "intervention": "IF nand +15%",
            }
        ],
    }

    captured_prompts = []

    def mock_tool_completion(system_prompt, **kwargs):
        captured_prompts.append(system_prompt)
        return "Devil's advocate argument."

    def mock_generate_structured(system_prompt, **kwargs):
        captured_prompts.append(system_prompt)
        return "Devil's advocate argument."

    with patch.object(mod, "tool_completion", mock_tool_completion):
        with patch.object(mod, "generate_structured", mock_generate_structured):
            with patch.object(mod, "load_credibility_context", return_value=""):
                with patch.object(mod, "_get_agent_track_record", return_value=""):
                    with patch.object(mod, "_build_agent_context", return_value="weights: WDC=9%"):
                        mod.run_devils_advocate(state)

    assert captured_prompts, "run_devils_advocate must call tool_completion or generate_structured"
    combined = " ".join(captured_prompts)
    assert "WDC" in combined, "causal mechanism symbol must appear in devil's advocate prompt"
    assert "catalyst_imminent" in combined, "causal mechanism timing must appear in prompt"
