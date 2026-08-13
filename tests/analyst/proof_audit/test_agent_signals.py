"""Each wf_ic-method agent in components.py must have a real registered signal function."""
from ascent.analyst.proof_audit.components import COMPONENTS
from ascent.analyst.proof_audit.agent_signals import AGENT_SIGNAL_FUNCS


def test_every_wf_ic_agent_is_registered():
    wf_ic_agents = {c.name for c in COMPONENTS if c.kind == "agent" and c.method == "wf_ic"}
    assert wf_ic_agents == set(AGENT_SIGNAL_FUNCS)


def test_registered_funcs_are_callable():
    for fn in AGENT_SIGNAL_FUNCS.values():
        assert callable(fn)
