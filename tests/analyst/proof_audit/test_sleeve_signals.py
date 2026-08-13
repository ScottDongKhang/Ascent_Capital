"""Every wf_ic-method sleeve in components.py must have a real registered signal function."""
from ascent.analyst.proof_audit.components import COMPONENTS
from ascent.analyst.proof_audit.sleeve_signals import SLEEVE_SIGNAL_FUNCS


def test_every_wf_ic_sleeve_is_registered():
    wf_ic_sleeves = {
        c.name for c in COMPONENTS if c.kind == "alpha_sleeve" and c.method == "wf_ic"
    }
    assert wf_ic_sleeves.issubset(set(SLEEVE_SIGNAL_FUNCS))


def test_registered_funcs_are_callable():
    for fn in SLEEVE_SIGNAL_FUNCS.values():
        assert callable(fn)
