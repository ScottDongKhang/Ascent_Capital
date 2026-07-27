# tests/portfolio/test_crash_overlay_parity.py
"""
Research and production must read the SAME crash-overlay config.

Precedent: ascent/portfolio/exposure.py exists because research had vol
targeting and production did not, and the two silently diverged.
"""
import inspect

from ascent.config.settings import get_config


def test_both_paths_reference_the_config_flag():
    import ascent.main as main_mod
    from ascent.research.wf_framework import ascent_strategy

    prod = inspect.getsource(main_mod)
    research = inspect.getsource(ascent_strategy)

    assert "momentum_crash_overlay_enabled" in prod, (
        "ascent/main.py must pass the crash-overlay flag through"
    )
    assert "momentum_crash_overlay_enabled" in research, (
        "the WF strategy must read the same flag or research and production "
        "diverge silently"
    )


def test_config_values_are_sane():
    bt = get_config().backtest
    assert 0.0 < bt.momentum_crash_multiplier <= 1.0
