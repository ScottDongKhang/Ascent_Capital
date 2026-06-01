# tests/test_causal_compatibility.py
import pytest


def test_momentum_catalyst_compatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("momentum_catalyst", "calm_bull") is True


def test_valuation_incompatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("valuation", "calm_bull") is False


def test_mean_reversion_incompatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("mean_reversion", "calm_bull") is False


def test_supply_demand_compatible_calm_bull():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("supply_demand_inflection", "calm_bull") is True


def test_quality_defensive_compatible_stressed():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("quality_defensive", "stressed") is True


def test_momentum_catalyst_incompatible_crisis():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("momentum_catalyst", "crisis") is False


def test_macro_hedge_compatible_crisis():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("macro_hedge", "crisis") is True


def test_macro_hedge_compatible_all_regimes():
    from ascent.causal.compatibility import regime_compatible
    for regime in ("calm_bull", "stressed", "crisis", "neutral", "uncertain"):
        assert regime_compatible("macro_hedge", regime) is True, \
            f"macro_hedge should be compatible with {regime}"


def test_unknown_regime_defaults_to_conservative():
    from ascent.causal.compatibility import regime_compatible
    assert regime_compatible("quality_defensive", "euphoric") is True
    assert regime_compatible("momentum_catalyst", "euphoric") is False


def test_filter_mechanisms_returns_compatible_only():
    from ascent.causal.compatibility import filter_mechanisms

    mechanisms = [
        {"mechanism_type": "momentum_catalyst", "timing": "catalyst_imminent"},
        {"mechanism_type": "valuation",          "timing": "not_yet_priced"},
        {"mechanism_type": "supply_demand_inflection", "timing": "not_yet_priced"},
        {"mechanism_type": "mean_reversion",     "timing": "priced_in"},
    ]
    compatible = filter_mechanisms(mechanisms, regime="calm_bull")
    types = [m["mechanism_type"] for m in compatible]
    assert "valuation" not in types
    assert "mean_reversion" not in types
    assert "momentum_catalyst" in types
    assert "supply_demand_inflection" in types
