import pytest
import numpy as np
import pandas as pd


def _make_regime(label: str, confidence: float = 0.80, entropy: float = 0.30):
    from ascent.regime.types import RegimeSignal, RegimeLabel
    probs = np.zeros(3)
    probs[0] = confidence
    probs[1] = (1 - confidence) / 2
    probs[2] = (1 - confidence) / 2
    return RegimeSignal(
        date=pd.Timestamp("2026-05-01"),
        probs=probs,
        label=RegimeLabel.from_str(label),
        entropy=entropy,
        transition_flag=False,
        risk_multiplier=1.0,
        sleeve_adjustments={},
        dwell_days=5,
    )


def _make_weights():
    raw = {
        "AAPL": 0.10, "MSFT": 0.09, "JPM": 0.08, "XOM": 0.07,
        "NEE": 0.06, "MRK": 0.06, "WMT": 0.05, "CAT": 0.05,
        "EQIX": 0.05, "AMZN": 0.07, "EEM": 0.06, "GLD": 0.05,
        "TLT": 0.06, "HYG": 0.05, "VNQ": 0.05,
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def test_calm_bull_no_hedge():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    hedged, meta = apply_hedge_overlay(weights, _make_regime("calm_bull"))
    assert "VIXY" not in hedged or hedged.get("VIXY", 0) < 0.001
    assert abs(sum(hedged.values()) - 1.0) < 1e-6
    assert meta["hedge_weight"] == 0.0


def test_crisis_adds_vixy():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    hedged, meta = apply_hedge_overlay(weights, _make_regime("crisis", confidence=0.90))
    assert hedged["VIXY"] > 0.05, "Crisis regime must add meaningful VIXY"
    assert abs(sum(hedged.values()) - 1.0) < 1e-6, "Weights must still sum to 1.0"


def test_stressed_adds_smaller_vixy_than_crisis():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    crisis_hedged, _   = apply_hedge_overlay(weights, _make_regime("crisis", confidence=0.85))
    stressed_hedged, _ = apply_hedge_overlay(weights, _make_regime("stressed", confidence=0.85))
    assert stressed_hedged["VIXY"] < crisis_hedged["VIXY"]


def test_weights_sum_to_one_in_all_regimes():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    for label in ["calm_bull", "stressed", "crisis", "euphoric", "uncertain"]:
        hedged, _ = apply_hedge_overlay(weights, _make_regime(label))
        total = sum(hedged.values())
        assert abs(total - 1.0) < 1e-6, f"Weights don't sum to 1.0 for regime={label}, got {total}"


def test_existing_vixy_replaced_not_doubled():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    weights["VIXY"] = 0.04
    total_before = sum(weights.values())
    weights = {k: v / total_before for k, v in weights.items()}

    hedged, meta = apply_hedge_overlay(weights, _make_regime("crisis", confidence=0.90))
    assert hedged["VIXY"] < 0.15, "Should not double-count existing VIXY allocation"
    assert abs(sum(hedged.values()) - 1.0) < 1e-6


def test_no_regime_signal_returns_unchanged():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    hedged, meta = apply_hedge_overlay(weights, regime_signal=None)
    assert hedged == weights
    assert meta["hedge_weight"] == 0.0


def test_confidence_scales_hedge_weight():
    from ascent.portfolio.hedge_overlay import compute_hedge_weight
    from ascent.regime.types import RegimeLabel
    low_conf  = compute_hedge_weight(RegimeLabel.CRISIS, confidence=0.55)
    high_conf = compute_hedge_weight(RegimeLabel.CRISIS, confidence=0.95)
    assert high_conf > low_conf, "Higher confidence should produce larger hedge"


def test_no_position_exceeds_original_weight_after_hedge():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    hedged, _ = apply_hedge_overlay(weights, _make_regime("crisis", confidence=0.90))
    for sym, w in hedged.items():
        if sym != "VIXY":
            assert w <= weights.get(sym, 0) + 1e-6, f"{sym} weight increased after hedge overlay"


def test_metadata_contains_required_keys():
    from ascent.portfolio.hedge_overlay import apply_hedge_overlay
    weights = _make_weights()
    _, meta = apply_hedge_overlay(weights, _make_regime("stressed"))
    for key in ["hedge_weight", "regime_label", "confidence", "vixy_before", "vixy_after"]:
        assert key in meta, f"Metadata missing key: {key}"
