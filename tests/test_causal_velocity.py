# tests/test_causal_velocity.py
import pytest


def test_causal_mechanism_dataclass_fields():
    """CausalMechanism must have all spec-required fields with correct types."""
    from ascent.config.types import CausalMechanism
    m = CausalMechanism(
        symbol="WDC",
        mechanism="NAND oversupply correction → margin expansion → EPS rerating",
        intervention="IF NAND spot +15% from trough THEN WDC gross margin > 40%",
        falsification_condition="IF WDC Q3 gross margin < 38%, thesis broken",
        horizon_days=63,
        timing="catalyst_imminent",
        velocity=0.72,
        mechanism_type="supply_demand_inflection",
        regime_compatible=True,
    )
    assert m.symbol == "WDC"
    assert m.timing in ("priced_in", "not_yet_priced", "catalyst_imminent")
    assert 0.0 <= m.velocity <= 1.0
    assert isinstance(m.horizon_days, int)
    assert isinstance(m.regime_compatible, bool)


def test_causal_mechanism_timing_values():
    """timing must be one of three permitted values."""
    from ascent.config.types import CausalMechanism
    for t in ("priced_in", "not_yet_priced", "catalyst_imminent"):
        m = CausalMechanism(
            symbol="X", mechanism="m", intervention="i",
            falsification_condition="f", horizon_days=21,
            timing=t, velocity=0.5, mechanism_type="momentum_catalyst",
            regime_compatible=True,
        )
        assert m.timing == t


def test_velocity_mid_progress():
    from ascent.causal.velocity import mechanism_velocity_score
    # 12% of 20% needed = 0.60
    assert abs(mechanism_velocity_score(
        current_value=0.12, baseline_value=0.0, threshold_value=0.20
    ) - 0.60) < 1e-9


def test_velocity_clamped_below_zero():
    from ascent.causal.velocity import mechanism_velocity_score
    # current below baseline → clamp to 0.0
    assert mechanism_velocity_score(
        current_value=-5.0, baseline_value=0.0, threshold_value=10.0
    ) == 0.0


def test_velocity_clamped_above_one():
    from ascent.causal.velocity import mechanism_velocity_score
    # current beyond threshold → clamp to 1.0
    assert mechanism_velocity_score(
        current_value=25.0, baseline_value=0.0, threshold_value=20.0
    ) == 1.0


def test_velocity_zero_range_returns_zero():
    from ascent.causal.velocity import mechanism_velocity_score
    # baseline == threshold → undefined, return 0.0
    assert mechanism_velocity_score(
        current_value=5.0, baseline_value=5.0, threshold_value=5.0
    ) == 0.0


def test_velocity_at_exactly_threshold():
    from ascent.causal.velocity import mechanism_velocity_score
    assert mechanism_velocity_score(
        current_value=20.0, baseline_value=0.0, threshold_value=20.0
    ) == 1.0
