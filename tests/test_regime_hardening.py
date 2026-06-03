# tests/test_regime_hardening.py
import pytest
import numpy as np
import pandas as pd


def test_regime_config_has_asymmetric_threshold_keys():
    from ascent.regime.types import REGIME_CONFIG_DEFAULTS
    assert "regime_downgrade_threshold" in REGIME_CONFIG_DEFAULTS, \
        "regime_downgrade_threshold must be in REGIME_CONFIG_DEFAULTS"
    assert "regime_upgrade_threshold" in REGIME_CONFIG_DEFAULTS, \
        "regime_upgrade_threshold must be in REGIME_CONFIG_DEFAULTS"
    assert REGIME_CONFIG_DEFAULTS["regime_downgrade_threshold"] == 0.40
    assert REGIME_CONFIG_DEFAULTS["regime_upgrade_threshold"] == 0.70


def _make_machine(severity=None, downgrade_threshold=0.40, upgrade_threshold=0.70,
                  enter_threshold=0.55):
    """Helper: build a _HysteresisStateMachine starting in state 0 (calm_bull).
    entropy_uncertain_threshold=1.01 disables the entropy gate so tests can focus
    solely on threshold behavior without K=2 near-50/50 probs triggering uncertain mode.
    """
    from ascent.regime.decision import _HysteresisStateMachine
    return _HysteresisStateMachine(
        initial_regime=0,
        enter_threshold=enter_threshold,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=1.01,  # disabled — not what we're testing here
        severity={0: 0, 1: 2},   # state 0 = calm_bull severity 0, state 1 = stressed severity 2
        downgrade_threshold=downgrade_threshold,
        upgrade_threshold=upgrade_threshold,
    )


def test_asymmetric_hysteresis_downgrade_fires_at_low_prob():
    """
    Transition to worse regime (0→1) fires at prob=0.53 (> downgrade_threshold 0.40).
    With the old symmetric enter_threshold=0.55 this would NOT have fired.
    """
    machine = _make_machine()
    probs = np.array([0.47, 0.53])   # state 1 dominant at 0.53, above 0.40 but below old 0.55
    state, _, _, _ = machine.step(probs)
    assert state == 1, f"Expected transition to state 1 (stressed), got {state}"


def test_asymmetric_hysteresis_upgrade_blocked_at_mid_prob():
    """Transition to better regime (1→0) should NOT fire at prob=0.65 (< upgrade_threshold 0.70)."""
    from ascent.regime.decision import _HysteresisStateMachine
    machine = _HysteresisStateMachine(
        initial_regime=1,
        enter_threshold=0.55,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=1.01,  # disabled
        severity={0: 0, 1: 2},
        downgrade_threshold=0.40,
        upgrade_threshold=0.70,
    )
    probs = np.array([0.65, 0.35])  # state 0 dominant at 0.65 — below upgrade_threshold
    state, _, _, _ = machine.step(probs)
    assert state == 1, f"Upgrade should be blocked at 0.65 < 0.70, but got state {state}"


def test_asymmetric_hysteresis_upgrade_fires_above_threshold():
    """Transition to better regime (1→0) should fire at prob=0.75 (> upgrade_threshold 0.70)."""
    from ascent.regime.decision import _HysteresisStateMachine
    machine = _HysteresisStateMachine(
        initial_regime=1,
        enter_threshold=0.55,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=1.01,  # disabled
        severity={0: 0, 1: 2},
        downgrade_threshold=0.40,
        upgrade_threshold=0.70,
    )
    probs = np.array([0.75, 0.25])
    state, _, _, _ = machine.step(probs)
    assert state == 0, f"Upgrade should fire at 0.75 > 0.70, but got state {state}"


def test_asymmetric_hysteresis_unknown_severity_uses_enter_threshold():
    """When severity dict doesn't include a state, fall back to enter_threshold."""
    from ascent.regime.decision import _HysteresisStateMachine
    machine = _HysteresisStateMachine(
        initial_regime=0,
        enter_threshold=0.55,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=1.01,  # disabled
        severity={},
        downgrade_threshold=0.40,
        upgrade_threshold=0.70,
    )
    probs = np.array([0.45, 0.55])  # state 1 at 0.55, just above fallback enter_threshold
    state, _, _, _ = machine.step(probs)
    assert state == 1, "With unknown severity, should fall back to enter_threshold=0.55"


def test_decision_engine_passes_severity_to_machine():
    """
    RegimeDecisionEngine.process() must produce asymmetric behaviour end-to-end:
    a downgrade transition fires at prob=0.53 (> downgrade_threshold 0.40,
    below old symmetric enter_threshold 0.55).
    """
    from ascent.regime.decision import RegimeDecisionEngine
    from ascent.regime.types import RegimeLabel

    state_labels = {0: RegimeLabel.CALM_BULL, 1: RegimeLabel.STRESSED}
    engine = RegimeDecisionEngine(
        state_labels=state_labels,
        enter_threshold=0.55,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=1.01,  # disabled so near-50/50 probs don't trigger uncertain
        downgrade_threshold=0.40,
        upgrade_threshold=0.70,
    )

    idx = pd.bdate_range("2026-01-02", periods=5)
    # First 2 days: calm_bull firmly established. Days 3-5: stressed dominant at 0.53.
    rows = (
        [[0.90, 0.10]] * 2
        + [[0.47, 0.53]] * 3  # stressed dominant at 0.53, above downgrade_threshold 0.40
    )
    prob_df = pd.DataFrame(rows, index=idx, columns=[0, 1])

    signals = engine.process(prob_df)
    labels = [s.label.value for s in signals]
    assert "stressed" in labels, \
        f"Stressed should appear (downgrade fires at 0.53 > 0.40), got: {labels}"


def test_entropy_penalty_reduces_risk_multiplier_when_frozen():
    """Entropy below 1e-6 must reduce risk_multiplier by 0.90×."""
    from ascent.regime.decision import RegimeDecisionEngine
    from ascent.regime.types import RegimeLabel

    state_labels = {0: RegimeLabel.STRESSED}
    engine = RegimeDecisionEngine(state_labels=state_labels, min_dwell_days=1)

    idx = pd.bdate_range("2026-01-02", periods=1)
    # All probability in state 0 — entropy = 0.0 (< 1e-6)
    prob_df = pd.DataFrame([[1.0]], index=idx, columns=[0])

    signals = engine.process(prob_df)
    assert len(signals) == 1
    # stressed normal risk_mult = 0.65; with penalty = 0.65 * 0.90 = 0.585
    expected = round(0.65 * 0.90, 6)
    assert abs(signals[0].risk_multiplier - expected) < 1e-5, \
        f"Expected risk_multiplier ≈ {expected}, got {signals[0].risk_multiplier}"


def test_entropy_penalty_not_applied_at_normal_entropy():
    """Entropy above 1e-6 must leave risk_multiplier unchanged."""
    from ascent.regime.decision import RegimeDecisionEngine
    from ascent.regime.types import RegimeLabel

    state_labels = {0: RegimeLabel.STRESSED, 1: RegimeLabel.CALM_BULL}
    engine = RegimeDecisionEngine(state_labels=state_labels, min_dwell_days=1)

    idx = pd.bdate_range("2026-01-02", periods=1)
    prob_df = pd.DataFrame([[0.70, 0.30]], index=idx, columns=[0, 1])

    signals = engine.process(prob_df)
    assert len(signals) == 1
    # stressed normal risk_mult = 0.65; entropy is not frozen, no penalty
    assert abs(signals[0].risk_multiplier - 0.65) < 1e-5, \
        f"Expected risk_multiplier = 0.65 (no penalty), got {signals[0].risk_multiplier}"
