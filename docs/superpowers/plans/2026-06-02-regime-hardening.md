# Regime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden regime detection with three targeted changes: a rule-based crisis override (A), asymmetric hysteresis thresholds (C), and an entropy overconfidence penalty (D).

**Architecture:** All changes are confined to the regime subsystem. A is a post-processing pass on the signal DataFrame in `engine.py`. C modifies the hysteresis state machine in `decision.py` to use different entry thresholds depending on whether a transition worsens or improves the regime. D applies a small risk multiplier reduction in `decision.py` whenever Shannon entropy is pathologically low.

**Tech Stack:** Python, pandas, numpy, pytest. No new dependencies.

---

## File map

| File | Role |
|------|------|
| `ascent/regime/types.py` | Add two new config keys to `REGIME_CONFIG_DEFAULTS` |
| `ascent/regime/decision.py` | Asymmetric hysteresis (C) + entropy penalty (D) |
| `ascent/regime/engine.py` | Hard crisis override (A) |
| `tests/test_regime_hardening.py` | All new tests (create) |

---

## Task 1: Add config keys for asymmetric hysteresis (types.py)

**Files:**
- Modify: `ascent/regime/types.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/scott/Downloads/ascent\ capital\ v2\ up\ to\ phase\ 5.1
.venv/bin/python -m pytest tests/test_regime_hardening.py::test_regime_config_has_asymmetric_threshold_keys -v
```
Expected: FAIL — `AssertionError: regime_downgrade_threshold must be in REGIME_CONFIG_DEFAULTS`

- [ ] **Step 3: Add keys to REGIME_CONFIG_DEFAULTS in types.py**

In `ascent/regime/types.py`, inside `REGIME_CONFIG_DEFAULTS`, add after the existing `"regime_enter_threshold"` and `"regime_exit_threshold"` lines:

```python
    # Asymmetric hysteresis — downgrade (→ worse regime) is faster to trigger
    # than upgrade (→ better regime). See decision.py _HysteresisStateMachine.
    "regime_downgrade_threshold": 0.40,   # prob needed to enter a worse regime
    "regime_upgrade_threshold":   0.70,   # prob needed to enter a better regime
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py::test_regime_config_has_asymmetric_threshold_keys -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ascent/regime/types.py tests/test_regime_hardening.py
git commit -m "feat: add regime_downgrade_threshold and regime_upgrade_threshold config keys"
```

---

## Task 2: Asymmetric hysteresis in _HysteresisStateMachine (decision.py — part 1)

**Files:**
- Modify: `ascent/regime/decision.py`
- Test: `tests/test_regime_hardening.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_regime_hardening.py`:

```python
def _make_machine(severity=None, downgrade_threshold=0.40, upgrade_threshold=0.70,
                  enter_threshold=0.55):
    """Helper: build a _HysteresisStateMachine starting in state 0 (calm_bull)."""
    from ascent.regime.decision import _HysteresisStateMachine
    return _HysteresisStateMachine(
        initial_regime=0,
        enter_threshold=enter_threshold,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=0.90,
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
        initial_regime=1,          # start in stressed
        enter_threshold=0.55,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=0.90,
        severity={0: 0, 1: 2},    # state 0 = calm_bull severity 0, state 1 = stressed severity 2
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
        entropy_uncertain_threshold=0.90,
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
        entropy_uncertain_threshold=0.90,
        severity={},           # empty — unknown severity
        downgrade_threshold=0.40,
        upgrade_threshold=0.70,
    )
    probs = np.array([0.45, 0.55])  # state 1 at 0.55, just above fallback enter_threshold
    state, _, _, _ = machine.step(probs)
    assert state == 1, "With unknown severity, should fall back to enter_threshold=0.55"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -k "asymmetric" -v
```
Expected: FAIL — `TypeError: _HysteresisStateMachine.__init__() got unexpected keyword argument 'severity'`

- [ ] **Step 3: Modify _HysteresisStateMachine in decision.py**

Replace the `__init__` signature and add `_severity`, `_downgrade_threshold`, `_upgrade_threshold`:

```python
def __init__(
    self,
    initial_regime: int,
    enter_threshold: float = 0.55,
    exit_threshold: float = 0.35,
    min_dwell_days: int = 3,
    entropy_uncertain_threshold: float = 0.90,
    severity: Optional[dict] = None,
    downgrade_threshold: Optional[float] = None,
    upgrade_threshold: Optional[float] = None,
):
    self.enter_threshold = enter_threshold
    self.exit_threshold = exit_threshold
    self.min_dwell_days = min_dwell_days
    self.entropy_threshold = entropy_uncertain_threshold
    self._severity = severity or {}
    self._downgrade_threshold = downgrade_threshold if downgrade_threshold is not None else enter_threshold
    self._upgrade_threshold = upgrade_threshold if upgrade_threshold is not None else enter_threshold

    self._current: int = initial_regime
    self._dwell: int = 0
    self._candidate: Optional[int] = None
    self._candidate_streak: int = 0
```

Then in `step()`, replace the single threshold check:

```python
# was: if dominant != self._current and dominant_prob >= self.enter_threshold:
# becomes:
if dominant != self._current:
    current_sev = self._severity.get(self._current, -1)
    candidate_sev = self._severity.get(dominant, -1)
    if current_sev == -1 or candidate_sev == -1:
        threshold = self.enter_threshold           # unknown severity → fallback
    elif candidate_sev > current_sev:
        threshold = self._downgrade_threshold      # transitioning to worse regime
    else:
        threshold = self._upgrade_threshold        # transitioning to better regime

    if dominant_prob >= threshold:
        if self._candidate == dominant:
            self._candidate_streak += 1
        else:
            self._candidate = dominant
            self._candidate_streak = 1

        if self._candidate_streak >= self.min_dwell_days:
            self._current = dominant
            self._dwell = 1
            self._candidate = None
            self._candidate_streak = 0
            transition_flag = True
    else:
        self._candidate = None
        self._candidate_streak = 0
```

Note: remove the old `else: reset candidate` block that was at the same level — it's now inside the `if dominant != self._current:` block above.

The full revised `step()` after the change:

```python
def step(self, probs: np.ndarray) -> tuple:
    entropy = _entropy(probs)
    dominant = int(np.argmax(probs))
    dominant_prob = float(probs[dominant])

    is_uncertain = entropy >= self.entropy_threshold

    self._dwell += 1
    transition_flag = False

    if is_uncertain:
        self._candidate = None
        self._candidate_streak = 0
        return self._current, transition_flag, self._dwell, True

    if dominant != self._current:
        current_sev = self._severity.get(self._current, -1)
        candidate_sev = self._severity.get(dominant, -1)
        if current_sev == -1 or candidate_sev == -1:
            threshold = self.enter_threshold
        elif candidate_sev > current_sev:
            threshold = self._downgrade_threshold
        else:
            threshold = self._upgrade_threshold

        if dominant_prob >= threshold:
            if self._candidate == dominant:
                self._candidate_streak += 1
            else:
                self._candidate = dominant
                self._candidate_streak = 1

            if self._candidate_streak >= self.min_dwell_days:
                self._current = dominant
                self._dwell = 1
                self._candidate = None
                self._candidate_streak = 0
                transition_flag = True
        else:
            self._candidate = None
            self._candidate_streak = 0

    return self._current, transition_flag, self._dwell, False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -k "asymmetric" -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add ascent/regime/decision.py tests/test_regime_hardening.py
git commit -m "feat: asymmetric hysteresis — downgrade 0.40, upgrade 0.70"
```

---

## Task 3: Wire severity dict in RegimeDecisionEngine (decision.py — part 2)

The `RegimeDecisionEngine` instantiates `_HysteresisStateMachine`. It must now build a severity dict from `state_labels` and pass the new thresholds.

**Files:**
- Modify: `ascent/regime/decision.py`
- Test: `tests/test_regime_hardening.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_regime_hardening.py`:

```python
def test_decision_engine_passes_severity_to_machine():
    """
    RegimeDecisionEngine.process() must produce asymmetric behaviour end-to-end:
    a downgrade transition should commit at prob=0.50 (> downgrade_threshold 0.40).
    """
    from ascent.regime.decision import RegimeDecisionEngine
    from ascent.regime.types import RegimeLabel

    # Two states: state 0 = calm_bull, state 1 = stressed
    state_labels = {0: RegimeLabel.CALM_BULL, 1: RegimeLabel.STRESSED}
    engine = RegimeDecisionEngine(
        state_labels=state_labels,
        enter_threshold=0.55,
        exit_threshold=0.35,
        min_dwell_days=1,
        entropy_uncertain_threshold=0.90,
        downgrade_threshold=0.40,
        upgrade_threshold=0.70,
    )

    idx = pd.bdate_range("2026-01-02", periods=5)
    # First 2 days: calm_bull dominant. Days 3-5: stressed at 0.50.
    rows = (
        [[0.90, 0.10]] * 2   # calm_bull firmly established
        + [[0.50, 0.50]] * 3  # stressed dominant at 0.50
    )
    prob_df = pd.DataFrame(rows, index=idx, columns=[0, 1])

    signals = engine.process(prob_df)
    # After 1 dwell day of stressed≥0.50, machine should have transitioned
    labels = [s.label.value for s in signals]
    assert "stressed" in labels, \
        f"Stressed should appear (downgrade fires at 0.50 > 0.40), got: {labels}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py::test_decision_engine_passes_severity_to_machine -v
```
Expected: FAIL — `TypeError: RegimeDecisionEngine.__init__() got unexpected keyword argument 'downgrade_threshold'`

- [ ] **Step 3: Modify RegimeDecisionEngine in decision.py**

Add `downgrade_threshold` and `upgrade_threshold` to `__init__`, and add the `_REGIME_SEVERITY` constant and severity-dict-building logic.

At module level, add below the existing imports:

```python
# Severity ordering for asymmetric hysteresis.
# Higher = more defensive / risk-off. Used to decide which threshold to apply.
_REGIME_SEVERITY: dict[str, int] = {
    "calm_bull": 0,
    "euphoric":  0,
    "uncertain": 1,
    "stressed":  2,
    "crisis":    3,
}
```

Update `RegimeDecisionEngine.__init__` signature (add two new params after `entropy_uncertain_threshold`):

```python
def __init__(
    self,
    state_labels: Dict[int, RegimeLabel],
    enter_threshold: float = 0.55,
    exit_threshold: float = 0.35,
    min_dwell_days: int = 3,
    entropy_uncertain_threshold: float = 0.90,
    downgrade_threshold: float = 0.40,
    upgrade_threshold: float = 0.70,
    risk_multipliers: Optional[Dict[str, float]] = None,
    sleeve_adjustments: Optional[Dict[str, Dict[str, float]]] = None,
):
    self.state_labels = state_labels
    self.enter_threshold = enter_threshold
    self.exit_threshold = exit_threshold
    self.min_dwell_days = min_dwell_days
    self.entropy_threshold = entropy_uncertain_threshold
    self.downgrade_threshold = downgrade_threshold
    self.upgrade_threshold = upgrade_threshold
    # ... rest of existing __init__ unchanged ...
```

In `process()`, replace the `machine = _HysteresisStateMachine(...)` block with:

```python
# Build severity dict: {state_idx: severity_int}
severity = {
    idx: _REGIME_SEVERITY.get(lbl.value if hasattr(lbl, "value") else str(lbl), -1)
    for idx, lbl in self.state_labels.items()
}

first_probs = prob_df.iloc[0].values
initial_state = int(np.argmax(first_probs))
machine = _HysteresisStateMachine(
    initial_regime=initial_state,
    enter_threshold=self.enter_threshold,
    exit_threshold=self.exit_threshold,
    min_dwell_days=self.min_dwell_days,
    entropy_uncertain_threshold=self.entropy_threshold,
    severity=severity,
    downgrade_threshold=self.downgrade_threshold,
    upgrade_threshold=self.upgrade_threshold,
)
```

- [ ] **Step 4: Wire the new config keys in engine.py**

In `engine.py`, where `RegimeDecisionEngine` is instantiated (around line 335), add the two new kwargs:

```python
self._decision_engine = RegimeDecisionEngine(
    state_labels=self._model.state_labels if self._model else {},
    enter_threshold=self._cfg["regime_enter_threshold"],
    exit_threshold=self._cfg["regime_exit_threshold"],
    min_dwell_days=self._cfg["regime_min_dwell_days"],
    entropy_uncertain_threshold=self._cfg["regime_entropy_uncertain_threshold"],
    downgrade_threshold=self._cfg.get("regime_downgrade_threshold", 0.40),
    upgrade_threshold=self._cfg.get("regime_upgrade_threshold", 0.70),
    risk_multipliers=self._cfg["regime_risk_multiplier"],
    sleeve_adjustments=self._cfg["regime_sleeve_adjustments"],
)
```

- [ ] **Step 5: Run all regime hardening tests so far**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -v
```
Expected: all tests pass (config + 4 asymmetric + decision_engine_passes_severity)

- [ ] **Step 6: Commit**

```bash
git add ascent/regime/decision.py ascent/regime/engine.py tests/test_regime_hardening.py
git commit -m "feat: wire asymmetric hysteresis into RegimeDecisionEngine and engine config"
```

---

## Task 4: Entropy overconfidence penalty (decision.py — part 3)

**Files:**
- Modify: `ascent/regime/decision.py`
- Test: `tests/test_regime_hardening.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_regime_hardening.py`:

```python
def test_entropy_penalty_reduces_risk_multiplier_when_frozen():
    """Entropy below 1e-6 must reduce risk_multiplier by 0.90×."""
    from ascent.regime.decision import RegimeDecisionEngine
    from ascent.regime.types import RegimeLabel

    state_labels = {0: RegimeLabel.STRESSED}
    engine = RegimeDecisionEngine(state_labels=state_labels, min_dwell_days=1)

    idx = pd.bdate_range("2026-01-02", periods=1)
    # All probability in state 0 — entropy will be exactly 0.0 (< 1e-6)
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -k "entropy_penalty" -v
```
Expected: FAIL (risk_multiplier is 0.65 in both, penalty not yet applied)

- [ ] **Step 3: Add constants and penalty logic to decision.py**

At module level in `decision.py`, add two constants below the existing log line:

```python
ENTROPY_OVERCONFIDENCE_THRESHOLD = 1e-6   # below this, entropy is model lockup
ENTROPY_OVERCONFIDENCE_PENALTY   = 0.90   # risk_multiplier reduction when frozen
```

In `RegimeDecisionEngine.process()`, find the block that appends to `signals`:

```python
            signals.append(RegimeSignal(
                date=date,
                probs=probs,
                label=label,
                entropy=entropy,
                transition_flag=transition_flag,
                risk_multiplier=risk_mult,
                sleeve_adjustments=sleeve_adj,
                dwell_days=dwell_days,
            ))
```

Replace it with:

```python
            if entropy < ENTROPY_OVERCONFIDENCE_THRESHOLD:
                log.warning(
                    "regime.decision: entropy=%.2e on %s — model overconfidence, "
                    "applying %.0f%% risk penalty",
                    entropy, date.date(), (1 - ENTROPY_OVERCONFIDENCE_PENALTY) * 100,
                )
                risk_mult = round(risk_mult * ENTROPY_OVERCONFIDENCE_PENALTY, 6)

            signals.append(RegimeSignal(
                date=date,
                probs=probs,
                label=label,
                entropy=entropy,
                transition_flag=transition_flag,
                risk_multiplier=risk_mult,
                sleeve_adjustments=sleeve_adj,
                dwell_days=dwell_days,
            ))
```

- [ ] **Step 4: Run entropy tests**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -k "entropy_penalty" -v
```
Expected: 2 PASS

- [ ] **Step 5: Run all hardening tests**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -v
```
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add ascent/regime/decision.py tests/test_regime_hardening.py
git commit -m "feat: entropy overconfidence penalty — risk_mult × 0.90 when entropy < 1e-6"
```

---

## Task 5: Hard crisis override (engine.py)

**Files:**
- Modify: `ascent/regime/engine.py`
- Test: `tests/test_regime_hardening.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_regime_hardening.py`:

```python
def _make_signal_df(n=10, label="stressed", risk_multiplier=0.65):
    """Build a minimal signal DataFrame as produced by RegimeDecisionEngine."""
    idx = pd.bdate_range("2026-04-01", periods=n)
    return pd.DataFrame({
        "label": label,
        "risk_multiplier": risk_multiplier,
        "confidence": 1.0,
        "entropy": 6e-11,
        "transition_flag": False,
        "dwell_days": range(1, n + 1),
        "crisis_override": False,
    }, index=idx)


def _make_spy(n=10, crash_5d=-0.10, base=500.0):
    """SPY with a 5-day cumulative return of crash_5d at the end."""
    idx = pd.bdate_range("2026-03-20", periods=n + 5)
    # Build prices so last 5 returns total crash_5d
    step = (1 + crash_5d) ** (1 / 5) - 1
    daily = [0.0] * (len(idx) - 5) + [step] * 5
    px = pd.Series(base, index=idx)
    for i in range(1, len(idx)):
        px.iloc[i] = px.iloc[i - 1] * (1 + daily[i])
    return px


def _make_vix(n=15, level=35.0):
    idx = pd.bdate_range("2026-03-20", periods=n)
    return pd.Series(level, index=idx)


def test_crisis_override_fires_when_vix_high_and_spy_down():
    """Crisis override must flip stressed→crisis when VIX>30 AND SPY 5d < -7%."""
    from ascent.regime.engine import _apply_crisis_override

    signal_df = _make_signal_df(n=10, label="stressed", risk_multiplier=0.65)
    spy = _make_spy(n=10, crash_5d=-0.10)   # -10% over 5 days
    vix = _make_vix(n=15, level=35.0)        # VIX = 35 (> 30)

    result = _apply_crisis_override(signal_df, spy, vix)

    overridden = result[result["crisis_override"] == True]
    assert len(overridden) > 0, "Expected at least one crisis_override=True row"
    assert all(overridden["label"] == "crisis"), "Overridden rows must be labeled 'crisis'"
    assert all(overridden["risk_multiplier"] == 0.40), "Overridden rows must have risk_multiplier=0.40"


def test_crisis_override_does_not_fire_with_low_vix():
    """No override when VIX < 30, even with large SPY drawdown."""
    from ascent.regime.engine import _apply_crisis_override

    signal_df = _make_signal_df(n=10, label="stressed")
    spy = _make_spy(n=10, crash_5d=-0.10)
    vix = _make_vix(n=15, level=22.0)        # VIX = 22 (< 30)

    result = _apply_crisis_override(signal_df, spy, vix)
    assert result["crisis_override"].sum() == 0, "No override when VIX < 30"


def test_crisis_override_does_not_fire_with_small_drawdown():
    """No override when SPY 5d > -7%, even with high VIX."""
    from ascent.regime.engine import _apply_crisis_override

    signal_df = _make_signal_df(n=10, label="stressed")
    spy = _make_spy(n=10, crash_5d=-0.04)   # only -4%
    vix = _make_vix(n=15, level=35.0)

    result = _apply_crisis_override(signal_df, spy, vix)
    assert result["crisis_override"].sum() == 0, "No override when drawdown < 7%"


def test_crisis_override_preserves_existing_crisis_label():
    """Rows already labeled 'crisis' (HMM-native) keep crisis_override=False."""
    from ascent.regime.engine import _apply_crisis_override

    signal_df = _make_signal_df(n=10, label="crisis", risk_multiplier=0.40)
    spy = _make_spy(n=10, crash_5d=-0.10)
    vix = _make_vix(n=15, level=35.0)

    result = _apply_crisis_override(signal_df, spy, vix)
    # crisis_override should be False — it was already crisis
    assert result["crisis_override"].sum() == 0, \
        "crisis_override flag must be False for HMM-native crisis rows"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -k "crisis_override" -v
```
Expected: FAIL — `ImportError: cannot import name '_apply_crisis_override' from 'ascent.regime.engine'`

- [ ] **Step 3: Add constants and function to engine.py**

In `ascent/regime/engine.py`, add these constants near the top (after the existing `EMERGENCY_*` constants):

```python
CRISIS_VIX_THRESHOLD   = 30.0    # VIX must exceed this for crisis override
CRISIS_SPY_5D_THRESHOLD = -0.07  # SPY 5-day return must be below this
```

Then add the function after `_apply_vix_confirmation()`:

```python
def _apply_crisis_override(
    signal_df: pd.DataFrame,
    spy_prices: pd.Series,
    vix_prices: pd.Series,
) -> pd.DataFrame:
    """
    Post-process signal cache: force label='crisis' and risk_multiplier=0.40
    on any day where VIX > CRISIS_VIX_THRESHOLD AND SPY 5-day return < CRISIS_SPY_5D_THRESHOLD.

    Adds a boolean column 'crisis_override' — True only on rule-overridden rows
    (not on HMM-native crisis days). This allows downstream auditing.
    """
    if signal_df.empty or spy_prices is None or vix_prices is None:
        if "crisis_override" not in signal_df.columns:
            signal_df = signal_df.copy()
            signal_df["crisis_override"] = False
        return signal_df

    df = signal_df.copy()
    if "crisis_override" not in df.columns:
        df["crisis_override"] = False

    spy_ret_5d = spy_prices.pct_change(5).reindex(df.index, method="ffill")
    vix_aligned = vix_prices.reindex(df.index, method="ffill")

    trigger_mask = (
        (vix_aligned.fillna(0) > CRISIS_VIX_THRESHOLD)
        & (spy_ret_5d.fillna(0) < CRISIS_SPY_5D_THRESHOLD)
        & (df["label"] != "crisis")   # don't double-flag HMM-native crisis
    )

    n_triggered = int(trigger_mask.sum())
    if n_triggered > 0:
        df.loc[trigger_mask, "label"] = "crisis"
        df.loc[trigger_mask, "risk_multiplier"] = 0.40
        df.loc[trigger_mask, "crisis_override"] = True
        log.warning(
            "regime.engine: crisis override triggered on %d days "
            "(VIX > %.0f AND SPY 5d < %.0f%%)",
            n_triggered, CRISIS_VIX_THRESHOLD, CRISIS_SPY_5D_THRESHOLD * 100,
        )

    return df
```

- [ ] **Step 4: Call _apply_crisis_override in engine.fit()**

In `engine.fit()`, find the two-line block:

```python
        self._signal_cache = self._decision_engine.process_to_frame(prob_df)
        self._signal_cache = _apply_vix_confirmation(self._signal_cache, vix_prices)
```

Add the crisis override call immediately after:

```python
        self._signal_cache = self._decision_engine.process_to_frame(prob_df)
        self._signal_cache = _apply_vix_confirmation(self._signal_cache, vix_prices)
        self._signal_cache = _apply_crisis_override(
            self._signal_cache, spy_prices, vix_prices
        )
```

- [ ] **Step 5: Run crisis override tests**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -k "crisis_override" -v
```
Expected: 4 PASS

- [ ] **Step 6: Run all hardening tests**

```bash
.venv/bin/python -m pytest tests/test_regime_hardening.py -v
```
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add ascent/regime/engine.py tests/test_regime_hardening.py
git commit -m "feat: hard crisis override — VIX>30 AND SPY 5d<-7% forces crisis label"
```

---

## Task 6: Full regression — existing tests still pass

- [ ] **Step 1: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -q --tb=short 2>&1 | tail -20
```
Expected: all previously passing tests still pass, new hardening tests pass. No regressions.

- [ ] **Step 2: If any existing regime tests fail, diagnose and fix**

The most likely failure point: `RegimeDecisionEngine` is also constructed in `engine.py` in `check_and_run_emergency_refit` → calls `self.fit()` → the `fit()` path already uses `self._cfg` which now has the new keys from `REGIME_CONFIG_DEFAULTS`. This should be fine. If any test manually constructs `RegimeDecisionEngine` without the new kwargs, those default to `0.40`/`0.70` gracefully.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "test: regime hardening full regression clean"
```
