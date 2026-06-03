# Regime Detection Hardening — Design Spec
**Date:** 2026-06-02  
**Status:** Approved  
**Motivation:** April 2026 Liberation Day tariff crash. The HMM correctly read "stressed" from March 10 but never escalated to "crisis" during the actual drawdown. Entropy froze at 6e-11 (model lockup). Recovery flip on April 22 may have been premature. Three targeted fixes.

---

## Changes in scope

### A — Hard crisis override (engine.py)

**Problem:** No rule-based backstop. HMM stuck in "stressed" with entropy → 0 even during a genuine crisis event (SPY −10%, VIX 45).

**Solution:** New post-processing function `_apply_crisis_override(signal_df, spy_prices, vix_prices)` applied in `engine.fit()` after `_apply_vix_confirmation()`.

**Trigger condition (both required):**
- `VIX > CRISIS_VIX_THRESHOLD` (default: 30)
- Rolling 5-day SPY return `< CRISIS_SPY_5D_THRESHOLD` (default: −0.07)

**Effect:**
- Overwrite `label` → `"crisis"`
- Overwrite `risk_multiplier` → 0.40
- Set new column `crisis_override = True` on affected rows (False elsewhere)
- Log count of overridden days at INFO level

**What-if April 2026:** Triggers ~April 7 (SPY −10% cumulative, VIX ~45). Risk multiplier drops from 0.65 → 0.40, orchestrator shifts to US30/mac30/intl5/alt35. Estimated ~23% less equity gross exposure through the worst of the drawdown.

---

### C — Asymmetric hysteresis (decision.py)

**Problem:** `enter_threshold=0.65` is symmetric — downgrade (calm→stressed) and upgrade (stressed→calm) use the same threshold. Crises arrive fast; recoveries are slow. Symmetric thresholds make the model equally skeptical of both, which is wrong.

Note: `exit_threshold=0.45` already exists in the codebase but is a dead parameter — it is stored but never referenced in `_HysteresisStateMachine.step()`.

**Solution:** Add a `severity` dict `{state_idx: int}` to `_HysteresisStateMachine`. In `step()`, compare severity of candidate vs current:

```
SEVERITY = {calm_bull: 0, euphoric: 0, uncertain: 1, stressed: 2, crisis: 3}
```

- Transitioning to **worse** (higher severity) regime → `downgrade_threshold = 0.40`  
- Transitioning to **better** (lower severity) regime → `upgrade_threshold = 0.70`  
- Unknown severity → fall back to existing `enter_threshold`

**Config keys added to `REGIME_CONFIG_DEFAULTS`:**
- `"regime_downgrade_threshold": 0.40`
- `"regime_upgrade_threshold": 0.70`

**`RegimeDecisionEngine` change:** Build severity dict from `state_labels` before instantiating `_HysteresisStateMachine`. Pass as `severity=` kwarg.

**What-if April 2026:** The April 22 recovery flip (confident→calm_bull at 0.9999) still goes through; but any borderline flip near the 0.65 mark would have been held back. More importantly: future stressed transitions from calm_bull fire at 0.40, not 0.65 — regime detects the next crash ~1–2 days earlier.

---

### D — Entropy floor penalty (decision.py)

**Problem:** Entropy of 6e-11 is model lockup, not confidence. The system was treating maximum overconfidence as a reason to apply the full risk multiplier with no hedge.

**Solution:** In `RegimeDecisionEngine.process()`, after computing `risk_mult` for each signal row:

```python
ENTROPY_OVERCONFIDENCE_THRESHOLD = 1e-6
ENTROPY_OVERCONFIDENCE_PENALTY   = 0.90

if entropy < ENTROPY_OVERCONFIDENCE_THRESHOLD:
    risk_mult *= ENTROPY_OVERCONFIDENCE_PENALTY
    # log once per day at WARNING level
```

**Effect on April 2026:** Every "stressed" day with entropy 6e-11 → `risk_mult = 0.65 × 0.90 = 0.585`. Not dramatic on its own, but stacks with C and A.

**Effect on calm_bull:** Any calm_bull day with frozen entropy gets `1.00 × 0.90 = 0.90`. Mild; not expected to materially hurt performance on normal days.

---

## Files touched

| File | Change |
|------|--------|
| `ascent/regime/engine.py` | Add `_apply_crisis_override()`, call it in `fit()`. Add `CRISIS_VIX_THRESHOLD`, `CRISIS_SPY_5D_THRESHOLD` constants. |
| `ascent/regime/decision.py` | Add `ENTROPY_OVERCONFIDENCE_THRESHOLD`, `ENTROPY_OVERCONFIDENCE_PENALTY` constants. Modify `_HysteresisStateMachine.__init__` and `step()` for asymmetric thresholds. Modify `RegimeDecisionEngine.__init__` and `process()`. |
| `ascent/regime/types.py` | Add `"regime_downgrade_threshold"` and `"regime_upgrade_threshold"` to `REGIME_CONFIG_DEFAULTS`. |
| `tests/test_regime_hardening.py` | New test file (see below). |

---

## Tests

New file `tests/test_regime_hardening.py`:

1. **Crisis override fires:** Build a synthetic signal_df with "stressed" labels. Feed SPY series with 5-day return −0.10 and VIX=35. Assert labels flip to "crisis" and `crisis_override=True` on the relevant rows.
2. **Crisis override doesn't fire below threshold:** VIX=25 or SPY 5-day −0.05 — neither alone triggers. Assert labels unchanged.
3. **Asymmetric hysteresis — downgrade is faster:** State machine with calm_bull current, stressed candidate at prob=0.50. Should transition (0.50 > downgrade_threshold 0.40). Same setup with upgrade: stressed current, calm_bull candidate at prob=0.60. Should NOT transition (0.60 < upgrade_threshold 0.70).
4. **Entropy penalty applies:** Entropy below 1e-6 → risk_mult reduced by ×0.90. Entropy above 1e-6 → no change.
5. **Entropy penalty doesn't fire at normal entropy:** Entropy=0.30 → risk_mult unchanged.

---

## Integrity constraints preserved

- No look-ahead: `_apply_crisis_override` operates on the signal_df index using only prices up to each date (rolling 5-day window is causal).
- `crisis_override` column is auditable and distinct from HMM-native crisis labels.
- Debate layer, alpha stack, and portfolio construction are untouched.
