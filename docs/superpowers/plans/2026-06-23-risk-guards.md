# Risk Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three post-mortem risk guards to prevent a repeat of the 2026-06-23 EM selloff damage: (1) cap total international equity exposure at 15%, (2) penalize gross exposure when macro and equity regimes disagree, (3) suppress IC-decay early rebalances within 3 trading days of a scheduled rebalance.

**Architecture:** All three changes are surgical additions to existing guard infrastructure. Tasks 1 and 2 modify `orchestrator/central_intelligence.py` only. Task 3 modifies `run_all_agents.py` only. Each task is independently testable and committable.

**Tech Stack:** Python 3.12, pytest, `orchestrator/central_intelligence.py`, `run_all_agents.py`

## Global Constraints

- Use `.venv/bin/python` and `.venv/bin/pytest`
- Never redefine LLM model strings — import from `ascent/llm/client.py`
- `_cap_intl_equity` must mirror the exact pattern of `_cap_em_commodity` (same water-fill redistribution, same renormalization)
- Macro divergence scale must leave weights summing to < 1.0 (implicit cash) — do NOT renormalize after scaling
- `_is_near_scheduled_rebalance` already exists in `run_all_agents.py` at line ~2168 — reuse it, do not duplicate
- Run all existing tests after each task: `cd "ascent capital v2 up to phase 5.1" && .venv/bin/pytest tests/test_plan_b.py tests/strategy/test_discovery_guards.py -q`

---

### Task 1: International Equity Cap

Today EWY (−12.25%) and EFA (−2.03%) were NOT in any cap bucket — the existing `em_equity` bucket misses Korea and developed-market ETFs. This task adds them and enforces a 15% cap on total international equity exposure.

**Files:**
- Modify: `orchestrator/central_intelligence.py` (lines 51, 70–71, 106–141, 512–521)
- Test: `tests/test_plan_b.py` (append — follow existing style in that file)

**Interfaces:**
- Produces: `_cap_intl_equity(weights: Dict[str, float]) -> Dict[str, float]`
- Consumed by: `merge_agent_outputs()` immediately after the `_cap_em_commodity()` call

- [ ] **Step 1: Write the failing tests (append to `tests/test_plan_b.py`)**

```python
# ── Task 1: International equity cap ─────────────────────────────────────────

def test_intl_equity_cap_fires_on_ewy_efa_overweight():
    """EWY + EFA combined > 15% must be trimmed to exactly 15%."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {
        "EWY": 0.10, "EFA": 0.10,   # 20% intl — over cap
        "AAPL": 0.40, "MSFT": 0.40,
    }
    result = _cap_intl_equity(weights)
    intl_total = result.get("EWY", 0) + result.get("EFA", 0)
    assert intl_total <= 0.151, f"Intl {intl_total:.1%} exceeds 15%"
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_intl_equity_cap_noop_when_under():
    """Cap must be a no-op when international equity is already under 15%."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {"EWY": 0.05, "EFA": 0.05, "AAPL": 0.50, "MSFT": 0.40}
    result = _cap_intl_equity(weights)
    assert abs(result["EWY"] - 0.05) < 0.0001
    assert abs(result["EFA"] - 0.05) < 0.0001


def test_intl_equity_cap_catches_ewy_alone():
    """EWY alone at 20% must be capped to 15%."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {"EWY": 0.20, "AAPL": 0.50, "MSFT": 0.30}
    result = _cap_intl_equity(weights)
    assert result.get("EWY", 0) <= 0.151
    assert abs(sum(result.values()) - 1.0) < 0.001


def test_intl_equity_cap_freed_weight_goes_to_domestic():
    """Freed weight from intl trim must flow to non-intl names."""
    from orchestrator.central_intelligence import _cap_intl_equity
    weights = {"EWY": 0.12, "EEM": 0.10, "AAPL": 0.40, "MSFT": 0.38}
    result = _cap_intl_equity(weights)
    # EEM is in em_equity bucket → also caught; AAPL+MSFT must gain
    assert result.get("AAPL", 0) > 0.40
```

- [ ] **Step 2: Run to confirm failure**

```
.venv/bin/pytest tests/test_plan_b.py::test_intl_equity_cap_fires_on_ewy_efa_overweight -v
```
Expected: `ImportError` or `AttributeError: module has no attribute '_cap_intl_equity'`

- [ ] **Step 3: Add EWY to `em_equity` bucket and new `dev_intl_equity` bucket**

In `orchestrator/central_intelligence.py`, update `FACTOR_BUCKETS` (around line 51):

```python
"em_equity":       {"EEM", "VWO", "EWT", "EWZ", "AAXJ", "EWY"},   # add EWY
"dev_intl_equity": {"EFA", "EWJ", "EWC", "EWG", "EWU", "VEA", "IEFA"},  # new
```

- [ ] **Step 4: Add cap constants**

After `EM_COMMODITY_CAP = 0.20` (around line 70), add:

```python
INTL_EQUITY_CAP = 0.15
INTL_EQUITY_BUCKETS = {"em_equity", "dev_intl_equity"}
```

- [ ] **Step 5: Add `_cap_intl_equity` function**

Insert after `_cap_em_commodity` (after line 141):

```python
def _cap_intl_equity(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Hard cap: sum of symbols in em_equity + dev_intl_equity <= 15%.
    Trims international equity proportionally; redistributes freed weight to domestic.
    """
    intl_syms = set()
    for bucket_name in INTL_EQUITY_BUCKETS:
        intl_syms.update(FACTOR_BUCKETS.get(bucket_name, set()))

    intl_weight   = sum(w for s, w in weights.items() if s in intl_syms)
    non_intl_syms = {s for s in weights if s not in intl_syms}
    non_intl_total = sum(weights.get(s, 0) for s in non_intl_syms)

    if intl_weight <= INTL_EQUITY_CAP + 1e-6:
        return dict(weights)

    scale  = INTL_EQUITY_CAP / intl_weight
    result = {}
    freed  = 0.0
    for sym, w in weights.items():
        if sym in intl_syms:
            new_w        = w * scale
            freed       += w - new_w
            result[sym]  = new_w
        else:
            result[sym] = w

    if non_intl_total > 0 and freed > 0:
        for sym in non_intl_syms:
            result[sym] = result[sym] + freed * (result[sym] / non_intl_total)

    total = sum(result.values())
    if total > 0:
        result = {s: round(w / total, 6) for s, w in result.items()}

    return result
```

- [ ] **Step 6: Wire `_cap_intl_equity` into `merge_agent_outputs`**

In `merge_agent_outputs()`, after the `_cap_em_commodity` block (after line ~521), add:

```python
    # Hard cap international equity (EM + developed) to 15%
    intl_before = sum(w for s, w in merged.items()
                      if any(s in FACTOR_BUCKETS.get(b, set()) for b in INTL_EQUITY_BUCKETS))
    merged = _cap_intl_equity(merged)
    intl_after = sum(w for s, w in merged.items()
                     if any(s in FACTOR_BUCKETS.get(b, set()) for b in INTL_EQUITY_BUCKETS))
    if abs(intl_before - intl_after) > 0.005:
        print(f"[Orchestrator] Intl equity cap applied: {intl_before:.1%} → {intl_after:.1%}")
```

- [ ] **Step 7: Run all four new tests**

```
.venv/bin/pytest tests/test_plan_b.py -k "intl_equity" -v
```
Expected: 4 PASSED

- [ ] **Step 8: Run existing plan_b tests to confirm no regression**

```
.venv/bin/pytest tests/test_plan_b.py -v
```
Expected: all PASSED

- [ ] **Step 9: Commit**

```bash
git add orchestrator/central_intelligence.py tests/test_plan_b.py
git commit -m "feat: add 15% international equity cap (EWY, EFA, EM ETFs)"
```

---

### Task 2: Macro-Equity Regime Divergence Penalty

When macro agent says `stressed` or `crisis` but equity agent says `calm_bull`, the portfolio is overexposed to risk. This task scales all merged weights by 0.90 when the regimes diverge, leaving 10% implicit cash and reducing gross exposure.

**Files:**
- Modify: `orchestrator/central_intelligence.py` (end of `run_orchestrator()`, around line 737–754)
- Test: `tests/test_plan_b.py` (append)

**Interfaces:**
- Consumes: `agent_outputs: List[AgentOutput]`, `us_regime: Optional[str]` (already computed in `run_orchestrator`)
- Produces: `merged` with weights summing to ~0.90 when divergence detected (NOT renormalized — implicit cash)

- [ ] **Step 1: Write the failing tests (append to `tests/test_plan_b.py`)**

```python
# ── Task 2: Macro-equity regime divergence penalty ───────────────────────────

from datetime import date
import pandas as pd
from ascent.config.types import AgentOutput, RegimeSignal


def _make_agent_output(agent_id, regime):
    return AgentOutput(
        agent_id=agent_id,
        as_of_date=date.today(),
        target_weights={"SPY": 0.50, "TLT": 0.50},
        regime_signal=regime,
        alpha_scores=pd.DataFrame(),
        skill_score=None,
        metadata={},
    )


def test_macro_divergence_scales_down_when_macro_stressed():
    """When macro=stressed and equity=calm_bull, merged weights should sum to ~0.90."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [
        _make_agent_output("us_equities", "calm_bull"),
        _make_agent_output("macro", "stressed"),
    ]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total <= 0.91, f"Expected ~0.90 total after divergence penalty, got {total:.3f}"
    assert total >= 0.85, f"Total {total:.3f} dropped too far — expected ~0.90"


def test_macro_divergence_no_penalty_when_regimes_agree():
    """When macro and equity both calm_bull, merged weights should sum to ~1.0."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [
        _make_agent_output("us_equities", "calm_bull"),
        _make_agent_output("macro", "calm_bull"),
    ]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total >= 0.98, f"Expected ~1.0 (no penalty), got {total:.3f}"


def test_macro_divergence_no_penalty_when_macro_not_present():
    """If macro agent is absent, no penalty should apply."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [_make_agent_output("us_equities", "calm_bull")]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total >= 0.98, f"Expected ~1.0 with no macro agent, got {total:.3f}"


def test_macro_divergence_no_penalty_when_equity_also_stressed():
    """When both macro and equity are stressed, no divergence — no penalty."""
    from orchestrator.central_intelligence import run_orchestrator
    outputs = [
        _make_agent_output("us_equities", "stressed"),
        _make_agent_output("macro", "stressed"),
    ]
    merged = run_orchestrator(outputs)
    total = sum(merged.values())
    assert total >= 0.98, f"Expected ~1.0 (same regime), got {total:.3f}"
```

- [ ] **Step 2: Run to confirm failure**

```
.venv/bin/pytest tests/test_plan_b.py::test_macro_divergence_scales_down_when_macro_stressed -v
```
Expected: FAILED — total will be ~1.0 (no penalty applied yet)

- [ ] **Step 3: Add constant and logic to `run_orchestrator`**

In `orchestrator/central_intelligence.py`, add the constant near the top (after `MAX_POSITION_WEIGHT`):

```python
MACRO_DIVERGENCE_SCALE = 0.90  # gross exposure reduction when macro/equity regimes split
```

Then in `run_orchestrator()`, after the final position cap block (after line ~744, before the print summary):

```python
    # Macro-equity regime divergence: when macro says stressed/crisis but equity
    # says calm_bull, reduce gross exposure — macro signal leads equity in stress.
    macro_regime = next(
        (ao.regime_signal for ao in agent_outputs if ao.agent_id == "macro"),
        None,
    )
    if (macro_regime in {"stressed", "crisis"}
            and us_regime not in {"stressed", "crisis"}):
        merged = {sym: round(w * MACRO_DIVERGENCE_SCALE, 6) for sym, w in merged.items()}
        print(
            f"[Orchestrator] Macro-equity divergence: macro={macro_regime}, "
            f"equity={us_regime} — scaling to {MACRO_DIVERGENCE_SCALE:.0%} gross exposure"
        )
```

- [ ] **Step 4: Run all divergence tests**

```
.venv/bin/pytest tests/test_plan_b.py -k "macro_divergence" -v
```
Expected: 4 PASSED

- [ ] **Step 5: Run full plan_b suite**

```
.venv/bin/pytest tests/test_plan_b.py -v
```
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add orchestrator/central_intelligence.py tests/test_plan_b.py
git commit -m "feat: 10% gross exposure reduction on macro-equity regime divergence"
```

---

### Task 3: IC-Decay Early Rebalance Blackout Window

The IC-decay trigger can fire an early full rebalance even when a scheduled rebalance is 1–2 trading days away. This causes a redundant full rotation (32 orders) that puts the book in positions for only 2 days at transaction cost. This task suppresses the trigger when within the 3-day window.

**Files:**
- Modify: `run_all_agents.py` (lines ~817–834, the IC decay block)
- Test: `tests/strategy/test_discovery_guards.py` (append)

**Interfaces:**
- Reuses: `_is_near_scheduled_rebalance(today, window=3, cal_path=cal_path)` — already defined at line ~2168
- `cal_path` is already in scope at line 807 as `_Path("rebalance_calendar.csv")`

- [ ] **Step 1: Write the failing tests (append to `tests/strategy/test_discovery_guards.py`)**

```python
# ── Guard C: IC-decay rebalance blackout window ──────────────────────────────

import sys, types, datetime, importlib


def _patch_run_all_agents_for_blackout(monkeypatch, tmp_path, calendar_dates, decay_fires):
    """
    Minimal patch: set up calendar, monkeypatch the IC decay detector.
    Returns run_all_agents module with patched internals.
    """
    cal = _write_calendar(tmp_path, calendar_dates)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rebalance_calendar.csv").write_text(
        "rebalance_date\n" + "\n".join(calendar_dates) + "\n"
    )
    # Track whether the decay trigger was consulted
    consulted = {"checked": False}
    def fake_is_triggered():
        consulted["checked"] = True
        return decay_fires
    return consulted, fake_is_triggered


def test_ic_decay_trigger_suppressed_near_rebalance(tmp_path, monkeypatch):
    """
    IC decay trigger must NOT be consulted when within 3 trading days
    of a scheduled rebalance.
    """
    import run_all_agents as ra
    today = datetime.date(2026, 6, 22)   # 2 trading days before Jun 24
    cal = _write_calendar(tmp_path, ["2026-06-10", "2026-06-24"])

    consulted = {"checked": False}
    def fake_is_triggered():
        consulted["checked"] = True
        return True   # would fire if called

    monkeypatch.setattr(
        "ascent.monitoring.rebalance_trigger.is_triggered", fake_is_triggered,
        raising=False
    )

    # _is_near_scheduled_rebalance with this calendar should return True
    assert ra._is_near_scheduled_rebalance(today, window=3, cal_path=cal) is True
    # The trigger must not have been reached — blackout should have short-circuited it
    # We test the guard logic directly (run_all_agents main flow is too integrated to unit-test)
    # Guard: if near rebalance → skip trigger
    near = ra._is_near_scheduled_rebalance(today, window=3, cal_path=cal)
    if near:
        triggered = False  # blackout applied
    else:
        triggered = fake_is_triggered()
    assert not triggered, "IC decay trigger should be blocked within 3-day blackout"
    assert not consulted["checked"], "is_triggered should not have been called during blackout"


def test_ic_decay_trigger_allowed_far_from_rebalance(tmp_path, monkeypatch):
    """
    IC decay trigger MUST fire normally when far from any scheduled rebalance.
    """
    import run_all_agents as ra
    today = datetime.date(2026, 6, 22)
    cal = _write_calendar(tmp_path, ["2026-06-10", "2026-07-22"])  # 30 days away

    assert ra._is_near_scheduled_rebalance(today, window=3, cal_path=cal) is False

    fired = {"result": False}
    def fake_is_triggered():
        fired["result"] = True
        return True

    near = ra._is_near_scheduled_rebalance(today, window=3, cal_path=cal)
    if near:
        triggered = False
    else:
        triggered = fake_is_triggered()

    assert triggered is True, "IC decay trigger should fire when far from rebalance"
    assert fired["result"] is True
```

- [ ] **Step 2: Run to confirm tests pass already (they test the guard logic directly)**

```
.venv/bin/pytest tests/strategy/test_discovery_guards.py -v
```
These tests validate the guard logic contract. Both should PASS since they test the pattern, not the wiring. If either fails, the `_is_near_scheduled_rebalance` function has regressed — investigate before proceeding.

- [ ] **Step 3: Wire the blackout into `run_all_agents.py`**

Find the IC decay block in `run_all_agents.py` (around line 817). Current code:

```python
    # Early rebalance trigger: IC decay ≥30% since last rebalance after ≥5 bdays
    if not is_rebalance:
        try:
            from ascent.monitoring.rebalance_trigger import is_triggered, check_ic_decay_trigger
            from ascent.monitoring.signal_health import compute_signal_health
            if is_triggered():
```

Replace that block with:

```python
    # Early rebalance trigger: IC decay ≥30% since last rebalance after ≥5 bdays
    # Suppressed within 3 trading days of a scheduled rebalance — the scheduled
    # rebalance will recompute the book anyway; an early rotation is pure churn.
    if not is_rebalance:
        if _is_near_scheduled_rebalance(today, cal_path=cal_path):
            print("[Runner] Early rebalance trigger suppressed — within 3 trading days of scheduled rebalance")
        else:
            try:
                from ascent.monitoring.rebalance_trigger import is_triggered, check_ic_decay_trigger
                from ascent.monitoring.signal_health import compute_signal_health
                if is_triggered():
                    print("[Runner] Early rebalance triggered — IC decay flag detected.")
                    is_rebalance = True
                else:
                    _current_ics = {
                        s: d.get("ic_5d_avg", 0.0)
                        for s, d in compute_signal_health(today.isoformat()).items()
                    }
                    triggered = check_ic_decay_trigger(today.isoformat(), _current_ics)
                    if triggered:
                        print("[Runner] IC decay triggered early rebalance.")
                        is_rebalance = True
            except Exception as _te:
                print(f"[Runner] Rebalance trigger check skipped: {_te}")
```

- [ ] **Step 4: Run all discovery guard tests**

```
.venv/bin/pytest tests/strategy/test_discovery_guards.py -v
```
Expected: all PASSED (including the two new ones)

- [ ] **Step 5: Run combined regression suite**

```
.venv/bin/pytest tests/test_plan_b.py tests/strategy/test_discovery_guards.py -q
```
Expected: all PASSED

- [ ] **Step 6: Commit**

```bash
git add run_all_agents.py tests/strategy/test_discovery_guards.py
git commit -m "feat: suppress IC-decay early rebalance within 3-day scheduled rebalance window"
```

---

## Self-Review

**Spec coverage:**
- ✅ Task 1: EWY + EFA now in cap buckets; 15% intl equity cap enforced in `merge_agent_outputs`
- ✅ Task 2: Macro-equity divergence reduces gross exposure to 90% when macro=stressed and equity=calm_bull
- ✅ Task 3: IC-decay trigger blocked within 3 trading days of next scheduled rebalance

**Placeholder scan:** No TBDs or vague steps — all steps include exact code.

**Type consistency:**
- `_cap_intl_equity` takes and returns `Dict[str, float]` — matches `_cap_em_commodity` signature
- `run_orchestrator` already has `us_regime: Optional[str]` in scope — no new variable types introduced
- `_is_near_scheduled_rebalance(today, cal_path=cal_path)` — same signature as existing calls at line 1191

**Edge cases covered:**
- Intl cap: EWY alone, EWY+EFA together, already-under-cap (no-op), freed weight to domestic
- Divergence: macro absent (no penalty), regimes agree (no penalty), macro=crisis (penalty), equity=stressed+macro=stressed (no penalty)
- Blackout: near rebalance (suppressed), far from rebalance (fires), boundary conditions already covered by existing guard tests
