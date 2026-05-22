# Portfolio Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two structural upgrades: (1) IC-decay triggered early rebalance so the portfolio reacts to signal deterioration rather than waiting a fixed 10 days; (2) 130/30 long-short framework that shorts the bottom-decile alpha names, doubling information coefficient utilization with no new data cost.

**Architecture:** IC-decay trigger writes a flag file (`data_cache/rebalance_trigger.json`) that `run_all_agents.py` checks before deciding if today is a rebalance day. The 130/30 framework lives in `ascent/portfolio/long_short.py`, is gated by `LONG_SHORT_ENABLED = False` in `run_all_agents.py`, and feeds negative weights into the existing order engine (which gets short-sell support added). `pm_risk_validator.py` gets an `allow_shorts` flag to skip the negative-weight check when 130/30 is active.

**Tech Stack:** Python 3.12, pandas, numpy. Alpaca paper trading already supports short selling. No new API keys.

---

### Task 1: IC-Decay Triggered Early Rebalance

**Files:**
- Create: `ascent/monitoring/rebalance_trigger.py`
- Modify: `run_all_agents.py` (check trigger before deciding `is_rebalance`)
- Test: `tests/monitoring/test_rebalance_trigger.py`

**How it works:** On non-rebalance days, after computing sleeve ICs via `compute_signal_health`, check if the composite IC (weighted average across sleeves) has dropped ≥30% below the baseline stored in `data_cache/last_rebalance_state.json`. If yes AND ≥5 business days since last rebalance, write `data_cache/rebalance_trigger.json`. `run_all_agents.py` reads this flag: if present, treats today as a rebalance day and deletes the flag after execution.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_rebalance_trigger.py
import json
import pytest
from pathlib import Path
from unittest.mock import patch
import pandas as pd


def _make_state(date: str, ics: dict) -> dict:
    return {
        "date": date,
        "sleeve_ics": ics,
        "composite_ic": sum(ics.values()) / len(ics),
    }


def test_trigger_fires_on_ic_decay(tmp_path):
    from ascent.monitoring.rebalance_trigger import check_ic_decay_trigger

    state_path = tmp_path / "last_rebalance_state.json"
    trigger_path = tmp_path / "rebalance_trigger.json"

    # Baseline: healthy ICs
    state = _make_state("2026-05-10", {"trend": 0.10, "ml": 0.08, "statarb": 0.05})
    state_path.write_text(json.dumps(state))

    # Current: ICs have dropped 50%
    current_ics = {"trend": 0.05, "ml": 0.04, "statarb": 0.025}

    result = check_ic_decay_trigger(
        date="2026-05-19",          # 7 business days after 2026-05-10
        current_ics=current_ics,
        state_path=state_path,
        trigger_path=trigger_path,
        decay_threshold=0.30,
        min_days=5,
    )

    assert result is True
    assert trigger_path.exists()
    data = json.loads(trigger_path.read_text())
    assert "triggered_date" in data
    assert data["ic_decay_pct"] > 0.30


def test_trigger_does_not_fire_when_healthy(tmp_path):
    from ascent.monitoring.rebalance_trigger import check_ic_decay_trigger

    state_path = tmp_path / "last_rebalance_state.json"
    trigger_path = tmp_path / "rebalance_trigger.json"

    state = _make_state("2026-05-10", {"trend": 0.10, "ml": 0.08})
    state_path.write_text(json.dumps(state))

    # ICs only dropped 10% — below threshold
    current_ics = {"trend": 0.09, "ml": 0.073}

    result = check_ic_decay_trigger(
        date="2026-05-19",
        current_ics=current_ics,
        state_path=state_path,
        trigger_path=trigger_path,
        decay_threshold=0.30,
        min_days=5,
    )

    assert result is False
    assert not trigger_path.exists()


def test_trigger_does_not_fire_too_soon(tmp_path):
    from ascent.monitoring.rebalance_trigger import check_ic_decay_trigger

    state_path = tmp_path / "last_rebalance_state.json"
    trigger_path = tmp_path / "rebalance_trigger.json"

    state = _make_state("2026-05-19", {"trend": 0.10})   # rebalanced TODAY
    state_path.write_text(json.dumps(state))

    # ICs collapsed
    current_ics = {"trend": 0.02}

    result = check_ic_decay_trigger(
        date="2026-05-20",          # only 1 business day later
        current_ics=current_ics,
        state_path=state_path,
        trigger_path=trigger_path,
        decay_threshold=0.30,
        min_days=5,
    )

    assert result is False


def test_consume_trigger_deletes_file(tmp_path):
    from ascent.monitoring.rebalance_trigger import consume_trigger

    trigger_path = tmp_path / "rebalance_trigger.json"
    trigger_path.write_text(json.dumps({"triggered_date": "2026-05-19"}))

    assert consume_trigger(trigger_path=trigger_path) is True
    assert not trigger_path.exists()

    assert consume_trigger(trigger_path=trigger_path) is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/monitoring/test_rebalance_trigger.py -v
```
Expected: `ModuleNotFoundError: No module named 'ascent.monitoring.rebalance_trigger'`

- [ ] **Step 3: Write `ascent/monitoring/rebalance_trigger.py`**

```python
"""
ascent/monitoring/rebalance_trigger.py

Checks if sleeve ICs have decayed enough since last rebalance to warrant
an early rebalance. Writes a flag file that run_all_agents.py reads.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_STATE_PATH   = Path("data_cache/last_rebalance_state.json")
_DEFAULT_TRIGGER_PATH = Path("data_cache/rebalance_trigger.json")


def check_ic_decay_trigger(
    date: str,
    current_ics: dict[str, float],
    state_path: Path = _DEFAULT_STATE_PATH,
    trigger_path: Path = _DEFAULT_TRIGGER_PATH,
    decay_threshold: float = 0.30,
    min_days: int = 5,
) -> bool:
    """
    Returns True and writes trigger_path if composite IC has decayed ≥decay_threshold
    since last rebalance AND ≥min_days business days have passed.

    Args:
        date:            Today's date (YYYY-MM-DD).
        current_ics:     {sleeve_name: ic_value} from compute_signal_health.
        state_path:      Path to last_rebalance_state.json.
        trigger_path:    Path to write rebalance_trigger.json.
        decay_threshold: Fractional IC decay that triggers (default 0.30 = 30% drop).
        min_days:        Minimum business days since last rebalance before trigger fires.
    """
    if not state_path.exists():
        return False

    try:
        state = json.loads(state_path.read_text())
        last_date = pd.Timestamp(state["date"])
        baseline_ics: dict[str, float] = state.get("sleeve_ics", {})
        baseline_composite: float = state.get("composite_ic", 0.0)
    except Exception as e:
        log.warning("[RebalanceTrigger] Could not read state: %s", e)
        return False

    today = pd.Timestamp(date)
    bdays_since = len(pd.bdate_range(last_date, today)) - 1
    if bdays_since < min_days:
        return False

    if not current_ics or baseline_composite <= 0:
        return False

    # Compute current composite IC using same keys as baseline
    shared_keys = [k for k in baseline_ics if k in current_ics]
    if not shared_keys:
        return False

    current_composite = sum(current_ics[k] for k in shared_keys) / len(shared_keys)
    if baseline_composite <= 0:
        return False

    decay_pct = (baseline_composite - current_composite) / abs(baseline_composite)

    if decay_pct >= decay_threshold:
        payload = {
            "triggered_date": date,
            "days_since_rebalance": bdays_since,
            "baseline_ic": round(baseline_composite, 4),
            "current_ic": round(current_composite, 4),
            "ic_decay_pct": round(decay_pct, 4),
        }
        trigger_path.parent.mkdir(parents=True, exist_ok=True)
        trigger_path.write_text(json.dumps(payload, indent=2))
        log.warning(
            "[RebalanceTrigger] IC decay %.1f%% (baseline=%.4f, current=%.4f) "
            "after %d days — early rebalance triggered",
            decay_pct * 100, baseline_composite, current_composite, bdays_since,
        )
        return True

    return False


def consume_trigger(trigger_path: Path = _DEFAULT_TRIGGER_PATH) -> bool:
    """
    Returns True and deletes trigger_path if it exists. Call after executing
    an early rebalance so the flag is cleared.
    """
    if trigger_path.exists():
        trigger_path.unlink()
        return True
    return False


def is_triggered(trigger_path: Path = _DEFAULT_TRIGGER_PATH) -> bool:
    """Return True if an early rebalance has been flagged."""
    return trigger_path.exists()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_rebalance_trigger.py -v
```
Expected: `4 passed`

- [ ] **Step 5: Wire into `run_all_agents.py`**

Find the section where `is_rebalance` is determined (look for `rebalance_calendar.csv` read). After the calendar check, add:

```python
    # Early rebalance trigger: IC decay ≥30% since last rebalance after ≥5 bdays
    if not is_rebalance:
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

    # Consume trigger after rebalance executes (add at end of rebalance block)
```

Also add at the end of the rebalance execution block (after orders are submitted):

```python
    if is_rebalance:
        try:
            from ascent.monitoring.rebalance_trigger import consume_trigger
            consume_trigger()
        except Exception:
            pass
```

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/python -m pytest -q
```
Expected: 510+ passed, 1 skipped

- [ ] **Step 7: Commit**

```bash
git add ascent/monitoring/rebalance_trigger.py run_all_agents.py \
        tests/monitoring/test_rebalance_trigger.py
git commit -m "feat: IC-decay triggered early rebalance — fires when composite IC drops ≥30%"
```

---

### Task 2: 130/30 Long-Short Framework

**Files:**
- Create: `ascent/portfolio/long_short.py`
- Modify: `ascent/risk/pm_risk_validator.py` (add `allow_shorts` flag)
- Modify: `ascent/execution/order_engine.py` (short-sell orders for negative target weights)
- Modify: `run_all_agents.py` (add `LONG_SHORT_ENABLED = False` kill switch, wire in)
- Test: `tests/portfolio/test_long_short.py`

**How 130/30 works:** Go long the top-N names at 130% of NAV (same as before, slightly levered). Short the bottom-M names at 30% of NAV. Net exposure = 100%. The short book uses proceeds from shorting to fund the 30% overweight longs. Expected alpha: short alpha from bottom decile is typically stronger than long alpha from top decile.

**Kill switch:** `LONG_SHORT_ENABLED = False` in `run_all_agents.py`. Paper-validates before enabling. Do NOT set to True until ≥30 rebalances of paper history.

- [ ] **Step 1: Write the failing test**

```python
# tests/portfolio/test_long_short.py
import pytest
import pandas as pd
import numpy as np
from ascent.portfolio.long_short import build_long_short_weights


def _make_alpha(symbols: list[str], seed: int = 42) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(np.random.normal(0, 1, len(symbols)), index=symbols)


def test_weights_sum_to_one():
    alpha = _make_alpha(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"])
    weights = build_long_short_weights(alpha, long_n=6, short_n=3)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_long_short_signs():
    """Top 3 should be positive (long), bottom 2 should be negative (short)."""
    alpha = pd.Series({"A": 2.0, "B": 1.5, "C": 1.0, "D": -0.5, "E": -2.0})
    weights = build_long_short_weights(alpha, long_n=3, short_n=2,
                                       long_pct=1.30, short_pct=0.30)
    assert weights["A"] > 0
    assert weights["B"] > 0
    assert weights["C"] > 0
    assert weights["D"] < 0
    assert weights["E"] < 0


def test_long_exposure_is_130_pct():
    alpha = _make_alpha(["A", "B", "C", "D", "E", "F"])
    weights = build_long_short_weights(alpha, long_n=4, short_n=2,
                                       long_pct=1.30, short_pct=0.30)
    long_sum = sum(v for v in weights.values() if v > 0)
    assert abs(long_sum - 1.30) < 1e-6


def test_short_exposure_is_30_pct():
    alpha = _make_alpha(["A", "B", "C", "D", "E", "F"])
    weights = build_long_short_weights(alpha, long_n=4, short_n=2,
                                       long_pct=1.30, short_pct=0.30)
    short_sum = abs(sum(v for v in weights.values() if v < 0))
    assert abs(short_sum - 0.30) < 1e-6


def test_max_position_cap_respected():
    alpha = _make_alpha(["A", "B", "C", "D", "E"])
    weights = build_long_short_weights(alpha, long_n=3, short_n=2,
                                       long_pct=1.30, short_pct=0.30,
                                       max_long_weight=0.15)
    for sym, w in weights.items():
        if w > 0:
            assert w <= 0.15 + 1e-9


def test_not_enough_symbols():
    """Fewer symbols than long_n + short_n — should raise ValueError."""
    alpha = pd.Series({"A": 1.0, "B": -1.0})
    with pytest.raises(ValueError, match="Not enough symbols"):
        build_long_short_weights(alpha, long_n=3, short_n=2)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/portfolio/test_long_short.py -v
```
Expected: `ModuleNotFoundError: No module named 'ascent.portfolio.long_short'`

- [ ] **Step 3: Write `ascent/portfolio/long_short.py`**

```python
"""
ascent/portfolio/long_short.py

130/30 long-short portfolio construction.

Takes cross-sectional alpha scores and builds a market-neutral portfolio:
  - Long top `long_n` names at `long_pct` of NAV (default 130%)
  - Short bottom `short_n` names at `short_pct` of NAV (default 30%)
  - Net exposure = long_pct - short_pct = 100%

Kill switch: LONG_SHORT_ENABLED = False in run_all_agents.py.
Do not enable until ≥30 rebalances of paper trading history.
"""
from __future__ import annotations

import logging

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def build_long_short_weights(
    alpha_scores: pd.Series,
    long_n: int = 15,
    short_n: int = 5,
    long_pct: float = 1.30,
    short_pct: float = 0.30,
    max_long_weight: float = 0.15,
    max_short_weight: float = 0.10,
) -> dict[str, float]:
    """
    Build a 130/30 long-short portfolio from cross-sectional alpha scores.

    Args:
        alpha_scores:    Cross-sectional alpha scores indexed by symbol.
        long_n:          Number of long names (top alpha).
        short_n:         Number of short names (bottom alpha).
        long_pct:        Total long exposure as fraction of NAV (default 1.30).
        short_pct:       Total short exposure as fraction of NAV (default 0.30).
        max_long_weight: Max weight per long name (default 0.15 = 15% NAV).
        max_short_weight: Max weight per short name (default 0.10 = 10% NAV).

    Returns:
        Dict {symbol: weight} where longs are positive and shorts are negative.
        Weights sum to long_pct - short_pct = 1.0 (net 100%).

    Raises:
        ValueError: If fewer symbols than long_n + short_n.
    """
    scores = alpha_scores.dropna().sort_values(ascending=False)

    if len(scores) < long_n + short_n:
        raise ValueError(
            f"Not enough symbols: need {long_n + short_n}, got {len(scores)}"
        )

    long_names  = scores.head(long_n).index.tolist()
    short_names = scores.tail(short_n).index.tolist()

    # Rank-weight longs proportionally, cap at max_long_weight
    long_ranks  = range(long_n, 0, -1)
    long_raw    = {sym: float(r) for sym, r in zip(long_names, long_ranks)}
    long_total  = sum(long_raw.values())
    long_weights = {sym: (w / long_total) * long_pct for sym, w in long_raw.items()}

    # Apply max_long_weight cap (water-fill)
    for _ in range(50):
        capped   = {s: min(w, max_long_weight) for s, w in long_weights.items()}
        overflow = long_pct - sum(capped.values())
        if abs(overflow) < 1e-9:
            long_weights = capped
            break
        uncapped = [s for s, w in long_weights.items() if w < max_long_weight - 1e-9]
        if not uncapped:
            long_weights = capped
            break
        uncapped_total = sum(capped[s] for s in uncapped)
        if uncapped_total <= 0:
            long_weights = capped
            break
        for s in uncapped:
            capped[s] += overflow * (capped[s] / uncapped_total)
        long_weights = capped

    # Equal-weight shorts (simpler — fewer short names), cap at max_short_weight
    short_weight_each = min(short_pct / short_n, max_short_weight)
    short_weights = {sym: -short_weight_each for sym in short_names}

    # Combine and verify net = 1.0
    weights: dict[str, float] = {**long_weights, **short_weights}

    net = sum(weights.values())
    expected_net = long_pct - short_pct
    if abs(net - expected_net) > 1e-6:
        log.warning("[LongShort] Net exposure %.4f != expected %.4f", net, expected_net)

    return weights
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/portfolio/test_long_short.py -v
```
Expected: `6 passed`

- [ ] **Step 5: Add `allow_shorts` flag to `ascent/risk/pm_risk_validator.py`**

Find `validate_pm_proposal` in `pm_risk_validator.py`. Change the function signature and add a short-circuit before the negative-weight check:

```python
def validate_pm_proposal(
    portfolio: dict[str, float],
    allow_shorts: bool = False,
) -> tuple[bool, list[str]]:
    """
    Validate AI PM portfolio proposal.
    allow_shorts: if True, negative weights are permitted (130/30 mode).
    """
    if not portfolio:
        return False, ["empty portfolio"]

    violations = []

    # Negative weight check (skip if allow_shorts=True)
    if not allow_shorts:
        neg = [s for s, w in portfolio.items() if w < 0]
        if neg:
            return False, [f"negative weights not allowed: {neg}"]

    # ... rest of existing checks unchanged ...
```

- [ ] **Step 6: Update `ascent/execution/order_engine.py` to handle short orders**

Find `compute_orders` in `order_engine.py`. The function computes `target - current` for each symbol. For short positions:
- If `target_weight < 0` and we don't currently hold it → "sell short" (submit a `sell` order for shares to short)
- If `target_weight < 0` and we currently hold a long → close long first (sell to zero), then short

Add this handling after the existing order computation:

```python
    # Handle short positions (negative target weights)
    for sym, target_w in target_weights.items():
        if target_w >= 0:
            continue
        # Short position
        target_shares = int(abs(target_w) * nav / prices.get(sym, 1.0))
        current_shares = current_positions.get(sym, 0)
        if current_shares > 0:
            # Close existing long first
            orders.append({"symbol": sym, "side": "sell", "qty": current_shares,
                           "reason": "close long before shorting"})
        if target_shares > 0:
            orders.append({"symbol": sym, "side": "sell", "qty": target_shares,
                           "reason": "short sale (130/30)"})
```

Note: This is added in the section where orders are built. Exact line numbers depend on current order engine structure — search for `target_weights.items()` in `order_engine.py` and add after.

- [ ] **Step 7: Add `LONG_SHORT_ENABLED = False` to `run_all_agents.py`**

Near the top of `run_all_agents.py` (with other feature flags like `EVENT_TRADING_ENABLED`):

```python
LONG_SHORT_ENABLED = False  # 130/30 — enable after ≥30 paper rebalances (~July 2026)
```

Then in the portfolio construction section (after orchestrator blends weights), add:

```python
    # 130/30 long-short overlay (kill-switched)
    if LONG_SHORT_ENABLED:
        try:
            from ascent.portfolio.long_short import build_long_short_weights
            from ascent.monitoring.signal_health import compute_signal_health
            # Get composite alpha from US equities agent
            _us = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
            if _us is not None and _us.alpha_scores is not None and not _us.alpha_scores.empty:
                _alpha = _us.alpha_scores.iloc[-1].dropna()
                merged_weights = build_long_short_weights(
                    _alpha, long_n=15, short_n=5, long_pct=1.30, short_pct=0.30
                )
                print(f"[LongShort] 130/30 applied: {len([v for v in merged_weights.values() if v>0])} longs, "
                      f"{len([v for v in merged_weights.values() if v<0])} shorts")
        except Exception as _ls_e:
            print(f"[LongShort] Skipped: {_ls_e}")
```

- [ ] **Step 8: Run full test suite**

```bash
.venv/bin/python -m pytest -q
```
Expected: 516+ passed, 1 skipped

- [ ] **Step 9: Commit**

```bash
git add ascent/portfolio/long_short.py ascent/risk/pm_risk_validator.py \
        ascent/execution/order_engine.py run_all_agents.py \
        tests/portfolio/test_long_short.py
git commit -m "feat: 130/30 long-short framework (LONG_SHORT_ENABLED=False, kill-switched)"
```

---

## Self-Review

**Spec coverage check:**

1. ✅ IC-decay triggered early rebalance → Task 1
2. ✅ 130/30 long-short framework (kill-switched) → Task 2
3. ✅ `allow_shorts` flag in pm_risk_validator → Task 2 Step 5
4. ✅ Order engine short-sell support → Task 2 Step 6
5. ✅ `LONG_SHORT_ENABLED = False` kill switch → Task 2 Step 7

**Placeholder scan:** None found.

**Type consistency:**
- `check_ic_decay_trigger(date: str, current_ics: dict, ...) → bool` — consistent Task 1
- `consume_trigger(trigger_path) → bool` — consistent Task 1
- `build_long_short_weights(alpha_scores: pd.Series, ...) → dict[str, float]` — consistent Task 2
- `validate_pm_proposal(portfolio, allow_shorts=False) → tuple[bool, list[str]]` — consistent Task 2

**Important notes for the implementer:**
- `order_engine.py` Step 6 requires reading the file first to find exact insertion point — search for where `target_weights.items()` is iterated and orders are appended.
- The 130/30 is kill-switched — `LONG_SHORT_ENABLED = False` must NOT be changed until ≥30 paper rebalances confirm the IC is positive. Expected enable date ~August 2026.
- Short selling in Alpaca paper trading works with `side="sell"` on a stock you don't hold. Verify with: `python -c "from ascent.execution.alpaca_broker import get_account; print(get_account())"` to confirm margin is enabled on the paper account.
