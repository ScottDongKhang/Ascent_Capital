# Plan B — Portfolio Hardening

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three portfolio construction gaps that guarantee underperformance: uncapped EM+commodity exposure, unenforced debate verdicts, and a stale regime signal.

**Architecture:** Three independent surgical changes: (1) EM+commodity cap in `central_intelligence.py`, (2) enforce `reduce_size` produces measurable weight change in `eod_runner.py`, (3) unstick the regime engine's scheduled refit in `ascent/regime/engine.py`.

**Tech Stack:** Existing orchestrator, eod_runner, regime engine. No new modules.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `orchestrator/central_intelligence.py` | Hard cap EM+commodity exposure at 20% |
| Modify | `ascent/execution/eod_runner.py` | Verify reduce_size actually reduces positions |
| Modify | `ascent/regime/engine.py` | Fix scheduled refit trigger |

---

## Task B1: Hard EM+commodity cap at 20%

**Problem:** Current factor contradiction detection only fires when *both* sides of a conflict are present. EM+commodity alone can grow to 37% with no cap. Regime rebalances toward macro but the merged weights still over-concentrate in commodities.

**The fix:** After all blending in `merge_agent_outputs()`, apply a hard sector cap: sum of all symbols in `em_equity` + `commodities` + `gold` buckets can't exceed 20% of merged portfolio.

**Files:**
- Modify: `orchestrator/central_intelligence.py:merge_agent_outputs` (add `_cap_em_commodity`)

- [ ] **Step 1: Write failing test**

```python
# tests/test_plan_b.py
import pytest

def test_em_commodity_cap_enforced():
    """Merged weights must not exceed 20% in EM+commodity+gold combined."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {
        "EEM":  0.12,
        "VWO":  0.10,
        "GLD":  0.08,
        "PDBC": 0.07,
        "AAPL": 0.25,
        "MSFT": 0.20,
        "JPM":  0.18,
    }
    capped = _cap_em_commodity(weights)

    em_commodity_total = (
        capped.get("EEM", 0) + capped.get("VWO", 0) +
        capped.get("GLD", 0) + capped.get("PDBC", 0)
    )
    assert em_commodity_total <= 0.201, f"EM+commodity total {em_commodity_total:.1%} exceeds 20% cap"
    assert abs(sum(capped.values()) - 1.0) < 0.001, "Weights must still sum to 1.0 after capping"


def test_em_commodity_cap_no_op_when_under():
    """Cap must be a no-op when EM+commodity is already under 20%."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {"EEM": 0.05, "GLD": 0.06, "AAPL": 0.50, "MSFT": 0.39}
    capped = _cap_em_commodity(weights)

    assert abs(capped["EEM"] - 0.05) < 0.0001
    assert abs(capped["GLD"] - 0.06) < 0.0001


def test_em_commodity_cap_preserves_non_em():
    """Non-EM symbols should gain weight proportionally when EM is trimmed."""
    from orchestrator.central_intelligence import _cap_em_commodity

    weights = {"EEM": 0.20, "GLD": 0.15, "AAPL": 0.40, "JPM": 0.25}
    capped = _cap_em_commodity(weights)

    em_total = capped.get("EEM", 0) + capped.get("GLD", 0)
    assert em_total <= 0.201
    # Non-EM symbols should have gained weight
    assert capped.get("AAPL", 0) > 0.40
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_plan_b.py::test_em_commodity_cap_enforced tests/test_plan_b.py::test_em_commodity_cap_no_op_when_under tests/test_plan_b.py::test_em_commodity_cap_preserves_non_em -v
```
Expected: FAIL — `_cap_em_commodity` does not exist.

- [ ] **Step 3: Add `_cap_em_commodity` and `EM_COMMODITY_CAP` to `central_intelligence.py`**

Add after the `FACTOR_CONTRADICTIONS` list (around line 52):

```python
# EM+commodity hard cap — prevents over-concentration in non-US-equity risk
EM_COMMODITY_CAP = 0.20
EM_COMMODITY_BUCKETS = {"em_equity", "commodities", "gold"}

def _cap_em_commodity(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Hard cap: sum of symbols in em_equity + commodities + gold buckets <= 20%.
    When over cap, trim EM/commodity symbols proportionally and redistribute
    freed weight to non-EM symbols.
    """
    # Identify which symbols fall in capped buckets
    em_syms = set()
    for bucket_name in EM_COMMODITY_BUCKETS:
        em_syms.update(FACTOR_BUCKETS.get(bucket_name, set()))

    # Current totals
    em_weight    = sum(w for s, w in weights.items() if s in em_syms)
    non_em_syms  = {s for s in weights if s not in em_syms}
    non_em_total = sum(weights.get(s, 0) for s in non_em_syms)

    if em_weight <= EM_COMMODITY_CAP + 1e-6:
        return dict(weights)  # no-op

    # Scale down EM symbols proportionally
    scale = EM_COMMODITY_CAP / em_weight
    result = {}
    freed  = 0.0
    for sym, w in weights.items():
        if sym in em_syms:
            new_w     = w * scale
            freed    += w - new_w
            result[sym] = round(new_w, 6)
        else:
            result[sym] = w

    # Redistribute freed weight to non-EM symbols proportionally
    if non_em_total > 0 and freed > 0:
        for sym in non_em_syms:
            result[sym] = round(result[sym] + freed * (result[sym] / non_em_total), 6)

    # Renorm to exactly 1.0
    total = sum(result.values())
    if total > 0:
        result = {s: round(w / total, 6) for s, w in result.items()}

    return result
```

- [ ] **Step 4: Call `_cap_em_commodity` in `merge_agent_outputs`**

In `merge_agent_outputs()`, after the crisis veto block and before the return statement, add:

```python
    # Hard cap EM+commodity to 20%
    em_before = sum(w for s, w in merged.items()
                    if any(s in FACTOR_BUCKETS.get(b, set()) for b in EM_COMMODITY_BUCKETS))
    merged = _cap_em_commodity(merged)
    em_after = sum(w for s, w in merged.items()
                   if any(s in FACTOR_BUCKETS.get(b, set()) for b in EM_COMMODITY_BUCKETS))
    if abs(em_before - em_after) > 0.005:
        print(f"[Orchestrator] EM+commodity cap applied: {em_before:.1%} → {em_after:.1%}")
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_plan_b.py::test_em_commodity_cap_enforced tests/test_plan_b.py::test_em_commodity_cap_no_op_when_under tests/test_plan_b.py::test_em_commodity_cap_preserves_non_em -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add orchestrator/central_intelligence.py tests/test_plan_b.py
git commit -m "feat(orchestrator): hard cap EM+commodity exposure at 20%"
```

---

## Task B2: Enforce `reduce_size` verdict produces measurable weight change

**Problem:** When the debate returns `reduce_size`, Claude Haiku adjusts weights. But there is no check that the adjusted weights are actually smaller than the original. Haiku can return `{"AAPL": 0.07, "MSFT": 0.06, ...}` that sums to 1.0 but individual positions haven't shrunk at all — the verdict has no teeth.

**The fix:** After Haiku returns adjusted weights, verify: (a) at least one position was reduced by ≥1%, or (b) total gross exposure is lower. If the Haiku response fails both checks, force a 20% haircut on all positions.

**Files:**
- Modify: `ascent/execution/eod_runner.py:_apply_verdict_adjustments`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_b.py
def test_reduce_size_enforces_actual_reduction():
    """reduce_size must produce weights that are measurably smaller than originals."""
    from ascent.execution.eod_runner import _enforce_reduce_size

    original = {"AAPL": 0.10, "MSFT": 0.09, "AMZN": 0.08, "NVDA": 0.07, "JPM": 0.06,
                "V": 0.06, "MA": 0.05, "UNH": 0.05, "HD": 0.05, "PG": 0.05,
                "BRK": 0.04, "XOM": 0.04, "LLY": 0.04, "JNJ": 0.04, "AVGO": 0.04,
                "META": 0.04, "GOOGL": 0.04, "TSLA": 0.03, "COST": 0.03, "NKE": 0.03}

    # Haiku returns same weights (no reduction)
    unchanged = dict(original)
    enforced = _enforce_reduce_size(original, unchanged)

    # Verify at least some positions were actually reduced
    reduced_count = sum(1 for s, w in enforced.items() if w < original.get(s, 0) - 0.005)
    assert reduced_count >= 3, f"Expected ≥3 positions reduced, got {reduced_count}"
    assert abs(sum(enforced.values()) - 1.0) < 0.001


def test_reduce_size_passes_through_genuine_reduction():
    """If Haiku genuinely reduced positions, pass through unchanged."""
    from ascent.execution.eod_runner import _enforce_reduce_size

    original = {"AAPL": 0.12, "MSFT": 0.10, "AMZN": 0.08, "NVDA": 0.07, "OTHER": 0.63}
    adjusted = {"AAPL": 0.08, "MSFT": 0.06, "AMZN": 0.05, "NVDA": 0.04, "OTHER": 0.77}

    result = _enforce_reduce_size(original, adjusted)
    # Should use Haiku's weights since they genuinely reduced top positions
    assert abs(result["AAPL"] - 0.08) < 0.001
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_plan_b.py::test_reduce_size_enforces_actual_reduction tests/test_plan_b.py::test_reduce_size_passes_through_genuine_reduction -v
```
Expected: FAIL — `_enforce_reduce_size` does not exist.

- [ ] **Step 3: Add `_enforce_reduce_size` to `eod_runner.py`**

Add after the existing imports in `eod_runner.py`:

```python
def _enforce_reduce_size(
    original_weights: dict,
    haiku_weights: dict,
    min_reduction_threshold: float = 0.01,
    min_positions_reduced: int = 3,
    forced_haircut: float = 0.80,  # multiply all weights by this if no genuine reduction
) -> dict:
    """
    Verify that Haiku's weight adjustment genuinely reduced positions.

    Checks: at least `min_positions_reduced` positions reduced by >=1%.
    If not, applies a forced `forced_haircut` to all haiku_weights and renorms.

    Args:
        original_weights:  Weights before reduce_size verdict
        haiku_weights:     Weights after Haiku adjustment
        min_reduction_threshold:  Min per-position reduction to count
        min_positions_reduced:    Min number of positions that must be reduced
        forced_haircut:    Multiplier applied if Haiku failed to reduce

    Returns:
        Final weights (either haiku_weights or forced reduction)
    """
    reduced_count = sum(
        1 for s, w in haiku_weights.items()
        if w < original_weights.get(s, 0) - min_reduction_threshold
    )

    if reduced_count >= min_positions_reduced:
        print(f"[EodRunner] reduce_size: Haiku reduced {reduced_count} positions — accepted")
        return haiku_weights

    # Haiku didn't actually reduce — apply forced haircut
    print(f"[EodRunner] reduce_size: Haiku only reduced {reduced_count} positions "
          f"(need {min_positions_reduced}) — forcing {(1-forced_haircut):.0%} haircut")
    haircut_weights = {s: w * forced_haircut for s, w in haiku_weights.items()}
    total = sum(haircut_weights.values())
    if total > 0:
        haircut_weights = {s: round(w / total, 6) for s, w in haircut_weights.items()}
    return haircut_weights
```

- [ ] **Step 4: Wire `_enforce_reduce_size` into `_apply_verdict_adjustments`**

In `_apply_verdict_adjustments` (or wherever Haiku's adjusted weights are accepted), add:

```python
    # After getting adjusted_weights from Haiku:
    adjusted_weights = _enforce_reduce_size(current_weights, adjusted_weights)
```

Find the existing Haiku adjustment block in `eod_runner.py`:
```bash
grep -n "reduce_size\|haiku\|adjusted_weights\|_apply_verdict" ascent/execution/eod_runner.py | head -30
```

Insert `_enforce_reduce_size` call after the Haiku JSON parse, before using `adjusted_weights`.

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_plan_b.py::test_reduce_size_enforces_actual_reduction tests/test_plan_b.py::test_reduce_size_passes_through_genuine_reduction -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ascent/execution/eod_runner.py tests/test_plan_b.py
git commit -m "feat(execution): enforce reduce_size verdict produces measurable weight reduction"
```

---

## Task B3: Fix regime detector staleness

**Problem:** `regime_refit_every_days=5` is set in `ascent/regime/types.py`, but the regime signal hasn't updated since March 19 (nearly a month). The engine's `check_and_run_emergency_refit()` method requires a manual trigger. The daily runner never calls `engine.maybe_refit()`.

**The fix:** Add a `last_refit_date` field to `regime_signal.json`. In `run_all_agents.py`, after agents run, check if the regime signal is stale (>5 days). If stale, trigger a refit on the US equities price data and update `regime_signal.json`.

**Files:**
- Modify: `run_all_agents.py` — add `_check_regime_staleness()` 
- Modify: `ascent/regime/engine.py` — expose `refit_and_export()` callable from runner

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_b.py
import json
from datetime import date, timedelta
from pathlib import Path

def test_regime_staleness_detected(tmp_path, monkeypatch):
    """Regime signal older than 5 days must be flagged as stale."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()

    stale_date = (date.today() - timedelta(days=8)).isoformat()
    signal = {
        "regime": "stressed",
        "label": "stressed",
        "as_of": stale_date,
        "last_refit_date": stale_date,
    }
    (tmp_path / "dashboard" / "regime_signal.json").write_text(json.dumps(signal))

    from run_all_agents import _is_regime_stale
    assert _is_regime_stale() is True, "Signal 8 days old should be stale"


def test_regime_staleness_fresh(tmp_path, monkeypatch):
    """Regime signal updated today must not be stale."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()

    signal = {
        "regime": "calm_bull",
        "as_of": date.today().isoformat(),
        "last_refit_date": date.today().isoformat(),
    }
    (tmp_path / "dashboard" / "regime_signal.json").write_text(json.dumps(signal))

    from run_all_agents import _is_regime_stale
    assert _is_regime_stale() is False, "Signal from today should not be stale"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_plan_b.py::test_regime_staleness_detected tests/test_plan_b.py::test_regime_staleness_fresh -v
```
Expected: FAIL — `_is_regime_stale` does not exist in `run_all_agents`.

- [ ] **Step 3: Add `_is_regime_stale` and `_refresh_regime` to `run_all_agents.py`**

Add after the imports section:

```python
REGIME_SIGNAL_PATH = Path("dashboard/regime_signal.json")
REGIME_STALE_DAYS  = 5

def _is_regime_stale() -> bool:
    """Return True if regime_signal.json is missing or older than REGIME_STALE_DAYS."""
    if not REGIME_SIGNAL_PATH.exists():
        return True
    try:
        sig = json.loads(REGIME_SIGNAL_PATH.read_text())
        # Check last_refit_date first, fall back to as_of
        date_str = sig.get("last_refit_date") or sig.get("as_of") or ""
        if not date_str:
            return True
        last = date.fromisoformat(date_str[:10])
        return (date.today() - last).days > REGIME_STALE_DAYS
    except Exception:
        return True


def _refresh_regime():
    """
    Trigger a regime refit using the US equities price cache.
    Updates regime_signal.json with the fresh label and today's last_refit_date.
    """
    print(f"[Runner] Regime signal stale — triggering refit")
    try:
        from ascent.config.settings import get_config
        from ascent.data.store.parquet import ParquetStore
        from ascent.regime.engine import RegimeEngine

        cfg = get_config()
        store = ParquetStore(cfg)
        prices = store.load("prices_live")
        if prices is None or prices.empty:
            print("[Runner] Cannot refit regime — no price data available")
            return

        spy_prices = prices["SPY"].dropna() if "SPY" in prices.columns else prices.iloc[:, 0].dropna()
        engine = RegimeEngine(cfg)
        engine.fit(spy_prices)
        label = engine.predict(spy_prices)

        # Write updated signal
        REGIME_SIGNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        sig = {
            "regime":           str(label),
            "label":            str(label),
            "as_of":            date.today().isoformat(),
            "last_refit_date":  date.today().isoformat(),
        }
        REGIME_SIGNAL_PATH.write_text(json.dumps(sig, indent=2))
        print(f"[Runner] Regime refreshed → {label}")

    except Exception as e:
        print(f"[Runner] Regime refit failed: {e}")
```

- [ ] **Step 4: Call `_refresh_regime` in the daily runner**

In the main block of `run_all_agents.py`, after `_check_halt_state()` and before agents start, add:

```python
    # Refresh regime signal if stale
    if _is_regime_stale():
        _refresh_regime()
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_plan_b.py::test_regime_staleness_detected tests/test_plan_b.py::test_regime_staleness_fresh -v
```
Expected: PASS

- [ ] **Step 6: Run all Plan B tests**

```bash
.venv/bin/pytest tests/test_plan_b.py -v
```
Expected: All PASS

- [ ] **Step 7: Smoke test regime refresh**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
# Artificially age the signal
python3 -c "
import json; from pathlib import Path; from datetime import date, timedelta
p = Path('dashboard/regime_signal.json')
sig = json.loads(p.read_text())
sig['last_refit_date'] = (date.today() - timedelta(days=10)).isoformat()
p.write_text(json.dumps(sig, indent=2))
print('Aged signal to:', sig['last_refit_date'])
"
# Now trigger the staleness check
.venv/bin/python -c "
from run_all_agents import _is_regime_stale, _refresh_regime
print('Stale?', _is_regime_stale())
_refresh_regime()
import json; sig = json.loads(open('dashboard/regime_signal.json').read())
print('New regime:', sig['regime'], '| last_refit:', sig['last_refit_date'])
"
```

- [ ] **Step 8: Commit**

```bash
git add run_all_agents.py tests/test_plan_b.py
git commit -m "feat(runner): auto-refresh regime signal when stale (>5 days)"
```
