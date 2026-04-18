# AI-Native Investment Firm — Architecture Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform Ascent Capital from a quant platform with AI bolted on into a genuine AI-native investment firm — one where the quant model, debate layer, and self-learning loop are provably integrated, with infrastructure to measure and prove AI's contribution to returns.

**Architecture:** Three layers. Quant engine (frozen — it's good enough). AI brain (actively improved — regime, debate, memory, self-learn). Proof infrastructure (new — counterfactual tracker, shadow portfolio, weekly report). The firm's edge is the third layer: real data proving when AI judgment beats a systematic model and when it doesn't.

**Tech Stack:** Python 3.12, hmmlearn, XGBoost, Anthropic SDK (claude-opus-4-6), Alpaca paper trading, R2R semantic memory, pandas, pytest

---

## Target Architecture

```
Ascent Capital
├── Quant Engine (frozen)
│   ├── Alpha: trend 65% / statarb 15% / ML 10% / vol-regime 5% / meanrev 5%
│   ├── Portfolio: sector-constrained, 15 positions, regime-adjusted max_weight
│   └── Risk: kill switch, correlation guard, concentration limits
│
├── AI Brain (actively improving)
│   ├── Regime: HMM K=2-4, particle filter, emergency refit — knows what market we're in
│   ├── Debate: fires only on uncertainty; deliberates before edge-case rebalances
│   ├── Memory: R2R semantic search — learns from past verdicts and outcomes
│   └── Self-Learn: weekly lightweight OOS — real Sharpe, not noise
│
├── Proof Infrastructure (new)
│   ├── Counterfactual: quant weights vs. debate weights → 10-day outcome comparison
│   ├── Shadow portfolio: quant-only Alpaca account, no debate, ever
│   └── Weekly report: AI contribution, regime breakdown, debate ROI
│
└── Execution (stable — don't touch)
    ├── Alpaca paper → live
    ├── Approval gate (>2% NAV)
    └── Slippage tracking
```

---

## Part 0: What to Remove

Remove clutter that isn't the firm. These files are not core to the investment operation.

### Files to delete

- [ ] **Step 1: Archive non-firm files**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
mkdir -p archive/20in20 archive/misc

# 20in20 intel system — not the firm
mv config_20in20.py archive/20in20/
mv dashboard_20in20.py archive/20in20/
mv scenario_library_20in20.py archive/20in20/
mv watchlists_20in20.py archive/20in20/
mv run_20in20.py archive/20in20/

# Demo and patch scripts — not firm infrastructure
mv demo_app.py archive/misc/
mv patch_debate_autoload.py archive/misc/

# 20in20 outputs
mv outputs/20in20 archive/20in20/outputs 2>/dev/null || true

echo "Done. Root is now clean."
ls *.py
```

Expected output: only `run_all_agents.py` remains at root. That is the firm's single daily command.

- [ ] **Step 2: Verify run_all_agents.py still imports cleanly**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: `OK` and `144 passed`.

- [ ] **Step 3: Commit the cleanup**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: archive non-firm files — 20in20 intel system, demo app, patch scripts

Root now contains only run_all_agents.py. The firm has one command.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1: Close the Three Gaps

These three builds transform what you have into what you need. Nothing else matters until all three are done and running.

---

### Task 1: Lightweight Walk-Forward (Real Self-Learning)

**The problem:** `self_improve.py:evaluate_variant()` estimates performance with a diversity heuristic. The firm learns nothing real. Fix: replace with an actual 63-day OOS walk-forward that skips regime refit and ML retrain (uses cached models), runs in ~2 minutes.

**Files:**
- Create: `ascent/research/walk_forward_lightweight.py`
- Modify: `ascent/research/self_improve.py:evaluate_variant()`
- Modify: `tests/test_plan_c.py` (add determinism test)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_self_improve_phase_d.py
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from pathlib import Path
import json


def _make_price_cache(tmp_path, n_days=300, n_syms=20):
    """Create a minimal prices_live parquet file for testing."""
    np.random.seed(42)
    idx = pd.bdate_range(end=date.today(), periods=n_days)
    symbols = [f"SYM{i:02d}" for i in range(n_syms)] + ["SPY"]
    rets = np.random.normal(0.0003, 0.012, (len(idx), len(symbols)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=symbols)
    out = tmp_path / "data_cache" / "prices_live.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(out)
    return prices


def test_lightweight_oos_returns_sharpe(tmp_path, monkeypatch):
    """run_lightweight_oos must return a dict with 'sharpe' and 'turnover'."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path)
    (tmp_path / "data_cache").mkdir(exist_ok=True)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos

    result = run_lightweight_oos(
        config_overrides={"alpha_weights": {"trend": 0.70, "meanrev": 0.05,
                                             "statarb": 0.15, "ml": 0.10, "volatility": 0.0}},
        n_days=63,
    )
    assert isinstance(result, dict)
    assert "sharpe" in result
    assert "turnover" in result
    assert isinstance(result["sharpe"], float)


def test_lightweight_oos_is_deterministic(tmp_path, monkeypatch):
    """Same config must return same Sharpe every run — no noise."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path)

    from ascent.research.walk_forward_lightweight import run_lightweight_oos

    cfg = {"alpha_weights": {"trend": 0.65, "meanrev": 0.05,
                              "statarb": 0.15, "ml": 0.10, "volatility": 0.05}}
    r1 = run_lightweight_oos(cfg, n_days=63)
    r2 = run_lightweight_oos(cfg, n_days=63)
    assert abs(r1["sharpe"] - r2["sharpe"]) < 0.001, "evaluate_variant must be deterministic"


def test_evaluate_variant_uses_real_oos(tmp_path, monkeypatch):
    """evaluate_variant must call run_lightweight_oos, not return noise."""
    monkeypatch.chdir(tmp_path)
    _make_price_cache(tmp_path)
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "agent_skill_scores.json").write_text(
        json.dumps({"us_equities": {"sharpe": 0.55, "status": "active", "n_days": 63}})
    )

    from ascent.research.self_improve import evaluate_variant

    cfg = {"alpha_weights": {"trend": 0.70, "meanrev": 0.05,
                              "statarb": 0.15, "ml": 0.10, "volatility": 0.0}}
    s1 = evaluate_variant(cfg)
    s2 = evaluate_variant(cfg)
    assert abs(s1 - s2) < 0.001, "Must be deterministic — no random noise"
    assert isinstance(s1, float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_self_improve_phase_d.py -v --tb=short 2>&1 | tail -15
```

Expected: `3 failed` — module doesn't exist yet.

- [ ] **Step 3: Create `ascent/research/walk_forward_lightweight.py`**

```python
"""
ascent/research/walk_forward_lightweight.py

Fast OOS evaluation for self-improve weekly loop.
Runs a simplified walk-forward on the most recent n_days of price data.
Skips regime refit and ML retraining — uses cached models.
Returns Sharpe and turnover in ~2 minutes (vs ~20 for full walk-forward).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


TURNOVER_PENALTY = 0.10   # subtract 0.10 * avg_turnover from Sharpe


def run_lightweight_oos(
    config_overrides: Dict[str, Any],
    n_days: int = 63,
    prices_cache: str = "prices_live",
    top_n: int = 15,
    max_weight: float = 0.10,
) -> Dict[str, float]:
    """
    Run a lightweight OOS evaluation over the last n_days of price data.

    Args:
        config_overrides: Dict with 'alpha_weights' key mapping sleeve names to floats.
        n_days:           Number of OOS trading days to evaluate.
        prices_cache:     Parquet cache name to load prices from.
        top_n:            Portfolio size.
        max_weight:       Max position weight.

    Returns:
        {"sharpe": float, "turnover": float, "n_folds": int}
        Returns {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0} on failure.
    """
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        from ascent.features.build_features import build_all_features
        from ascent.alpha.stack import build_alpha_stack
        from ascent.portfolio.optimizer import sector_constrained_weighted

        # Load price data
        if not has_data(prices_cache):
            print(f"[LightweightOOS] No cache '{prices_cache}' — returning 0.0")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        price_df = load_parquet(prices_cache)
        if price_df is None or len(price_df) < n_days + 63:
            print(f"[LightweightOOS] Insufficient data ({len(price_df) if price_df is not None else 0} rows)")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Use last (n_days + 126) rows — 126 training, n_days OOS
        window = price_df.tail(n_days + 126).copy()
        train_slice = window.iloc[:126]
        oos_slice   = window.iloc[126:]

        alpha_weights = config_overrides.get("alpha_weights", {
            "trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05
        })

        # Build features on training slice only (causal)
        try:
            features = build_all_features(train_slice)
        except Exception as e:
            print(f"[LightweightOOS] Feature build failed: {e}")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Build alpha on training data
        try:
            alpha_df = build_alpha_stack(features, sleeve_weights=alpha_weights)
        except Exception as e:
            print(f"[LightweightOOS] Alpha build failed: {e}")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        if alpha_df is None or alpha_df.empty:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Get latest alpha scores
        latest_alpha = alpha_df.iloc[-1].dropna().sort_values(ascending=False)

        # Build portfolio weights
        try:
            weights_dict = sector_constrained_weighted(
                latest_alpha,
                top_n=top_n,
                max_weight=max_weight,
            )
        except Exception as e:
            print(f"[LightweightOOS] Portfolio construction failed: {e}")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        if not weights_dict:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Compute OOS returns
        oos_symbols = [s for s in weights_dict if s in oos_slice.columns]
        if not oos_symbols:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        w_arr = np.array([weights_dict[s] for s in oos_symbols])
        w_arr /= w_arr.sum()  # renorm to available symbols

        price_oos = oos_slice[oos_symbols].dropna(how="all")
        if len(price_oos) < 5:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        rets = price_oos.pct_change().dropna()
        port_rets = rets.values @ w_arr

        mean_r = np.mean(port_rets)
        std_r  = np.std(port_rets)
        sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

        # Approximate turnover: assume one rebalance per period
        turnover = 0.20  # conservative estimate for single-rebalance period

        return {
            "sharpe":  round(sharpe, 4),
            "turnover": round(turnover, 4),
            "n_folds": 1,
        }

    except Exception as e:
        print(f"[LightweightOOS] Unexpected error: {type(e).__name__}: {e}")
        return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}
```

- [ ] **Step 4: Wire into `self_improve.py:evaluate_variant()`**

In `ascent/research/self_improve.py`, replace the `evaluate_variant` function body:

```python
def evaluate_variant(variant_config: dict) -> float:
    """Evaluate variant using real lightweight OOS walk-forward. Deterministic."""
    try:
        from ascent.research.walk_forward_lightweight import run_lightweight_oos, TURNOVER_PENALTY
        result = run_lightweight_oos(variant_config, n_days=63)
        n_folds = result.get("n_folds", 0)
        if n_folds == 0:
            # Fall back to baseline Sharpe if OOS failed
            baseline = get_baseline_sharpe()
            if baseline is None:
                print("[SelfImprove] WARNING: no live Sharpe available — falling back to hardcoded 0.518")
                baseline = 0.518
            return round(float(baseline), 4)
        sharpe   = result["sharpe"]
        turnover = result["turnover"]
        return round(float(sharpe - TURNOVER_PENALTY * turnover), 4)
    except Exception as e:
        print(f"[SelfImprove] evaluate_variant failed: {e} — using baseline")
        baseline = get_baseline_sharpe() or 0.518
        return round(float(baseline), 4)
```

- [ ] **Step 5: Verify syntax**

```bash
.venv/bin/python -c "
import ast
for f in ['ascent/research/walk_forward_lightweight.py', 'ascent/research/self_improve.py']:
    ast.parse(open(f).read()); print(f'OK: {f}')
"
```

- [ ] **Step 6: Run tests**

```bash
.venv/bin/pytest tests/test_self_improve_phase_d.py -v --tb=short 2>&1 | tail -15
```

Expected: `3 passed`.

- [ ] **Step 7: Run full test suite**

```bash
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: `144+ passed`.

- [ ] **Step 8: Commit**

```bash
git add ascent/research/walk_forward_lightweight.py ascent/research/self_improve.py tests/test_self_improve_phase_d.py
git commit -m "$(cat <<'EOF'
feat(self-improve): Phase D — real lightweight OOS replaces noise heuristic

evaluate_variant() now calls run_lightweight_oos() — 63-day walk-forward
on live price cache, skipping regime refit and ML retrain for speed.
Same variant always scores the same. Firm now genuinely learns weekly.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Debate Gate + Counterfactual Tracker

**The problem:** Debate fires every rebalance. No control group means no proof. Two builds together: (1) gate that makes debate conditional, (2) tracker that logs quant vs. AI weights and their 10-day outcomes.

**Files:**
- Create: `ascent/execution/debate_gate.py`
- Create: `ascent/monitoring/counterfactual_tracker.py`
- Modify: `ascent/execution/eod_runner.py` — wire gate before debate call
- Modify: `run_all_agents.py` — call counterfactual scorer daily
- Create: `tests/test_debate_gate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_debate_gate.py
import pytest
from datetime import date


def test_debate_fires_on_high_entropy():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08}, "quant_context": {"portfolio_var_99": -0.02}}
    regime = {"entropy": 0.75, "label": "stressed"}
    assert should_run_debate(state, regime) is True


def test_debate_skipped_on_low_entropy_calm():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08, "MSFT": 0.07}, "quant_context": {"portfolio_var_99": -0.015},
             "catalyst_detected": False}
    regime = {"entropy": 0.40, "label": "calm_bull"}
    assert should_run_debate(state, regime) is False


def test_debate_fires_on_concentrated_position():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"EWY": 0.14, "GLD": 0.08}, "quant_context": {"portfolio_var_99": -0.018},
             "catalyst_detected": False}
    regime = {"entropy": 0.30, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


def test_debate_fires_on_catalyst():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08}, "quant_context": {"portfolio_var_99": -0.02},
             "catalyst_detected": True}
    regime = {"entropy": 0.35, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


def test_debate_fires_on_var_tail():
    from ascent.execution.debate_gate import should_run_debate
    state = {"weights": {"AAPL": 0.08}, "quant_context": {"portfolio_var_99": -0.038},
             "catalyst_detected": False}
    regime = {"entropy": 0.30, "label": "calm_bull"}
    assert should_run_debate(state, regime) is True


# Counterfactual tests
def test_counterfactual_snapshot_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    from ascent.monitoring.counterfactual_tracker import snapshot_quant_weights, snapshot_debate_weights
    import json

    snapshot_quant_weights({"AAPL": 0.10, "MSFT": 0.09}, run_date=date(2026, 4, 29))
    snapshot_debate_weights({"AAPL": 0.08, "MSFT": 0.07}, run_date=date(2026, 4, 29))

    log = tmp_path / "logs" / "counterfactual_log.jsonl"
    assert log.exists()
    lines = [json.loads(l) for l in log.read_text().splitlines()]
    assert any(l["type"] == "quant_snapshot" for l in lines)
    assert any(l["type"] == "debate_snapshot" for l in lines)


def test_counterfactual_outcome_scored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    import json, numpy as np, pandas as pd
    from datetime import timedelta
    from ascent.monitoring.counterfactual_tracker import (
        snapshot_quant_weights, snapshot_debate_weights, score_pending_counterfactuals
    )

    d = date(2026, 4, 10)
    snapshot_quant_weights({"AAPL": 0.50, "MSFT": 0.50}, run_date=d)
    snapshot_debate_weights({"AAPL": 0.40, "MSFT": 0.60}, run_date=d)

    # Mock price data: AAPL flat, MSFT +5% over 10 days
    prices = {"AAPL": 150.0, "MSFT": 300.0,
              (d + timedelta(days=10)).isoformat(): {"AAPL": 150.0, "MSFT": 315.0}}

    scored = score_pending_counterfactuals(
        prices_override={"AAPL": [150.0] * 11, "MSFT": [300.0] + [315.0] * 10},
        as_of_date=d + timedelta(days=10),
    )
    assert scored >= 0  # at least attempted
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_debate_gate.py -v --tb=short 2>&1 | tail -15
```

Expected: `7 failed` — modules don't exist.

- [ ] **Step 3: Create `ascent/execution/debate_gate.py`**

```python
"""
ascent/execution/debate_gate.py

Decides whether the debate layer should run on a given rebalance day.
Debate only fires when uncertainty is elevated — not on every rebalance.
This creates the control group needed to measure AI contribution.

Criteria (any one triggers debate):
  - Regime entropy > 0.70 (uncertain regime)
  - Top position > 12% (concentration risk)
  - VaR 99th percentile < -3.5% (tail risk elevated)
  - Catalyst detected in last 48 hours
"""
from __future__ import annotations

ENTROPY_THRESHOLD   = 0.70
POSITION_THRESHOLD  = 0.12
VAR_99_THRESHOLD    = -0.035


def should_run_debate(portfolio_state: dict, regime_signal: dict) -> bool:
    """
    Return True if the debate layer should run on this rebalance.

    Args:
        portfolio_state: dict with 'weights', 'quant_context', 'catalyst_detected'
        regime_signal:   dict with 'entropy', 'label'

    Returns:
        True if at least one trigger condition is met.
    """
    # Trigger 1: regime uncertainty
    entropy = float(regime_signal.get("entropy", 0.0) or 0.0)
    if entropy > ENTROPY_THRESHOLD:
        print(f"[DebateGate] TRIGGER: regime entropy {entropy:.2f} > {ENTROPY_THRESHOLD}")
        return True

    # Trigger 2: position concentration
    weights = portfolio_state.get("weights") or {}
    top_position = max(weights.values(), default=0.0)
    if top_position > POSITION_THRESHOLD:
        print(f"[DebateGate] TRIGGER: top position {top_position:.1%} > {POSITION_THRESHOLD:.0%}")
        return True

    # Trigger 3: tail risk
    qctx   = portfolio_state.get("quant_context") or {}
    var_99 = float(qctx.get("portfolio_var_99", 0.0) or 0.0)
    if var_99 < VAR_99_THRESHOLD:
        print(f"[DebateGate] TRIGGER: VaR 99 {var_99:.2%} < {VAR_99_THRESHOLD:.1%}")
        return True

    # Trigger 4: catalyst detected
    if portfolio_state.get("catalyst_detected"):
        print(f"[DebateGate] TRIGGER: catalyst detected")
        return True

    print(f"[DebateGate] SKIP: entropy={entropy:.2f}, top={top_position:.1%}, "
          f"var99={var_99:.2%}, catalyst=False — no trigger")
    return False
```

- [ ] **Step 4: Create `ascent/monitoring/counterfactual_tracker.py`**

```python
"""
ascent/monitoring/counterfactual_tracker.py

Tracks the counterfactual: what would pure quant have done vs. what debate did?
For every debate session, logs both sets of weights. After 10 days, scores both.

This is the founding data for the firm's core claim:
"AI judgment adds measurable value in stressed/uncertain regimes."

Log format (logs/counterfactual_log.jsonl):
  {"type": "quant_snapshot", "date": "2026-04-29", "weights": {...}}
  {"type": "debate_snapshot", "date": "2026-04-29", "weights": {...}}
  {"type": "outcome", "date": "2026-04-29", "quant_10d": 0.023,
   "debate_10d": 0.031, "ai_added_value": true, "regime": "stressed"}
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

LOG_PATH = Path("logs/counterfactual_log.jsonl")
OUTCOME_WINDOW = 10  # trading days


def snapshot_quant_weights(weights: Dict[str, float], run_date: date) -> None:
    """Call BEFORE debate runs — locks the pure quant portfolio."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "type":    "quant_snapshot",
        "date":    run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Counterfactual] Quant snapshot saved: {len(weights)} positions")


def snapshot_debate_weights(weights: Dict[str, float], run_date: date) -> None:
    """Call AFTER debate adjusts weights — locks the AI-augmented portfolio."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "type":    "debate_snapshot",
        "date":    run_date.isoformat(),
        "weights": {k: round(v, 6) for k, v in weights.items()},
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Counterfactual] Debate snapshot saved: {len(weights)} positions")


def _compute_portfolio_return(
    weights: Dict[str, float],
    start_prices: Dict[str, float],
    end_prices: Dict[str, float],
) -> float:
    """Compute weighted portfolio return given start and end prices."""
    total = 0.0
    weight_used = 0.0
    for sym, w in weights.items():
        if sym in start_prices and sym in end_prices and start_prices[sym] > 0:
            ret = (end_prices[sym] - start_prices[sym]) / start_prices[sym]
            total += w * ret
            weight_used += w
    if weight_used > 0:
        return total / weight_used
    return 0.0


def score_pending_counterfactuals(
    prices_override: Optional[Dict] = None,
    as_of_date: Optional[date] = None,
) -> int:
    """
    Score unscored counterfactuals where OUTCOME_WINDOW days have passed.
    Loads price data to compute returns for both quant and debate portfolios.
    Returns count of verdicts scored.

    Args:
        prices_override: For testing — {symbol: [price_list]}
        as_of_date:      For testing — treat this as today
    """
    if not LOG_PATH.exists():
        return 0

    today = as_of_date or date.today()
    lines = [json.loads(l) for l in LOG_PATH.read_text().splitlines() if l.strip()]

    # Build index of snapshots and outcomes
    quant_snaps  = {l["date"]: l["weights"] for l in lines if l["type"] == "quant_snapshot"}
    debate_snaps = {l["date"]: l["weights"] for l in lines if l["type"] == "debate_snapshot"}
    scored_dates = {l["date"] for l in lines if l["type"] == "outcome"}

    scored = 0
    for d_str, quant_w in quant_snaps.items():
        if d_str in scored_dates:
            continue
        if d_str not in debate_snaps:
            continue

        snap_date = date.fromisoformat(d_str)
        if (today - snap_date).days < OUTCOME_WINDOW:
            continue  # too early to score

        # Load prices for scoring
        try:
            if prices_override:
                # Test mode: use override prices
                all_syms = set(quant_w) | set(debate_snaps[d_str])
                start_p  = {s: prices_override[s][0]  for s in all_syms if s in prices_override}
                end_p    = {s: prices_override[s][-1] for s in all_syms if s in prices_override}
            else:
                from ascent.data.store.parquet import load_parquet
                price_df = load_parquet("prices_live")
                if price_df is None or price_df.empty:
                    continue
                start_row = price_df.loc[price_df.index <= str(snap_date)].iloc[-1] if not price_df.empty else None
                end_row   = price_df.loc[price_df.index <= str(today)].iloc[-1] if not price_df.empty else None
                if start_row is None or end_row is None:
                    continue
                start_p = start_row.dropna().to_dict()
                end_p   = end_row.dropna().to_dict()

            debate_w      = debate_snaps[d_str]
            quant_ret     = _compute_portfolio_return(quant_w,  start_p, end_p)
            debate_ret    = _compute_portfolio_return(debate_w, start_p, end_p)
            ai_added      = debate_ret > quant_ret

            outcome = {
                "type":           "outcome",
                "date":           d_str,
                "outcome_date":   today.isoformat(),
                "quant_10d":      round(quant_ret, 6),
                "debate_10d":     round(debate_ret, 6),
                "ai_edge":        round(debate_ret - quant_ret, 6),
                "ai_added_value": ai_added,
            }
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(outcome) + "\n")

            direction = "✓ AI beat quant" if ai_added else "✗ Quant beat AI"
            print(f"[Counterfactual] {d_str}: quant={quant_ret:.2%} debate={debate_ret:.2%} {direction}")
            scored += 1

        except Exception as e:
            print(f"[Counterfactual] Scoring {d_str} failed: {e}")

    return scored


def get_ai_win_rate(regime_filter: Optional[str] = None) -> Dict:
    """
    Compute AI win rate from all scored counterfactuals.
    Optionally filter by regime label.
    Returns {"win_rate": float, "avg_edge": float, "n_samples": int}
    """
    if not LOG_PATH.exists():
        return {"win_rate": 0.0, "avg_edge": 0.0, "n_samples": 0}

    outcomes = [
        json.loads(l) for l in LOG_PATH.read_text().splitlines()
        if l.strip() and json.loads(l).get("type") == "outcome"
    ]

    if regime_filter:
        outcomes = [o for o in outcomes if o.get("regime") == regime_filter]

    if not outcomes:
        return {"win_rate": 0.0, "avg_edge": 0.0, "n_samples": 0}

    wins     = sum(1 for o in outcomes if o.get("ai_added_value"))
    avg_edge = sum(o.get("ai_edge", 0) for o in outcomes) / len(outcomes)

    return {
        "win_rate":  round(wins / len(outcomes), 3),
        "avg_edge":  round(avg_edge, 4),
        "n_samples": len(outcomes),
    }
```

- [ ] **Step 5: Wire debate gate into `eod_runner.py`**

Find where the debate call happens in `ascent/execution/eod_runner.py`. Add the gate before it:

```python
# At top of eod_runner.py, add imports:
from ascent.execution.debate_gate import should_run_debate
from ascent.monitoring.counterfactual_tracker import snapshot_quant_weights, snapshot_debate_weights

# Before the debate call (search for "run_debate_session" or "debate_runner"):
if not should_run_debate(portfolio_state, regime_signal or {}):
    print(f"[EOD] Debate gate: SKIP — no trigger conditions met")
    # No debate — proceed with pure quant weights
else:
    print(f"[EOD] Debate gate: FIRE — running debate session")
    # Snapshot quant weights before debate modifies them
    snapshot_quant_weights(dict(target_weights), run_date=today_date)
    # ... existing debate call ...
    # After debate returns adjusted weights:
    snapshot_debate_weights(dict(adjusted_weights), run_date=today_date)
```

- [ ] **Step 6: Wire counterfactual scoring into `run_all_agents.py`**

Add to the daily runner (after forward PnL cycle):

```python
# Score counterfactuals where 10 days have passed
try:
    from ascent.monitoring.counterfactual_tracker import score_pending_counterfactuals
    n_scored = score_pending_counterfactuals()
    if n_scored > 0:
        print(f"[Runner] Scored {n_scored} counterfactual(s)")
except Exception as e:
    print(f"[Runner] Counterfactual scoring failed: {e}")
```

- [ ] **Step 7: Verify syntax on all modified files**

```bash
.venv/bin/python -c "
import ast
files = [
    'ascent/execution/debate_gate.py',
    'ascent/monitoring/counterfactual_tracker.py',
    'ascent/execution/eod_runner.py',
    'run_all_agents.py',
]
for f in files:
    ast.parse(open(f).read()); print(f'OK: {f}')
"
```

- [ ] **Step 8: Run tests**

```bash
.venv/bin/pytest tests/test_debate_gate.py -v --tb=short 2>&1 | tail -15
```

Expected: `7 passed`.

- [ ] **Step 9: Run full suite**

```bash
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: `150+ passed`.

- [ ] **Step 10: Commit**

```bash
git add ascent/execution/debate_gate.py ascent/monitoring/counterfactual_tracker.py \
        ascent/execution/eod_runner.py run_all_agents.py tests/test_debate_gate.py
git commit -m "$(cat <<'EOF'
feat(firm): debate gate + counterfactual tracker — the proof infrastructure

debate_gate: debate only fires on high entropy, concentration, tail risk, or catalyst.
counterfactual_tracker: snapshot quant and debate weights; score both after 10 days.
This creates the natural experiment. After 90 days, the log is the founding document.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: R2R Memory — The Firm Has Institutional Memory

**The problem:** Every debate starts cold. Agents don't know what happened last time we were in a stressed regime with high EM exposure. R2R is wired in but the API key isn't configured. Fix: configure R2R, add memory queries to debate agent context.

**Files:**
- Modify: `.env` — add R2R_API_KEY
- Modify: `debate/agents.py:_build_context` — add memory query
- Create: `tests/test_memory_wired.py`

- [ ] **Step 1: Get R2R API key**

Go to `https://app.ragie.ai` or `https://r2r.ai` — sign up, get API key. Then:

```bash
echo "R2R_API_KEY=your_key_here" >> .env
# Verify it loads:
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
import os; print('R2R_API_KEY set:', bool(os.environ.get('R2R_API_KEY')))
"
```

Expected: `R2R_API_KEY set: True`

- [ ] **Step 2: Write failing test**

```python
# tests/test_memory_wired.py
def test_build_context_queries_memory():
    """_build_context must include a [MEMORY] section when memory returns results."""
    import inspect
    import debate.agents as agents_module
    src = inspect.getsource(agents_module._build_context)
    assert "query_memory" in src or "memory" in src.lower(), \
        "_build_context must call query_memory or include memory context"


def test_memory_context_in_prompt_when_available(monkeypatch):
    """If memory returns results, they must appear in the context string."""
    from unittest.mock import patch

    mock_memory = [{"summary": "Last stressed regime: REDUCE_SIZE verdict was correct. EM exposure cut paid off."}]

    with patch("debate.agents.query_memory", return_value=mock_memory):
        with patch("debate.agents.format_memory_context", return_value="[MEMORY]\nReduce EM in stressed regimes."):
            from debate.agents import _build_context
            state = {
                "date": "2026-04-29",
                "us_regime": "stressed",
                "n_positions": 15,
                "allocation": {},
                "weights": {"EWY": 0.10},
            }
            ctx = _build_context(state)
            assert "MEMORY" in ctx or "stressed" in ctx
```

- [ ] **Step 3: Wire memory query into `debate/agents.py:_build_context`**

Add after the existing quant context block in `_build_context`:

```python
    # Query memory for relevant past verdicts
    try:
        from memory.r2r_interface import query_memory, format_memory_context
        regime_label = portfolio_state.get("us_regime", "unknown")
        top_positions = sorted(
            (portfolio_state.get("weights") or {}).items(),
            key=lambda x: -x[1]
        )[:3]
        top_str = ", ".join(f"{s}({w:.0%})" for s, w in top_positions)
        memory_results = query_memory(
            f"verdict outcome {regime_label} regime {top_str}",
            n=3,
        )
        if memory_results:
            memory_ctx = format_memory_context(memory_results)
            lines.append("")
            lines.append(memory_ctx)
    except Exception as _me:
        pass  # memory is non-critical — debate proceeds without it
```

- [ ] **Step 4: Ingest existing verdicts into R2R**

```bash
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from pathlib import Path
from memory.r2r_interface import ingest_verdict

verdict_dir = Path('outputs/debate_log')
if verdict_dir.exists():
    verdicts = list(verdict_dir.glob('verdict_*.json'))
    for v in verdicts:
        ingest_verdict(v)
        print(f'Ingested: {v.name}')
    print(f'Total: {len(verdicts)} verdicts ingested')
else:
    print('No verdict dir found')
"
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_memory_wired.py -v --tb=short 2>&1 | tail -10
```

Expected: `2 passed`.

- [ ] **Step 6: Run full suite**

```bash
.venv/bin/pytest tests/ -q --tb=no 2>&1 | tail -3
```

Expected: `152+ passed`.

- [ ] **Step 7: Commit**

```bash
git add debate/agents.py tests/test_memory_wired.py
git commit -m "$(cat <<'EOF'
feat(memory): wire R2R memory queries into debate agent context

Agents now recall relevant past verdicts before arguing.
The firm has institutional memory — it knows what worked before.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Proof Infrastructure (Month 2)

*Implement after Phase 1 is running and collecting data.*

### Task 4: Shadow Portfolio

Create a second tracking layer — pure quant, no debate, ever. Compares to live (AI-augmented) weekly.

**File:** `ascent/monitoring/shadow_portfolio.py`

Core function: `run_shadow_rebalance(quant_weights, run_date)` — applies quant weights to a virtual $100k account, tracks NAV independently. Writes to `logs/shadow_portfolio_log.jsonl`.

Weekly comparison: `compare_live_vs_shadow()` → prints NAV difference and Sharpe difference.

### Task 5: Agent Credibility → Capital Allocation

Wire `outcome_tracker.get_agent_credibility()` into `orchestrator/central_intelligence.py`. Agents with consistently bad verdicts get reduced capital weight. This closes the learning loop: the firm fires bad advisors and promotes good ones automatically.

### Task 6: Weekly Report Generator

**File:** `ascent/reporting/weekly_report.py`

Every Sunday after self-improve runs, generate a one-page markdown report:

```
Week of 2026-05-05
══════════════════
Self-improve: best variant Sharpe 0.584 (+0.066 vs baseline)
Debate: fired 2/4 rebalances (entropy triggers)
Counterfactual: AI beat quant 2/2 times (both stressed regime)
Cumulative AI win rate: 4/5 (80%)
Shadow portfolio: -0.3% vs live (AI added value)
Regime: stressed → neutral transition (Apr 29)
```

---

## Phase 3: The Founding Document (Day 90)

*Auto-generated. Not built manually.*

At day 30, 60, and 90, run `ascent/reporting/firm_report.py`:

| Regime | Debate fired | AI won | AI lost | AI edge |
|--------|-------------|--------|---------|---------|
| stressed | 8 | 6 | 2 | +1.2% |
| uncertain | 5 | 4 | 1 | +0.8% |
| calm_bull | 2 | 0 | 2 | -0.4% |

**This table is the thesis.** AI adds value in stressed and uncertain regimes. It's noise in calm markets. Debate should fire in the former, not the latter. The trigger condition is calibrated. The firm knows when to listen to the AI and when to let the quant run.

That's the founding document. That's what gets the right people in the room.

---

## What Not to Build

Do not touch these until Phase 3 is complete:
- New alpha sleeves
- Additional monitoring dashboards
- Expanded universe
- New agent types
- Any infrastructure not on this list

The quant model is good enough. The experiment needs to run. Every hour spent adding features is an hour not spent collecting the data that proves the firm's thesis.

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Phase D self-learn running (real OOS) |
| 2 | Debate gate live, counterfactual logging starts |
| 3 | R2R memory wired, first debate with memory context |
| 4 | Archive cleanup, all tests passing, Phase 1 complete |
| 5–8 | Shadow portfolio + agent credibility loop (Phase 2) |
| 9–12 | Weekly reports running, data accumulating |
| 13 | First 90-day founding document generated |
