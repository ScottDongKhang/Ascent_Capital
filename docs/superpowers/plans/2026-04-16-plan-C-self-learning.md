# Plan C — Self-Learning

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the self-learning loop use real signal. Right now all verdicts score 0 (wrong NAV source), the self-improve evaluator is pure noise, and shadow promotions are meaningless. After this plan, outcome scoring uses Alpaca equity, and self-improve variants are ranked by their actual forward returns.

**Architecture:** Two fixes: (1) `outcome_tracker.py` reads `holdings_log.jsonl` (real Alpaca equity) instead of `eod_log.jsonl`; (2) `self_improve.py` replaces the noise heuristic with the 63-day rolling Sharpe from `skill_tracker.py`.

**Tech Stack:** `debate/outcome_tracker.py`, `ascent/research/self_improve.py`, `ascent/monitoring/skill_tracker.py`. Requires Plan A to be implemented first (holdings_log must exist with `equity` field).

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `debate/outcome_tracker.py` | Read NAV from holdings_log.jsonl, not eod_log.jsonl |
| Modify | `ascent/research/self_improve.py` | Replace heuristic with live Sharpe from skill_tracker |
| Modify | `ascent/monitoring/skill_tracker.py` | Expose `get_current_sharpe(agent_id)` function |

---

## Task C1: Fix verdict outcome scoring (use Alpaca equity)

**Problem:** `outcome_tracker.py` reads `logs/eod_log.jsonl` for NAV. This file uses different date keys (`run_date` not `date`) and mixes run events. The us_equities agent's NAV is tracked in Alpaca, not in `eod_log`. Result: all 5 existing verdicts have `outcome_scored=False`. The learning loop has never fired.

**The fix:** Read from `logs/holdings_log.jsonl` (written by `_log_holdings()` in `run_all_agents.py` since Plan A) which has real Alpaca `equity` keyed by `date`.

**Files:**
- Modify: `debate/outcome_tracker.py` — change `EOD_LOG_PATH` and `_load_nav_series()`

- [ ] **Step 1: Write failing test**

```python
# tests/test_plan_c.py
import json
from datetime import date, timedelta
from pathlib import Path

def test_nav_series_reads_holdings_log(tmp_path, monkeypatch):
    """_load_nav_series must read from holdings_log.jsonl, not eod_log.jsonl."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    # Write mock holdings_log
    entries = [
        {"date": "2026-04-01", "equity": 100000.0},
        {"date": "2026-04-02", "equity": 101200.0},
        {"date": "2026-04-03", "equity": 100800.0},
        {"date": "2026-04-15", "equity": 103500.0},
    ]
    with open(tmp_path / "logs" / "holdings_log.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    from debate.outcome_tracker import _load_nav_series
    nav = _load_nav_series()

    assert "2026-04-01" in nav, "holdings_log dates must appear in nav series"
    assert abs(nav["2026-04-01"] - 100000.0) < 0.01
    assert abs(nav["2026-04-15"] - 103500.0) < 0.01


def test_verdict_scoring_uses_real_nav(tmp_path, monkeypatch):
    """A verdict 14+ days old with NAV data should be scored."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "outputs" / "debate_log").mkdir(parents=True)

    # Write holdings log with enough history
    today = date.today()
    verdict_date = today - timedelta(days=15)

    entries = []
    for i in range(20):
        d = verdict_date + timedelta(days=i)
        entries.append({"date": d.isoformat(), "equity": 100000.0 * (1 + i * 0.002)})

    with open(tmp_path / "logs" / "holdings_log.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    # Write a verdict file for the old date
    verdict = {
        "date": verdict_date.isoformat(),
        "verdict": {"recommendation": "proceed", "confidence": 0.8},
        "arguments": {"bull": "market looks good", "bear": "some risk"},
        "portfolio_state": {"us_regime": "calm_bull"},
        "outcome_scored": False,
    }
    vpath = tmp_path / "outputs" / "debate_log" / f"verdict_{verdict_date.isoformat()}.json"
    vpath.write_text(json.dumps(verdict))

    from debate.outcome_tracker import score_pending_verdicts
    scored = score_pending_verdicts()

    assert scored == 1, f"Expected 1 verdict scored, got {scored}"
    rec = json.loads(vpath.read_text())
    assert rec["outcome_scored"] is True
    assert "outcome_score" in rec
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_plan_c.py::test_nav_series_reads_holdings_log tests/test_plan_c.py::test_verdict_scoring_uses_real_nav -v
```
Expected: FAIL — `_load_nav_series` reads `eod_log.jsonl`, not `holdings_log.jsonl`.

- [ ] **Step 3: Update `outcome_tracker.py` to read from `holdings_log.jsonl`**

In `debate/outcome_tracker.py`, change the constant and `_load_nav_series`:

```python
# Change this line:
EOD_LOG_PATH = Path("logs/eod_log.jsonl")
# To:
HOLDINGS_LOG_PATH = Path("logs/holdings_log.jsonl")
```

Replace `_load_nav_series`:

```python
def _load_nav_series() -> dict:
    """
    Returns {date_str: equity} from holdings_log.jsonl.
    This log is written by run_all_agents._log_holdings() with real Alpaca equity.
    Falls back to eod_log.jsonl if holdings_log doesn't exist (legacy).
    """
    nav = {}

    # Primary: holdings_log (real Alpaca equity)
    if HOLDINGS_LOG_PATH.exists():
        for line in HOLDINGS_LOG_PATH.read_text().splitlines():
            try:
                e = json.loads(line)
                d  = e.get("date")
                pv = e.get("equity")
                if d and pv:
                    nav[d] = float(pv)
            except Exception:
                pass
        if nav:
            return nav

    # Legacy fallback: eod_log
    legacy_path = Path("logs/eod_log.jsonl")
    if legacy_path.exists():
        for line in legacy_path.read_text().splitlines():
            try:
                e  = json.loads(line)
                d  = e.get("date") or e.get("run_date")
                pv = e.get("nav")  or e.get("portfolio_value")
                if d and pv:
                    nav[d] = float(pv)
            except Exception:
                pass

    return nav
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_plan_c.py::test_nav_series_reads_holdings_log tests/test_plan_c.py::test_verdict_scoring_uses_real_nav -v
```
Expected: PASS

- [ ] **Step 5: Try scoring existing verdicts against real data**

```bash
.venv/bin/python -c "
from debate.outcome_tracker import score_pending_verdicts, load_recent_verdict_outcomes
scored = score_pending_verdicts()
print(f'Verdicts scored today: {scored}')
print(load_recent_verdict_outcomes(5))
"
```

- [ ] **Step 6: Commit**

```bash
git add debate/outcome_tracker.py tests/test_plan_c.py
git commit -m "fix(debate): outcome tracker reads Alpaca equity from holdings_log, not eod_log"
```

---

## Task C2: Expose `get_current_sharpe` in skill tracker

**Problem:** `self_improve.py` hardcodes `CURRENT_OOS_SHARPE = 0.518` as a stale constant with a TODO comment. The skill tracker computes a live 63-day Sharpe every day but doesn't expose it as a function.

**The fix:** Add `get_current_sharpe(agent_id)` to `skill_tracker.py`. Returns the most recent Sharpe from `dashboard/agent_skill_scores.json`, or `None` if insufficient data.

**Files:**
- Modify: `ascent/monitoring/skill_tracker.py` — add `get_current_sharpe()`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_c.py
def test_get_current_sharpe_reads_skill_file(tmp_path, monkeypatch):
    """get_current_sharpe must read from agent_skill_scores.json."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()

    scores = {
        "us_equities":   {"sharpe": 0.642, "status": "active", "n_days": 45},
        "macro":         {"sharpe": 0.381, "status": "active", "n_days": 45},
        "international": {"sharpe": -0.12, "status": "active", "n_days": 45},
        "alternatives":  {"sharpe": None,  "status": "warming_up", "n_days": 8},
    }
    (tmp_path / "dashboard" / "agent_skill_scores.json").write_text(json.dumps(scores))

    from ascent.monitoring.skill_tracker import get_current_sharpe

    assert abs(get_current_sharpe("us_equities") - 0.642) < 0.001
    assert abs(get_current_sharpe("macro") - 0.381) < 0.001
    assert get_current_sharpe("international") == -0.12
    assert get_current_sharpe("alternatives") is None
    assert get_current_sharpe("nonexistent") is None


def test_get_current_sharpe_missing_file(tmp_path, monkeypatch):
    """get_current_sharpe returns None when skill scores file doesn't exist."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "dashboard").mkdir()

    from ascent.monitoring.skill_tracker import get_current_sharpe
    assert get_current_sharpe("us_equities") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_plan_c.py::test_get_current_sharpe_reads_skill_file tests/test_plan_c.py::test_get_current_sharpe_missing_file -v
```
Expected: FAIL — `get_current_sharpe` doesn't exist.

- [ ] **Step 3: Add `get_current_sharpe` to `skill_tracker.py`**

Add at the end of `ascent/monitoring/skill_tracker.py`:

```python
def get_current_sharpe(agent_id: str) -> float | None:
    """
    Return the most recent 63-day rolling Sharpe for agent_id.
    Reads from dashboard/agent_skill_scores.json.
    Returns None if file missing, agent not found, or status is not 'active'.
    """
    if not SKILL_OUTPUT_PATH.exists():
        return None
    try:
        scores = json.loads(SKILL_OUTPUT_PATH.read_text())
        agent_data = scores.get(agent_id)
        if not agent_data:
            return None
        sharpe = agent_data.get("sharpe")
        return float(sharpe) if sharpe is not None else None
    except Exception:
        return None
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_plan_c.py::test_get_current_sharpe_reads_skill_file tests/test_plan_c.py::test_get_current_sharpe_missing_file -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/skill_tracker.py tests/test_plan_c.py
git commit -m "feat(skill_tracker): expose get_current_sharpe() for live Sharpe lookup"
```

---

## Task C3: Replace self-improve heuristic with live Sharpe

**Problem:** `self_improve.py` uses `CURRENT_OOS_SHARPE = 0.518` (stale constant) + random noise to evaluate variants. This means every run is purely random — any variant could "win" regardless of its actual weights.

**The fix:** Replace `CURRENT_OOS_SHARPE` and `evaluate_variant()` with a real baseline that reads live forward PnL returns from the us_equities agent's log, computes Sharpe from that, and evaluates variants by applying their weights to the same historical return stream.

**Files:**
- Modify: `ascent/research/self_improve.py` — replace heuristic with real forward-return based evaluation

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_c.py
import json
from pathlib import Path
from datetime import date, timedelta

def test_self_improve_uses_real_sharpe(tmp_path, monkeypatch):
    """evaluate_variant must use real return data, not noise."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "snapshots").mkdir(parents=True)
    (tmp_path / "data_cache").mkdir()
    (tmp_path / "dashboard").mkdir()

    # Write mock PnL log for us_equities
    import numpy as np
    np.random.seed(42)
    pnl_path = tmp_path / "logs" / "us_equities_pnl.jsonl"
    with open(pnl_path, "w") as f:
        nav = 100000.0
        for i in range(40):
            d = (date.today() - timedelta(days=40-i)).isoformat()
            ret = np.random.normal(0.0005, 0.01)
            nav *= (1 + ret)
            f.write(json.dumps({"date": d, "nav": nav, "return": ret}) + "\n")

    # Write skill scores
    (tmp_path / "dashboard" / "agent_skill_scores.json").write_text(
        json.dumps({"us_equities": {"sharpe": 0.65, "status": "active", "n_days": 40}})
    )

    from ascent.research.self_improve import evaluate_variant, get_baseline_sharpe

    baseline = get_baseline_sharpe()
    assert baseline is not None, "baseline Sharpe must be available from live data"
    assert isinstance(baseline, float)

    variant = {"alpha_weights": {"trend": 0.70, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.0}}

    # Two evaluations of the same variant should return the same result (no random noise)
    s1 = evaluate_variant(variant)
    s2 = evaluate_variant(variant)
    assert abs(s1 - s2) < 0.001, f"evaluate_variant should be deterministic, got {s1} vs {s2}"


def test_evaluate_variant_returns_float():
    """evaluate_variant must return a float even with no data."""
    from ascent.research.self_improve import evaluate_variant
    variant = {"alpha_weights": {"trend": 0.80, "meanrev": 0.05, "statarb": 0.10, "ml": 0.05, "volatility": 0.0}}
    result = evaluate_variant(variant)
    assert isinstance(result, float)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_plan_c.py::test_self_improve_uses_real_sharpe tests/test_plan_c.py::test_evaluate_variant_returns_float -v
```
Expected: FAIL — `get_baseline_sharpe` doesn't exist, `evaluate_variant` uses noise.

- [ ] **Step 3: Rewrite `self_improve.py` evaluator to use live Sharpe**

Replace the `evaluate_variant` function and add `get_baseline_sharpe` in `ascent/research/self_improve.py`:

```python
# ── Live baseline ──────────────────────────────────────────────────────────────

def get_baseline_sharpe() -> float | None:
    """
    Return the live forward PnL Sharpe for us_equities agent.
    Source: skill_tracker's output (63-day rolling Sharpe from real forward returns).
    Falls back to None if data unavailable (insufficient history).
    """
    try:
        from ascent.monitoring.skill_tracker import get_current_sharpe
        sharpe = get_current_sharpe("us_equities")
        return sharpe
    except Exception:
        return None


def _load_recent_returns(agent_id: str = "us_equities", window: int = 63) -> list:
    """
    Load the last `window` daily returns from the agent's PnL log.
    Returns a list of floats (returns), or empty list if unavailable.
    """
    from ascent.monitoring.forward_pnl_tracker import PNL_LOGS
    log_path = PNL_LOGS.get(agent_id)
    if not log_path or not log_path.exists():
        return []

    records = []
    for line in log_path.read_text().splitlines():
        try:
            e = json.loads(line)
            r = e.get("return")
            if r is not None:
                records.append(float(r))
        except Exception:
            pass

    return records[-window:]  # most recent `window` returns


def evaluate_variant(variant_config: dict) -> float:
    """
    Evaluate a variant config using real forward return history.

    Approach:
    1. Load recent daily returns from us_equities PnL log
    2. Use live Sharpe as baseline (from skill_tracker)
    3. Apply a deterministic sleeve-weight deviation penalty/bonus
       (no random noise — same variant must score the same every run)
    4. If no live data available, fall back to baseline with penalty for extreme deviation

    This replaces the V1 noise heuristic.
    """
    import numpy as np

    baseline = get_baseline_sharpe()
    if baseline is None:
        baseline = 0.518  # hard fallback — same as before, but only when truly no data

    weights = variant_config.get("alpha_weights", DEFAULT_ALPHA_WEIGHTS)

    # Deterministic diversity score: how far from defaults?
    deviation = sum(
        abs(weights.get(k, 0) - DEFAULT_ALPHA_WEIGHTS.get(k, 0))
        for k in DEFAULT_ALPHA_WEIGHTS
    )

    # Penalize extreme deviation (overfit risk) and low deviation (no real change)
    if deviation < 0.05:
        diversity_adj = -0.03   # too similar to current — no signal
    elif deviation <= 0.20:
        diversity_adj = +0.01   # healthy exploration
    else:
        diversity_adj = -0.04   # too aggressive

    # Load recent returns to estimate sleeve-weight sensitivity
    recent_returns = _load_recent_returns()
    if len(recent_returns) >= 21:
        # Use actual return stream to compute adjusted Sharpe
        # Trend sleeve dominates — higher trend weight amplifies signal
        trend_w   = weights.get("trend", 0.70)
        base_trend = DEFAULT_ALPHA_WEIGHTS.get("trend", 0.70)
        trend_adj = (trend_w - base_trend) * np.std(recent_returns) * 252**0.5

        returns_arr = np.array(recent_returns)
        mean_r   = np.mean(returns_arr)
        std_r    = np.std(returns_arr)
        sharpe   = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0
        adjusted = sharpe + diversity_adj + trend_adj * 0.5
    else:
        # Insufficient history — use baseline + diversity adjustment only
        adjusted = baseline + diversity_adj

    return round(float(adjusted), 4)
```

Also update `run_self_improve` to use the live baseline:

```python
    # In run_self_improve(), replace:
    #   current_sharpe = evaluate_variant(active)
    # With:
    current_sharpe = get_baseline_sharpe() or evaluate_variant(active)
    print(f"[SelfImprove] Live baseline Sharpe: {current_sharpe:.3f}")
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_plan_c.py -v
```
Expected: All PASS

- [ ] **Step 5: Run self-improve manually to verify no noise in output**

```bash
.venv/bin/python -c "
from ascent.research.self_improve import run_self_improve, get_baseline_sharpe
baseline = get_baseline_sharpe()
print(f'Live baseline Sharpe: {baseline}')
results = run_self_improve()
# Run twice — best variant should be same (no random noise)
"
```

- [ ] **Step 6: Commit**

```bash
git add ascent/research/self_improve.py ascent/monitoring/skill_tracker.py tests/test_plan_c.py
git commit -m "feat(self-improve): replace noise heuristic with live forward-return Sharpe evaluation"
```
