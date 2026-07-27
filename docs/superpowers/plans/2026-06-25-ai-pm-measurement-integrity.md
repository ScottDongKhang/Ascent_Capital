# AI PM Measurement Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the AI PM evaluation system accurate and fair by (1) tagging and excluding force-sealed runs from the authority ladder, (2) adding a Track Δ (delta portfolio) that isolates what the AI PM actually changed, and (3) adding per-rebalance cross-sectional conviction IC to the calibration report.

**Architecture:** Three independent improvements to the counterfactual/calibration stack. Force-seal tagging propagates through AIPMResult → snapshot log → daily log → Sortino buffer. Track Δ is derived inside `score_daily()` from existing D and A★ weights. Per-rebalance IC is a new computation path in `calibration_tracker.py` that works within each rebalance's position set.

**Tech Stack:** Python 3.12, stdlib only (json, pathlib, dataclasses). scipy.stats.spearmanr (already imported by calibration_tracker). Tests use `unittest.mock.patch` to redirect log paths to tmp dirs.

## Global Constraints

- Never raise from monitoring/tracking code — all public functions must be silent-failure safe.
- Existing daily log records that lack new fields are valid (treat missing `track_d_force_sealed` as False).
- Tests must patch log paths to tmp dirs — never read/write real log files in tests.
- Run full test sweep with: `.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20`
- Pre-existing failures (openbb-network, WF fixture IndexError, Sortino field) must not increase.

---

### Task 1: Force-seal tagging — AIPMResult + snapshot

Tag the AI PM result and snapshot when the force-seal pass fires, so downstream code can distinguish genuine judgment from a budget-exhausted fallback.

**Files:**
- Modify: `agents/ai_pm_agent.py` — `AIPMResult` dataclass + `run_ai_pm()` force-seal block
- Modify: `ascent/monitoring/ai_pm_counterfactual.py` — `snapshot_ai_pm()` signature + record
- Test: `tests/test_ai_pm_counterfactual.py`

**Interfaces:**
- Produces: `AIPMResult.force_sealed: bool` field  
- Produces: `counterfactual_ai_snapshots.jsonl` records carry `"force_sealed": true/false`  
- Produces: `snapshot_ai_pm(run_date, weights, force_sealed=False)` new optional param

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ai_pm_counterfactual.py`:

```python
def test_snapshot_ai_pm_writes_force_sealed_false_by_default():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_ai_pm, AI_PM_LOG
    import tempfile, json
    from datetime import date
    from pathlib import Path
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ai_snapshots.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", log_path):
            snapshot_ai_pm(date(2026, 6, 25), {"AAPL": 0.6, "MSFT": 0.4})
        entry = json.loads(log_path.read_text().strip())
    assert entry.get("force_sealed") == False


def test_snapshot_ai_pm_writes_force_sealed_true():
    from ascent.monitoring.ai_pm_counterfactual import snapshot_ai_pm, AI_PM_LOG
    import tempfile, json
    from datetime import date
    from pathlib import Path
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "ai_snapshots.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.AI_PM_LOG", log_path):
            snapshot_ai_pm(date(2026, 6, 25), {"AAPL": 0.6, "MSFT": 0.4}, force_sealed=True)
        entry = json.loads(log_path.read_text().strip())
    assert entry.get("force_sealed") == True


def test_aipm_result_has_force_sealed_field():
    from agents.ai_pm_agent import AIPMResult
    r = AIPMResult(portfolio={}, thesis={})
    assert r.force_sealed == False
    r2 = AIPMResult(portfolio={}, thesis={}, force_sealed=True)
    assert r2.force_sealed == True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py::test_snapshot_ai_pm_writes_force_sealed_false_by_default tests/test_ai_pm_counterfactual.py::test_snapshot_ai_pm_writes_force_sealed_true tests/test_ai_pm_counterfactual.py::test_aipm_result_has_force_sealed_field -v 2>&1 | tail -15
```
Expected: 3 FAILs — AttributeError or TypeError.

- [ ] **Step 3: Add `force_sealed` to `AIPMResult`**

In `agents/ai_pm_agent.py`, the `AIPMResult` dataclass is at line 313. Add one field:

```python
@dataclass
class AIPMResult:
    portfolio: Dict[str, float]
    thesis: Dict[str, Any]
    fallback: bool = False
    tool_failures: List[str] = field(default_factory=list)
    force_sealed: bool = False      # ← add this line
```

- [ ] **Step 4: Update `snapshot_ai_pm()` to accept and write the flag**

In `ascent/monitoring/ai_pm_counterfactual.py`, the function starts at line 72. Change the signature and the record dict:

```python
def snapshot_ai_pm(run_date: date, weights: Dict[str, float], force_sealed: bool = False) -> None:
    """Track D: AI PM proposed portfolio, normalized to sum=1.0 (longs only for normalisation).
    Handles signed weights — shorts preserved, longs renormalized.
    Call this AFTER Phase 2 completes on rebalance days."""
    longs  = {k: v for k, v in weights.items() if v > 0}
    shorts = {k: v for k, v in weights.items() if v < 0}

    long_total = sum(longs.values())
    if long_total > 0:
        longs = {k: v / long_total for k, v in longs.items()}

    normalized = {**longs, **shorts}
    written = _idempotent_write(AI_PM_LOG, run_date.isoformat(), {
        "date":         run_date.isoformat(),
        "weights":      {k: round(v, 6) for k, v in normalized.items()},
        "force_sealed": force_sealed,       # ← add this key
    })
    if written:
        log.info("[Counterfactual] Track D snapshot: %d longs, %d shorts",
                 len(longs), len(shorts))
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py::test_snapshot_ai_pm_writes_force_sealed_false_by_default tests/test_ai_pm_counterfactual.py::test_snapshot_ai_pm_writes_force_sealed_true tests/test_ai_pm_counterfactual.py::test_aipm_result_has_force_sealed_field -v 2>&1 | tail -10
```
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add agents/ai_pm_agent.py ascent/monitoring/ai_pm_counterfactual.py tests/test_ai_pm_counterfactual.py && git commit -m "feat(ai-pm): add force_sealed field to AIPMResult and Track D snapshot"
```

---

### Task 2: Force-seal tagging — mark the result in `run_ai_pm()`

Set `force_sealed=True` on the result when the Phase 2 force-seal pass fires.

**Files:**
- Modify: `agents/ai_pm_agent.py` — `run_ai_pm()` force-seal block (~line 2591)
- Test: `tests/test_ai_pm_counterfactual.py` (integration assertion via the dataclass)

**Interfaces:**
- Consumes: `AIPMResult.force_sealed` from Task 1
- Produces: `run_ai_pm()` returns result with `force_sealed=True` when force-seal fires

- [ ] **Step 1: Write failing test**

Add to `tests/test_ai_pm_counterfactual.py`:

```python
def test_run_ai_pm_fallback_result_is_not_force_sealed():
    """Fallback result (portfolio={}) should not be marked force_sealed — it means no snapshot was taken."""
    from agents.ai_pm_agent import AIPMResult
    r = AIPMResult(portfolio={}, thesis={}, fallback=True)
    assert r.force_sealed == False

def test_aipm_result_force_sealed_defaults_false():
    """Normal result should default to not force_sealed."""
    from agents.ai_pm_agent import AIPMResult
    r = AIPMResult(portfolio={"AAPL": 0.5}, thesis={"market_view": "bullish"})
    assert r.force_sealed == False
```

These pass immediately (just verifying the dataclass defaults from Task 1 are correct). The real behavioral change can't be unit-tested without mocking the full LLM loop — acceptance is confirmed by reading the decision log on the next real run.

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py::test_run_ai_pm_fallback_result_is_not_force_sealed tests/test_ai_pm_counterfactual.py::test_aipm_result_force_sealed_defaults_false -v 2>&1 | tail -10
```
Expected: 2 PASS.

- [ ] **Step 3: Mark result in `run_ai_pm()` when force-seal fires**

In `agents/ai_pm_agent.py`, locate the force-seal pass block. It begins at the comment `# Force-seal pass: mirror of the Phase 1 force-seal.` (around line 2586). Add a local flag before the block, and set it when the force-seal succeeds:

Find this section:
```python
    if not result_store:
        log.warning("[AIPMAgent] Phase 2: main pass exhausted without sealing — running force-seal pass")
        try:
```

Change to:
```python
    _phase2_force_sealed = False
    if not result_store:
        log.warning("[AIPMAgent] Phase 2: main pass exhausted without sealing — running force-seal pass")
        try:
```

Then find the line:
```python
                if result_store:
                    print("[AIPMAgent] Phase 2: force-seal succeeded")
                    break
```

Change to:
```python
                if result_store:
                    print("[AIPMAgent] Phase 2: force-seal succeeded")
                    _phase2_force_sealed = True
                    break
```

Then find the line `initial_result = result_store[-1]` (around line 2651) and add one line after it:
```python
    initial_result = result_store[-1]
    if _phase2_force_sealed:
        initial_result.force_sealed = True
```

- [ ] **Step 4: Full test sweep — no new failures**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: same pass/fail count as before these changes.

- [ ] **Step 5: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add agents/ai_pm_agent.py tests/test_ai_pm_counterfactual.py && git commit -m "feat(ai-pm): mark AIPMResult.force_sealed=True when Phase 2 force-seal pass fires"
```

---

### Task 3: Force-seal tagging — propagate to daily log + exclude from Sortino buffer

Write `track_d_force_sealed` into the daily log and have `rebuild_buffers_from_counterfactual()` skip those days.

**Files:**
- Modify: `ascent/monitoring/ai_pm_counterfactual.py` — `score_daily()` add `force_sealed` param
- Modify: `ascent/strategy/earned_authority.py` — `rebuild_buffers_from_counterfactual()` skip flag
- Modify: `run_all_agents.py` — pass `force_sealed` to `snapshot_ai_pm()` and `cf_score_daily()`
- Test: `tests/test_ai_pm_counterfactual.py` (score_daily writes flag)
- Test: `tests/test_ai_pm_authority.py` (rebuild excludes force-sealed days)

**Interfaces:**
- Consumes: `AIPMResult.force_sealed` from Task 1+2
- Produces: daily log records carry `"track_d_force_sealed": bool`
- Produces: `rebuild_buffers_from_counterfactual()` returns count of non-force-sealed observations

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ai_pm_counterfactual.py`:

```python
def test_score_daily_writes_track_d_force_sealed_false():
    from ascent.monitoring.ai_pm_counterfactual import score_daily, DAILY_LOG
    import tempfile, json
    from datetime import date
    from pathlib import Path
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 25),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.01,
                spy_return=0.005,
                prices={"AAPL": {"prev": 100.0, "curr": 101.0}, "MSFT": {"prev": 200.0, "curr": 201.0}},
            )
        entry = json.loads(log_path.read_text().strip())
    assert entry.get("track_d_force_sealed") == False


def test_score_daily_writes_track_d_force_sealed_true():
    from ascent.monitoring.ai_pm_counterfactual import score_daily, DAILY_LOG
    import tempfile, json
    from datetime import date
    from pathlib import Path
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 25),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.6, "MSFT": 0.4},
                track_b_return=0.01,
                spy_return=0.005,
                prices={"AAPL": {"prev": 100.0, "curr": 101.0}, "MSFT": {"prev": 200.0, "curr": 201.0}},
                force_sealed=True,
            )
        entry = json.loads(log_path.read_text().strip())
    assert entry.get("track_d_force_sealed") == True
```

Add to `tests/test_ai_pm_authority.py`:

```python
def test_rebuild_buffers_excludes_force_sealed_days():
    import json, tempfile
    from pathlib import Path
    from unittest.mock import patch
    import ascent.strategy.earned_authority as ea
    import ascent.monitoring.ai_pm_counterfactual as cf

    with tempfile.TemporaryDirectory() as tmp:
        daily_log = Path(tmp) / "counterfactual_daily.jsonl"
        state_path = Path(tmp) / "earned_authority.json"

        rows = [
            {"date": "2026-06-10", "track_d_return": 0.01,  "track_astar_return": 0.005, "track_d_force_sealed": False},
            {"date": "2026-06-11", "track_d_return": 0.02,  "track_astar_return": 0.015, "track_d_force_sealed": False},
            {"date": "2026-06-12", "track_d_return": 0.03,  "track_astar_return": 0.010, "track_d_force_sealed": True},
        ]
        daily_log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        state_path.write_text(json.dumps({
            "level": 1, "title": "Analyst", "ai_weight": 0.05,
            "phase": 1, "level_start_date": "2026-06-01",
            "days_at_level": 0, "days_stuck": 0,
            "in_cooldown": False, "cooldown_until": None,
            "auto_revert_count": 0, "last_updated": "2026-06-09",
            "track_d_returns": [], "track_astar_returns": [],
            "ai_returns_21d": [], "quant_returns_21d": [],
            "disable_sleeve_priors": False,
        }))

        with patch.object(cf, "DAILY_LOG", daily_log):
            with patch.object(ea, "STATE_PATH", state_path):
                n = ea.rebuild_buffers_from_counterfactual()

        state = json.loads(state_path.read_text())

    assert n == 2, f"Expected 2 non-force-sealed obs, got {n}"
    assert state["track_d_returns"] == [0.01, 0.02]
    assert state["track_astar_returns"] == [0.005, 0.015]


def test_rebuild_buffers_includes_days_without_force_sealed_field():
    """Old log records without track_d_force_sealed should be treated as not force-sealed."""
    import json, tempfile
    from pathlib import Path
    from unittest.mock import patch
    import ascent.strategy.earned_authority as ea
    import ascent.monitoring.ai_pm_counterfactual as cf

    with tempfile.TemporaryDirectory() as tmp:
        daily_log = Path(tmp) / "counterfactual_daily.jsonl"
        state_path = Path(tmp) / "earned_authority.json"

        # Old-format records without the field
        rows = [
            {"date": "2026-06-10", "track_d_return": 0.01, "track_astar_return": 0.005},
            {"date": "2026-06-11", "track_d_return": 0.02, "track_astar_return": 0.015},
        ]
        daily_log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        state_path.write_text(json.dumps({
            "level": 1, "title": "Analyst", "ai_weight": 0.05,
            "phase": 1, "level_start_date": "2026-06-01",
            "days_at_level": 0, "days_stuck": 0,
            "in_cooldown": False, "cooldown_until": None,
            "auto_revert_count": 0, "last_updated": "2026-06-09",
            "track_d_returns": [], "track_astar_returns": [],
            "ai_returns_21d": [], "quant_returns_21d": [],
            "disable_sleeve_priors": False,
        }))

        with patch.object(cf, "DAILY_LOG", daily_log):
            with patch.object(ea, "STATE_PATH", state_path):
                n = ea.rebuild_buffers_from_counterfactual()

    assert n == 2, f"Old records without field should be included; got {n}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py::test_score_daily_writes_track_d_force_sealed_false tests/test_ai_pm_counterfactual.py::test_score_daily_writes_track_d_force_sealed_true tests/test_ai_pm_authority.py::test_rebuild_buffers_excludes_force_sealed_days tests/test_ai_pm_authority.py::test_rebuild_buffers_includes_days_without_force_sealed_field -v 2>&1 | tail -15
```
Expected: 4 FAILs.

- [ ] **Step 3: Update `score_daily()` to accept and write `force_sealed`**

In `ascent/monitoring/ai_pm_counterfactual.py`, the `score_daily()` function starts at line 125. Add `force_sealed: bool = False` to the signature and `"track_d_force_sealed": force_sealed` to the record:

```python
def score_daily(
    run_date: date,
    quant_star_weights: Optional[Dict[str, float]],
    quant_weights: Optional[Dict[str, float]],
    ai_pm_weights: Optional[Dict[str, float]],
    track_b_return: float,
    spy_return: float,
    prices: Dict[str, dict],
    force_sealed: bool = False,        # ← add this param
) -> dict:
    """Compute all five track daily returns and append to DAILY_LOG."""
    astar_ret = _portfolio_return(quant_star_weights, prices) if quant_star_weights else None
    a_ret     = _portfolio_return(quant_weights, prices)      if quant_weights     else None
    d_ret     = _portfolio_return(ai_pm_weights, prices)      if ai_pm_weights     else None

    record = {
        "date":                  run_date.isoformat(),
        "track_astar_return":    round(astar_ret, 6) if astar_ret is not None else None,
        "track_a_return":        round(a_ret, 6)     if a_ret     is not None else None,
        "track_b_return":        round(float(track_b_return), 6),
        "track_c_return":        round(float(spy_return), 6),
        "track_d_return":        round(d_ret, 6)     if d_ret     is not None else None,
        "track_d_force_sealed":  force_sealed,        # ← add this field
    }

    _upsert_daily(record)
    return record
```

- [ ] **Step 4: Update `rebuild_buffers_from_counterfactual()` to skip force-sealed days**

In `ascent/strategy/earned_authority.py`, the function starts at line 112. Change the inner loop body:

```python
def rebuild_buffers_from_counterfactual() -> int:
    """..."""
    from ascent.monitoring.ai_pm_counterfactual import load_daily_records
    d_buf, as_buf = [], []
    for r in load_daily_records():
        d, a = r.get("track_d_return"), r.get("track_astar_return")
        if d is None or a is None:
            continue
        if r.get("track_d_force_sealed", False):   # ← skip force-sealed days
            log.debug("[EarnedAuthority] Skipping force-sealed day %s from Sortino buffer", r.get("date"))
            continue
        d_buf.append(float(d))
        as_buf.append(float(a))
    d_buf, as_buf = d_buf[-63:], as_buf[-63:]
    state = get_state()
    state["track_d_returns"]     = d_buf
    state["track_astar_returns"] = as_buf
    state["ai_returns_21d"]      = d_buf[-21:]
    state["quant_returns_21d"]   = as_buf[-21:]
    _save_state(state)
    log.info("[EarnedAuthority] Buffers rebuilt from counterfactual log: %d common-window obs", len(d_buf))
    return len(d_buf)
```

- [ ] **Step 5: Update `run_all_agents.py` to propagate the flag**

Three changes in `run_all_agents.py`:

**Change A** — Initialize `_ai_pm_force_sealed = False` before the AI PM try block. Find the line `ai_pm_result = run_ai_pm(` (around line 1395) and add the init just before the surrounding `try:`:

```python
        _ai_pm_force_sealed = False   # ← add this line
        try:
            print("[Runner] AI PM Phase 2 — synthesising pre-thesis with quant validation...")
            ai_pm_result = run_ai_pm(
```

**Change B** — Pass `force_sealed` to `snapshot_ai_pm()`. Find this line (~1420):
```python
                    snapshot_ai_pm(today, dict(ai_pm_result.portfolio))
```
Change to:
```python
                    _ai_pm_force_sealed = ai_pm_result.force_sealed
                    snapshot_ai_pm(today, dict(ai_pm_result.portfolio), force_sealed=_ai_pm_force_sealed)
```

**Change C** — Pass `force_sealed` to `cf_score_daily()`. Find the call to `cf_score_daily` (~line 2054):
```python
            _cf_record = cf_score_daily(
                run_date=today,
                quant_star_weights=_as_w or None,
                quant_weights=_a_w or None,
                ai_pm_weights=_d_w or None,
                track_b_return=day_ret,
                spy_return=spy_ret,
                prices=_cf_prices,
            )
```
Change to:
```python
            _cf_record = cf_score_daily(
                run_date=today,
                quant_star_weights=_as_w or None,
                quant_weights=_a_w or None,
                ai_pm_weights=_d_w or None,
                track_b_return=day_ret,
                spy_return=spy_ret,
                prices=_cf_prices,
                force_sealed=_ai_pm_force_sealed,
            )
```

- [ ] **Step 6: Run all four new tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py::test_score_daily_writes_track_d_force_sealed_false tests/test_ai_pm_counterfactual.py::test_score_daily_writes_track_d_force_sealed_true tests/test_ai_pm_authority.py::test_rebuild_buffers_excludes_force_sealed_days tests/test_ai_pm_authority.py::test_rebuild_buffers_includes_days_without_force_sealed_field -v 2>&1 | tail -15
```
Expected: 4 PASS.

- [ ] **Step 7: Full test sweep**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: no new failures vs. baseline.

- [ ] **Step 8: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add ascent/monitoring/ai_pm_counterfactual.py ascent/strategy/earned_authority.py run_all_agents.py tests/test_ai_pm_counterfactual.py tests/test_ai_pm_authority.py && git commit -m "feat(ai-pm): propagate force_sealed to daily log; exclude from Sortino buffer"
```

---

### Task 4: Track Δ — delta portfolio return in daily log

Add `track_delta_return` to every daily record: the return of the dollar-neutral "what the AI PM actually changed" portfolio. This is D weights minus A★ weights, normalized to one-way = 1.0. Positive means the AI PM's bets outperformed the quant on a given day.

**Files:**
- Modify: `ascent/monitoring/ai_pm_counterfactual.py` — new `_delta_portfolio_return()`, update `score_daily()`, `get_cumulative_returns()`, `print_cumulative_report()`
- Test: `tests/test_ai_pm_counterfactual.py`

**Interfaces:**
- Produces: `track_delta_return: Optional[float]` in daily log records
- Produces: `get_cumulative_returns()` returns `"track_delta"` key
- Produces: `print_cumulative_report()` prints Track Δ line

- [ ] **Step 1: Write failing tests**

Add to `tests/test_ai_pm_counterfactual.py`:

```python
def test_delta_portfolio_return_positive_when_ai_overweight_wins():
    """AI PM overweights AAPL vs A★. AAPL goes up. Delta return should be positive."""
    from ascent.monitoring.ai_pm_counterfactual import _delta_portfolio_return
    ai_pm   = {"AAPL": 0.7, "MSFT": 0.3}   # overweight AAPL vs A★
    astar   = {"AAPL": 0.5, "MSFT": 0.5}
    prices  = {
        "AAPL": {"prev": 100.0, "curr": 102.0},   # +2%
        "MSFT": {"prev": 200.0, "curr": 200.0},   # flat
    }
    result = _delta_portfolio_return(ai_pm, astar, prices)
    assert result is not None
    assert result > 0


def test_delta_portfolio_return_zero_when_ai_equals_astar():
    """When D == A★, delta return is 0.0 (no divergence to score)."""
    from ascent.monitoring.ai_pm_counterfactual import _delta_portfolio_return
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    prices  = {"AAPL": {"prev": 100.0, "curr": 102.0}, "MSFT": {"prev": 200.0, "curr": 201.0}}
    result = _delta_portfolio_return(weights, weights, prices)
    assert result == 0.0


def test_delta_portfolio_return_none_when_no_prices():
    """Returns None when no symbols can be priced."""
    from ascent.monitoring.ai_pm_counterfactual import _delta_portfolio_return
    result = _delta_portfolio_return({"AAPL": 0.7}, {"AAPL": 0.5}, {})
    assert result is None


def test_score_daily_includes_track_delta():
    """score_daily() writes track_delta_return to the daily log."""
    from ascent.monitoring.ai_pm_counterfactual import score_daily, DAILY_LOG
    import tempfile, json
    from datetime import date
    from pathlib import Path
    from unittest.mock import patch
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "daily.jsonl"
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            score_daily(
                run_date=date(2026, 6, 25),
                quant_star_weights={"AAPL": 0.5, "MSFT": 0.5},
                quant_weights={"AAPL": 0.5, "MSFT": 0.5},
                ai_pm_weights={"AAPL": 0.7, "MSFT": 0.3},
                track_b_return=0.01,
                spy_return=0.005,
                prices={
                    "AAPL": {"prev": 100.0, "curr": 102.0},
                    "MSFT": {"prev": 200.0, "curr": 199.0},
                },
            )
        entry = json.loads(log_path.read_text().strip())
    assert "track_delta_return" in entry


def test_get_cumulative_returns_includes_delta():
    """get_cumulative_returns() includes track_delta key."""
    from ascent.monitoring.ai_pm_counterfactual import get_cumulative_returns, DAILY_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch
    rows = [
        {"date": "2026-06-10", "track_b_return": 0.01, "track_astar_return": 0.005,
         "track_a_return": 0.007, "track_c_return": 0.003, "track_d_return": 0.012,
         "track_delta_return": 0.008},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "daily.jsonl"
        log_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        with patch("ascent.monitoring.ai_pm_counterfactual.DAILY_LOG", log_path):
            c = get_cumulative_returns()
    assert "track_delta" in c
    assert c["track_delta"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py::test_delta_portfolio_return_positive_when_ai_overweight_wins tests/test_ai_pm_counterfactual.py::test_delta_portfolio_return_zero_when_ai_equals_astar tests/test_ai_pm_counterfactual.py::test_delta_portfolio_return_none_when_no_prices tests/test_ai_pm_counterfactual.py::test_score_daily_includes_track_delta tests/test_ai_pm_counterfactual.py::test_get_cumulative_returns_includes_delta -v 2>&1 | tail -15
```
Expected: 5 FAILs.

- [ ] **Step 3: Add `_delta_portfolio_return()` function**

In `ascent/monitoring/ai_pm_counterfactual.py`, add this function after `_portfolio_return()` (after the line `return ret`):

```python
def _delta_portfolio_return(
    ai_pm_weights: Dict[str, float],
    quant_star_weights: Dict[str, float],
    prices: Dict[str, dict],
) -> Optional[float]:
    """Return of the normalized delta portfolio: long where D overweights A★, short where D underweights.
    One-way deviation normalized to 1.0. Returns None when no symbol can be priced, 0.0 when D == A★."""
    import math
    all_syms = set(ai_pm_weights) | set(quant_star_weights)
    deltas = {
        s: ai_pm_weights.get(s, 0.0) - quant_star_weights.get(s, 0.0)
        for s in all_syms
    }
    one_way = sum(abs(d) for d in deltas.values()) / 2.0
    if one_way < 1e-6:
        return 0.0  # D identical to A★ — no active bet
    normalized = {s: d / one_way for s, d in deltas.items() if abs(d) > 1e-9}
    return _portfolio_return(normalized, prices)
```

- [ ] **Step 4: Update `score_daily()` to compute and write Track Δ**

In `score_daily()` in `ascent/monitoring/ai_pm_counterfactual.py`, add Track Δ computation. The function's body currently builds `record`. Change it to:

```python
def score_daily(
    run_date: date,
    quant_star_weights: Optional[Dict[str, float]],
    quant_weights: Optional[Dict[str, float]],
    ai_pm_weights: Optional[Dict[str, float]],
    track_b_return: float,
    spy_return: float,
    prices: Dict[str, dict],
    force_sealed: bool = False,
) -> dict:
    """Compute all five track daily returns and append to DAILY_LOG."""
    astar_ret = _portfolio_return(quant_star_weights, prices) if quant_star_weights else None
    a_ret     = _portfolio_return(quant_weights, prices)      if quant_weights     else None
    d_ret     = _portfolio_return(ai_pm_weights, prices)      if ai_pm_weights     else None
    delta_ret = (
        _delta_portfolio_return(ai_pm_weights, quant_star_weights, prices)
        if ai_pm_weights and quant_star_weights else None
    )

    record = {
        "date":                  run_date.isoformat(),
        "track_astar_return":    round(astar_ret, 6)  if astar_ret  is not None else None,
        "track_a_return":        round(a_ret, 6)      if a_ret      is not None else None,
        "track_b_return":        round(float(track_b_return), 6),
        "track_c_return":        round(float(spy_return), 6),
        "track_d_return":        round(d_ret, 6)      if d_ret      is not None else None,
        "track_delta_return":    round(delta_ret, 6)  if delta_ret  is not None else None,
        "track_d_force_sealed":  force_sealed,
    }

    _upsert_daily(record)
    return record
```

- [ ] **Step 5: Update `get_cumulative_returns()` and `print_cumulative_report()`**

In `ascent/monitoring/ai_pm_counterfactual.py`, the `get_cumulative_returns()` function returns a dict. Add `track_delta` and a helper count:

```python
def get_cumulative_returns() -> dict:
    """..."""
    records = load_daily_records()
    if not records:
        return {}

    return {
        "n_days":           len(records),
        "start_date":       records[0]["date"],
        "end_date":         records[-1]["date"],
        "track_astar":      _cumret_over(records, "track_astar_return"),
        "track_a":          _cumret_over(records, "track_a_return"),
        "track_b":          _cumret_over(records, "track_b_return"),
        "track_c":          _cumret_over(records, "track_c_return"),
        "track_d":          _cumret_over(records, "track_d_return"),
        "track_delta":      _cumret_over(records, "track_delta_return"),
        # Honest apples-to-apples comparisons (common window only):
        "ai_value_add_b_vs_astar":  _common_window_diff(records, "track_b_return", "track_astar_return"),
        "ai_signal_d_vs_astar":     _common_window_diff(records, "track_d_return", "track_astar_return"),
        "ai_signal_delta":          _common_window_diff(records, "track_delta_return", "track_astar_return"),
        "n_common_b_astar":         len(_common_window(records, "track_b_return", "track_astar_return")),
        "n_common_d_astar":         len(_common_window(records, "track_d_return", "track_astar_return")),
        "n_delta_days":             sum(1 for r in records if r.get("track_delta_return") is not None),
        "n_delta_zero":             sum(1 for r in records if r.get("track_delta_return") == 0.0),
    }
```

Then update `print_cumulative_report()` to print Track Δ:

```python
def print_cumulative_report() -> None:
    c = get_cumulative_returns()
    if not c:
        print("[Counterfactual] No data yet")
        return
    def _f(v):
        return f"{v:+.2f}%" if v is not None else "  n/a"
    def _fp(v):
        return f"{v:+.2f}pp" if v is not None else "n/a (no common window)"
    print(f"[Counterfactual] Since AI PM live ({c['start_date']} → {c['end_date']}, {c['n_days']} days):")
    print(f"  Track A★ (Pure Quant):    {_f(c['track_astar'])}")
    print(f"  Track A  (Quant+P1):      {_f(c['track_a'])}")
    print(f"  Track B  (Actual):        {_f(c['track_b'])}")
    print(f"  Track C  (SPY):           {_f(c['track_c'])}")
    print(f"  Track D  (Pure AI PM):    {_f(c['track_d'])}")
    print(f"  Track Δ  (AI Active Bet): {_f(c['track_delta'])}  "
          f"({c.get('n_delta_days', 0)} days, {c.get('n_delta_zero', 0)} zero-divergence)")
    print(f"  AI value add  (B−A★): {_fp(c['ai_value_add_b_vs_astar'])} vs pure quant ({c['n_common_b_astar']} common days)")
    print(f"  AI signal     (D−A★): {_fp(c['ai_signal_d_vs_astar'])} — full authority estimate ({c['n_common_d_astar']} common days)")
```

Note: `ai_signal_delta` is intentionally not printed in the main report — Track Δ is already a delta (long-short), so comparing its cumulative to A★ doesn't make semantic sense. The cumulative Track Δ alone tells the story.

- [ ] **Step 6: Run all new Track Δ tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_ai_pm_counterfactual.py::test_delta_portfolio_return_positive_when_ai_overweight_wins tests/test_ai_pm_counterfactual.py::test_delta_portfolio_return_zero_when_ai_equals_astar tests/test_ai_pm_counterfactual.py::test_delta_portfolio_return_none_when_no_prices tests/test_ai_pm_counterfactual.py::test_score_daily_includes_track_delta tests/test_ai_pm_counterfactual.py::test_get_cumulative_returns_includes_delta -v 2>&1 | tail -15
```
Expected: 5 PASS.

- [ ] **Step 7: Full test sweep**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: no new failures.

- [ ] **Step 8: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add ascent/monitoring/ai_pm_counterfactual.py tests/test_ai_pm_counterfactual.py && git commit -m "feat(ai-pm): add Track Δ delta-portfolio return to daily log and report"
```

---

### Task 5: Per-rebalance cross-sectional conviction IC

Add `get_per_rebalance_ic()` to `calibration_tracker.py` and surface it in the calibration report. This answers "on rebalances where the AI had differentiated convictions, did higher conviction predict better realized return?"

**Files:**
- Modify: `ascent/strategy/calibration_tracker.py` — new function + update `get_calibration_report()`
- Test: `tests/test_calibration_tracker.py`

**Interfaces:**
- Produces: `get_per_rebalance_ic(n_rebalances=10) -> list[dict]`  
  Each element: `{"date": str, "ic": float, "n_positions": int, "n_conviction_levels": int}`  
  Only entries where ≥2 distinct conviction levels exist are included.
- Produces: `get_calibration_report()` includes per-rebalance IC breakdown section

- [ ] **Step 1: Write failing tests**

Add to `tests/test_calibration_tracker.py`:

```python
def test_get_per_rebalance_ic_returns_empty_when_all_same_conviction():
    """If every position has the same conviction level, IC is undefined — returns []."""
    from ascent.strategy.calibration_tracker import get_per_rebalance_ic, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": 0.02},
            "MSFT": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": 0.01},
            "GOOG": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": -0.01},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            result = get_per_rebalance_ic()
    assert result == []


def test_get_per_rebalance_ic_positive_when_high_conviction_wins():
    """High conviction position outperforms → positive IC for that rebalance."""
    from ascent.strategy.calibration_tracker import get_per_rebalance_ic, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.2, "conviction": "high",          "realized_21d": 0.05},
            "MSFT": {"weight": 0.1, "conviction": "medium",        "realized_21d": 0.02},
            "GOOG": {"weight": 0.1, "conviction": "quant_agreed",  "realized_21d": 0.01},
            "AMZN": {"weight": 0.1, "conviction": "quant_agreed",  "realized_21d": -0.01},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            result = get_per_rebalance_ic()
    assert len(result) == 1
    assert result[0]["date"] == "2026-06-10"
    assert result[0]["ic"] > 0
    assert result[0]["n_conviction_levels"] >= 2


def test_get_per_rebalance_ic_skips_entry_without_realized_outcomes():
    """Entry where no position has realized_21d is excluded entirely."""
    from ascent.strategy.calibration_tracker import get_per_rebalance_ic, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.2, "conviction": "high",         "realized_21d": None},
            "MSFT": {"weight": 0.1, "conviction": "quant_agreed", "realized_21d": None},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            result = get_per_rebalance_ic()
    assert result == []


def test_calibration_report_includes_per_rebalance_section():
    """get_calibration_report() contains per-rebalance IC section when data exists."""
    from ascent.strategy.calibration_tracker import get_calibration_report, CALIBRATION_LOG
    import tempfile, json
    from pathlib import Path
    from unittest.mock import patch

    entry = {
        "date": "2026-06-10",
        "positions": {
            "AAPL": {"weight": 0.2, "conviction": "high",          "realized_21d": 0.05},
            "MSFT": {"weight": 0.1, "conviction": "medium",        "realized_21d": 0.02},
            "GOOG": {"weight": 0.1, "conviction": "quant_agreed",  "realized_21d": -0.01},
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "calibration.jsonl"
        log_path.write_text(json.dumps(entry) + "\n")
        with patch("ascent.strategy.calibration_tracker.CALIBRATION_LOG", log_path):
            report = get_calibration_report()
    assert "per-rebalance" in report.lower() or "Per-rebalance" in report
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_calibration_tracker.py::test_get_per_rebalance_ic_returns_empty_when_all_same_conviction tests/test_calibration_tracker.py::test_get_per_rebalance_ic_positive_when_high_conviction_wins tests/test_calibration_tracker.py::test_get_per_rebalance_ic_skips_entry_without_realized_outcomes tests/test_calibration_tracker.py::test_calibration_report_includes_per_rebalance_section -v 2>&1 | tail -15
```
Expected: 4 FAILs.

- [ ] **Step 3: Add `get_per_rebalance_ic()` to `calibration_tracker.py`**

Add this function after `get_calibration_report()` in `ascent/strategy/calibration_tracker.py`:

```python
def get_per_rebalance_ic(n_rebalances: int = 10) -> list:
    """Spearman IC of conviction_order → realized_21d within each rebalance.

    Returns one dict per rebalance entry that has (a) at least one realized outcome
    and (b) at least 2 distinct conviction levels. Entries where all positions share
    the same conviction level are skipped — IC is undefined there.

    Each result: {"date": str, "ic": float, "n_positions": int, "n_conviction_levels": int}
    Never raises.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return []

    try:
        entries = _read_log()
        entries_with_outcomes = [
            e for e in entries
            if any(pos.get("realized_21d") is not None for pos in e.get("positions", {}).values())
        ][-n_rebalances:]

        results = []
        for entry in entries_with_outcomes:
            convictions, returns = [], []
            for pos in entry.get("positions", {}).values():
                r21 = pos.get("realized_21d")
                if r21 is None:
                    continue
                conviction = pos.get("conviction", "quant_agreed")
                convictions.append(CONVICTION_ORDER.get(conviction, 0))
                returns.append(float(r21))

            if len(convictions) < 3:
                continue
            if len(set(convictions)) < 2:
                continue  # all same conviction level — IC undefined

            try:
                corr, _ = spearmanr(convictions, returns)
                if corr != corr:  # NaN check
                    continue
                results.append({
                    "date":               entry["date"],
                    "ic":                 round(float(corr), 3),
                    "n_positions":        len(convictions),
                    "n_conviction_levels": len(set(convictions)),
                })
            except Exception:
                continue

        return results

    except Exception as exc:
        log.warning("[CalibrationTracker] get_per_rebalance_ic failed: %s", exc)
        return []
```

- [ ] **Step 4: Update `get_calibration_report()` to include per-rebalance section**

In `ascent/strategy/calibration_tracker.py`, `get_calibration_report()` builds a list of `lines`. At the end of the function, before `return "\n".join(lines)`, add the per-rebalance section:

```python
        # ── Per-rebalance cross-sectional IC ──────────────────────────────────
        per_rb = get_per_rebalance_ic(n_rebalances=n_rebalances)
        lines.append("")
        lines.append("  Per-rebalance cross-sectional IC (conviction → 21d within each date):")
        if not per_rb:
            lines.append("    No rebalances yet with multiple conviction levels and realized outcomes.")
        else:
            for rb in per_rb:
                interpretation = "✓" if rb["ic"] >= 0.10 else ("~" if rb["ic"] >= 0 else "✗")
                lines.append(
                    f"    {rb['date']}: IC {rb['ic']:+.3f} {interpretation}  "
                    f"({rb['n_positions']} positions, {rb['n_conviction_levels']} conviction levels)"
                )
            if len(per_rb) > 1:
                avg_ic = sum(r["ic"] for r in per_rb) / len(per_rb)
                lines.append(f"    Average per-rebalance IC: {avg_ic:+.3f}")

        return "\n".join(lines)
```

Note: the existing `return "\n".join(lines)` at the end of the function must be removed and replaced by the new one in this block. The old return is the last line of the try block.

- [ ] **Step 5: Run all new calibration tests**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/test_calibration_tracker.py::test_get_per_rebalance_ic_returns_empty_when_all_same_conviction tests/test_calibration_tracker.py::test_get_per_rebalance_ic_positive_when_high_conviction_wins tests/test_calibration_tracker.py::test_get_per_rebalance_ic_skips_entry_without_realized_outcomes tests/test_calibration_tracker.py::test_calibration_report_includes_per_rebalance_section -v 2>&1 | tail -15
```
Expected: 4 PASS.

- [ ] **Step 6: Full test sweep**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: no new failures.

- [ ] **Step 7: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add ascent/strategy/calibration_tracker.py tests/test_calibration_tracker.py && git commit -m "feat(ai-pm): add per-rebalance cross-sectional conviction IC to calibration report"
```

---

## Self-Review

**Spec coverage check:**

| Requirement | Task |
|-------------|------|
| Force-seal runs tagged so authority ladder can exclude them | Tasks 1+2+3 |
| `force_sealed` in AIPMResult | Task 1 |
| `force_sealed` propagated to Track D snapshot | Task 1 |
| Force-sealed days excluded from Sortino buffer | Task 3 |
| Delta portfolio Track Δ | Task 4 |
| Track Δ in daily log, cumulative report, printed | Task 4 |
| Per-rebalance cross-sectional IC | Task 5 |
| IC report updated | Task 5 |
| No new test failures | All tasks |
| Backward compat with old log records (missing new fields) | Task 3 (rebuild test), Task 4 (cumret_over handles None) |

**Placeholder scan:** No TBDs, no "similar to above", all code blocks complete.

**Type consistency:**
- `force_sealed: bool` flows: `AIPMResult` → `snapshot_ai_pm()` → snapshot record → `score_daily()` → daily record → `rebuild_buffers_from_counterfactual()`
- `_delta_portfolio_return()` signature matches its call sites in `score_daily()`
- `get_per_rebalance_ic()` called by `get_calibration_report()` — both in same module, no import issues
