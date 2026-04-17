# Phase 1 System Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three quick-win integrity bugs: skill score staleness in orchestrator, silent sector constraint failure, and one-day-only debate halt.

**Architecture:** Three independent fixes in three file clusters. Fix 1a touches the execution-order DAG in `run_all_agents.py` and adds a staleness guard in the orchestrator. Fix 1b adds a startup validation call and turns the optimizer's silent fallback into a hard raise. Fix 1c adds a persistent halt-state JSON and a `check_halt_state()` function that gates every rebalance.

**Tech Stack:** Python 3.12, stdlib only (json, pathlib, datetime, threading already in use). No new dependencies.

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `run_all_agents.py` | Modify | Reorder forward-PnL → skill-scores → orchestrator; add `validate_sector_data()` call at startup; add `check_halt_state()` call before debate; add `--skip-sector-check` flag |
| `ascent/monitoring/skill_tracker.py` | Modify | Add `skill_score_as_of` (ISO date string) to the output payload |
| `orchestrator/central_intelligence.py` | Modify | Add staleness check in `_load_skill_scores()` — fall back to empty dict if scores are >1 day stale |
| `ascent/portfolio/optimizer.py` | Modify | Add `SectorDataError` exception; replace silent fallback log with `raise SectorDataError` inside `sector_constrained_weighted()` |
| `debate/debate_runner.py` | Modify | Write `execution/halt_state.json` when verdict is `halt_and_review` |
| `tests/test_phase1_hardening.py` | Create | Unit tests for all three fixes |

---

## Task 1 — Fix 1a: Add `skill_score_as_of` to skill tracker output

**Files:**
- Modify: `ascent/monitoring/skill_tracker.py:131-145`
- Test: `tests/test_phase1_hardening.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_phase1_hardening.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

def test_skill_tracker_writes_as_of_date(tmp_path):
    """export_skill_scores() must write skill_score_as_of to the JSON payload."""
    from ascent.monitoring import skill_tracker

    # Patch paths to use tmp_path
    with patch.object(skill_tracker, "SKILL_OUTPUT_PATH", tmp_path / "agent_skill_scores.json"), \
         patch.object(skill_tracker, "SKILL_LOG_PATH", tmp_path / "skill_scores_log.jsonl"), \
         patch.object(skill_tracker, "compute_all_skill_scores", return_value={}):
        skill_tracker.export_skill_scores()

    payload = json.loads((tmp_path / "agent_skill_scores.json").read_text())
    assert "skill_score_as_of" in payload, "skill_score_as_of key missing from output"
    # Should be today's ISO date (YYYY-MM-DD)
    from datetime import date
    assert payload["skill_score_as_of"] == date.today().isoformat()
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_skill_tracker_writes_as_of_date -v
```

Expected: `FAILED` — `AssertionError: skill_score_as_of key missing from output`

- [ ] **Step 3: Add `skill_score_as_of` to the payload in `export_skill_scores()`**

In `ascent/monitoring/skill_tracker.py`, find the `payload = { ... }` block (line ~133) and add the field:

```python
    scores = compute_all_skill_scores(agent_ids)

    payload = {
        "generated_at":     datetime.now().isoformat(),
        "skill_score_as_of": datetime.now().date().isoformat(),   # ADD THIS LINE
        "agents":            scores,
    }
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_skill_tracker_writes_as_of_date -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
git add ascent/monitoring/skill_tracker.py tests/test_phase1_hardening.py
git commit -m "feat: add skill_score_as_of timestamp to skill tracker output"
```

---

## Task 2 — Fix 1a: Add staleness guard to orchestrator `_load_skill_scores()`

**Files:**
- Modify: `orchestrator/central_intelligence.py:138-151`
- Test: `tests/test_phase1_hardening.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_phase1_hardening.py`:

```python
import json
import tempfile
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

def test_orchestrator_rejects_stale_skill_scores(tmp_path):
    """_load_skill_scores() returns {} when skill_score_as_of is more than 1 day old."""
    from orchestrator import central_intelligence as ci

    stale_date = (date.today() - timedelta(days=3)).isoformat()
    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-09T14:00:00",
        "skill_score_as_of": stale_date,
        "agents": {
            "us_equities": {"skill_score": 1.23, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {}, f"Expected empty dict for stale scores, got {result}"


def test_orchestrator_accepts_fresh_skill_scores(tmp_path):
    """_load_skill_scores() returns scores when skill_score_as_of is today."""
    from orchestrator import central_intelligence as ci

    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-12T14:00:00",
        "skill_score_as_of": date.today().isoformat(),
        "agents": {
            "us_equities": {"skill_score": 1.23, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {"us_equities": 1.23}


def test_orchestrator_accepts_yesterday_scores_on_monday(tmp_path):
    """_load_skill_scores() returns scores when as_of is 1 day ago (weekend buffer)."""
    from orchestrator import central_intelligence as ci

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    scores_file = tmp_path / "agent_skill_scores.json"
    scores_file.write_text(json.dumps({
        "generated_at": "2026-04-12T14:00:00",
        "skill_score_as_of": yesterday,
        "agents": {
            "us_equities": {"skill_score": 0.80, "n_days": 63, "status": "active"}
        }
    }))

    with patch.object(ci, "SKILL_SCORES_PATH", scores_file):
        result = ci._load_skill_scores()

    assert result == {"us_equities": 0.80}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_orchestrator_rejects_stale_skill_scores tests/test_phase1_hardening.py::test_orchestrator_accepts_fresh_skill_scores tests/test_phase1_hardening.py::test_orchestrator_accepts_yesterday_scores_on_monday -v
```

Expected: all three `FAILED`

- [ ] **Step 3: Rewrite `_load_skill_scores()` in `orchestrator/central_intelligence.py`**

Replace the existing `_load_skill_scores()` function (lines 138–151) with:

```python
def _load_skill_scores() -> Dict[str, Optional[float]]:
    """
    Load latest skill scores from the dashboard JSON.
    Returns {} (triggers base allocation) if the file is missing or scores are
    more than 1 day stale — stale scores are worse than no scores.
    The 1-day buffer allows Friday scores to remain valid on Monday.
    """
    if not SKILL_SCORES_PATH.exists():
        return {}
    try:
        with open(SKILL_SCORES_PATH) as f:
            data = json.load(f)

        as_of = data.get("skill_score_as_of", "")
        if as_of:
            import pandas as pd
            today_str = date.today().isoformat()
            staleness_days = (pd.Timestamp(today_str) - pd.Timestamp(as_of)).days
            if staleness_days > 1:
                print(
                    f"[Orchestrator] Skill scores are {staleness_days}d stale "
                    f"(as_of={as_of}) — falling back to base allocation"
                )
                return {}

        agents = data.get("agents", {})
        return {
            agent_id: info.get("skill_score")
            for agent_id, info in agents.items()
        }
    except Exception:
        return {}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_orchestrator_rejects_stale_skill_scores tests/test_phase1_hardening.py::test_orchestrator_accepts_fresh_skill_scores tests/test_phase1_hardening.py::test_orchestrator_accepts_yesterday_scores_on_monday -v
```

Expected: all three `PASSED`

- [ ] **Step 5: Commit**

```bash
git add orchestrator/central_intelligence.py tests/test_phase1_hardening.py
git commit -m "feat: orchestrator rejects stale skill scores (>1 day), falls back to base allocation"
```

---

## Task 3 — Fix 1a: Reorder calls in `run_all_agents.py`

**Files:**
- Modify: `run_all_agents.py:109-123`

The current order is:
1. `run_forward_pnl_cycle(...)` — writes today's PnL log
2. `export_skill_scores()` — reads 63-day window (correct, depends on step 1)
3. `run_orchestrator(...)` — reads fresh scores (correct, depends on step 2)

Looking at the current code, steps 2 and 3 are already in the right sequential order (lines 110–122). However the comment on line 121 says `# ── Step 3: Run orchestrator` duplicating the step number from line 115. This is cosmetic and will be fixed, but more importantly we need to verify no parallelization is happening across these three calls.

- [ ] **Step 1: Verify the current ordering is correct**

```bash
grep -n "run_forward_pnl_cycle\|export_skill_scores\|run_orchestrator" run_all_agents.py
```

Expected output showing PnL cycle at ~line 111, skill scores at ~line 117, orchestrator at ~line 122 — in that order, with no `executor.submit()` wrapping them.

- [ ] **Step 2: Fix the duplicate step comment and add a clarifying comment about sequential dependency**

In `run_all_agents.py`, replace the comment block around lines 109–122:

```python
    # ── Steps 2/3/4: Sequential pipeline — DO NOT parallelize ───────────────
    # Each step depends on the previous: PnL log → skill scores → orchestrator.
    # run_forward_pnl_cycle writes today's NAV to the PnL log.
    # export_skill_scores reads that log to compute the 63-day rolling Sharpe.
    # run_orchestrator reads the fresh Sharpe to weight capital allocation.
    try:
        run_forward_pnl_cycle(agent_outputs, today=today)
    except Exception as e:
        print(f"[Runner] Forward PnL cycle failed: {e} — continuing")

    try:
        export_skill_scores()
    except Exception as e:
        print(f"[Runner] Skill score update failed: {e} — continuing with stale scores")

    # ── Step 5: Run orchestrator (reads fresh skill scores written above) ─────
    merged_weights = run_orchestrator(agent_outputs)
```

- [ ] **Step 3: Verify the file parses cleanly**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add run_all_agents.py
git commit -m "fix: add comment clarifying sequential dependency of PnL→skill→orchestrator pipeline"
```

---

## Task 4 — Fix 1b: Add `SectorDataError` to optimizer and replace silent fallback

**Files:**
- Modify: `ascent/portfolio/optimizer.py`
- Test: `tests/test_phase1_hardening.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_phase1_hardening.py`:

```python
import pandas as pd
import pytest

def test_sector_constrained_raises_on_low_coverage_at_construction_time():
    """
    sector_constrained_weighted() must raise SectorDataError (not silently degrade)
    when sector coverage of the candidate pool is below 80%.
    This mirrors what startup validation catches — but if for any reason
    portfolio construction sees bad data, it must fail loudly.
    """
    from ascent.portfolio.optimizer import sector_constrained_weighted, SectorDataError

    # Build a small alpha DataFrame (3 dates, 5 symbols)
    dates = pd.date_range("2026-01-01", periods=3, freq="B")
    syms = ["A", "B", "C", "D", "E"]
    alpha = pd.DataFrame(
        [[0.5, 0.4, 0.3, 0.2, 0.1]] * 3,
        index=dates,
        columns=syms,
    )

    # Provide a sector_map with < 80% coverage (only 3 of 5 = 60%)
    sector_map = {"A": "Tech", "B": "Health", "C": "Energy"}  # D, E unknown

    with pytest.raises(SectorDataError):
        sector_constrained_weighted(alpha, n=5, sector_map=sector_map)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_sector_constrained_raises_on_low_coverage_at_construction_time -v
```

Expected: `FAILED` — either `ImportError: cannot import name 'SectorDataError'` or the test fails because no exception is raised.

- [ ] **Step 3: Add `SectorDataError` and replace silent fallback in `ascent/portfolio/optimizer.py`**

At the top of the file, after the `from __future__ import annotations` line, add the exception class:

```python
class SectorDataError(RuntimeError):
    """Raised when sector coverage is below the 80% threshold required for safe portfolio construction."""
    pass
```

Then in `sector_constrained_weighted()`, find the block that handles low coverage (lines ~207–215):

```python
        if not use_sector_caps and not _sector_fallback_logged:
            print(
                f"[Optimizer] Sector coverage {coverage:.0%} < 80% on {dt.date()} — "
                "skipping sector caps, using plain rank weighting."
            )
            _sector_fallback_logged = True
```

Replace it with:

```python
        if not use_sector_caps:
            raise SectorDataError(
                f"sector_constrained_weighted(): sector coverage {coverage:.0%} < 80% "
                f"on {dt.date()}. This should have been caught by validate_sector_data() "
                f"at startup. Regenerate profiles.parquet or pass --skip-sector-check."
            )
```

Also remove the `_sector_fallback_logged = False` initialization a few lines above (it is now unused). Find:

```python
    _sector_fallback_logged = False
```

And delete that line.

- [ ] **Step 4: Run test to confirm it passes**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_sector_constrained_raises_on_low_coverage_at_construction_time -v
```

Expected: `PASSED`

- [ ] **Step 5: Verify the module imports cleanly**

```bash
.venv/bin/python -c "from ascent.portfolio.optimizer import sector_constrained_weighted, SectorDataError; print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add ascent/portfolio/optimizer.py tests/test_phase1_hardening.py
git commit -m "feat: SectorDataError replaces silent sector fallback in optimizer — hard fail on low coverage"
```

---

## Task 5 — Fix 1b: Add `validate_sector_data()` startup check to `run_all_agents.py`

**Files:**
- Modify: `run_all_agents.py`
- Test: `tests/test_phase1_hardening.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phase1_hardening.py`:

```python
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

def test_validate_sector_data_raises_when_profiles_missing(tmp_path):
    """validate_sector_data() must raise SectorDataError when profiles.parquet is absent."""
    import run_all_agents
    from ascent.portfolio.optimizer import SectorDataError

    symbols = ["AAPL", "MSFT", "GOOGL"]
    with patch("run_all_agents.has_data", return_value=False):
        with pytest.raises(SectorDataError, match="profiles.parquet missing"):
            run_all_agents.validate_sector_data(symbols)


def test_validate_sector_data_raises_on_low_coverage(tmp_path):
    """validate_sector_data() must raise when sector coverage < 80%."""
    import pandas as pd
    import run_all_agents
    from ascent.portfolio.optimizer import SectorDataError

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    # Only AAPL and MSFT have known sectors — 40% coverage
    profiles_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "sector": ["Tech", "Tech", None, None, None],
    })

    with patch("run_all_agents.has_data", return_value=True), \
         patch("run_all_agents.load_parquet", return_value=profiles_df):
        with pytest.raises(SectorDataError, match="Sector coverage"):
            run_all_agents.validate_sector_data(symbols)


def test_validate_sector_data_passes_on_good_coverage(tmp_path):
    """validate_sector_data() must not raise when coverage >= 80%."""
    import pandas as pd
    import run_all_agents

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    profiles_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        "sector": ["Tech", "Tech", "Tech", "Consumer", "Tech"],
    })

    with patch("run_all_agents.has_data", return_value=True), \
         patch("run_all_agents.load_parquet", return_value=profiles_df):
        # Should not raise
        run_all_agents.validate_sector_data(symbols)


def test_validate_sector_data_skip_flag_logs_and_returns(tmp_path):
    """validate_sector_data(skip=True) must log override and return without checking."""
    import run_all_agents

    log_path = tmp_path / "sector_override.jsonl"
    with patch("run_all_agents.SECTOR_OVERRIDE_LOG", log_path), \
         patch("run_all_agents.has_data") as mock_has_data:
        run_all_agents.validate_sector_data(symbols=["AAPL"], skip=True)

    # has_data should never have been called
    mock_has_data.assert_not_called()
    # Log entry should exist
    assert log_path.exists()
    entry = json.loads(log_path.read_text().strip())
    assert entry["action"] == "sector_check_skipped"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_validate_sector_data_raises_when_profiles_missing tests/test_phase1_hardening.py::test_validate_sector_data_raises_on_low_coverage tests/test_phase1_hardening.py::test_validate_sector_data_passes_on_good_coverage tests/test_phase1_hardening.py::test_validate_sector_data_skip_flag_logs_and_returns -v
```

Expected: all four `FAILED` with `AttributeError: module 'run_all_agents' has no attribute 'validate_sector_data'`

- [ ] **Step 3: Add imports and `validate_sector_data()` to `run_all_agents.py`**

At the top of `run_all_agents.py`, add these imports after the existing imports:

```python
from ascent.data.store.parquet import has_data, load_parquet
from ascent.portfolio.optimizer import SectorDataError
from ascent.config.settings import UniverseConfig
```

Add the module-level constant for the override log path (so tests can patch it):

```python
SECTOR_OVERRIDE_LOG = Path("logs/sector_override.jsonl")
```

Add the function before `main()`:

```python
def validate_sector_data(symbols: list, skip: bool = False) -> None:
    """
    Validates profiles.parquet exists and covers >= 80% of the US equities universe.
    Called once at startup before agents are spawned.
    Raises SectorDataError if coverage is insufficient.
    Pass skip=True (--skip-sector-check flag) to bypass with audit log entry.
    """
    import pandas as pd

    if skip:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "sector_check_skipped",
            "required_reason": "see CLI flag --skip-sector-check",
        }
        SECTOR_OVERRIDE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SECTOR_OVERRIDE_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print("[Startup] Sector check SKIPPED — override logged to logs/sector_override.jsonl")
        return

    if not has_data("profiles"):
        raise SectorDataError(
            "profiles.parquet missing. Regenerate with:\n"
            "  .venv/bin/python -m ascent.data.ingest.profiles\n"
            "Or bypass with --skip-sector-check (override is logged)."
        )

    profiles = load_parquet("profiles")
    known = set(profiles["symbol"].dropna())

    unknown_sectors = profiles[
        profiles["sector"].isna() | profiles["sector"].isin(["Unknown", "unknown", ""])
    ]["symbol"].tolist()

    missing_from_profiles = [s for s in symbols if s not in known]
    total_unknown = len(set(missing_from_profiles + unknown_sectors))
    coverage = 1.0 - total_unknown / len(symbols) if symbols else 1.0

    if coverage < 0.80:
        raise SectorDataError(
            f"Sector coverage {coverage:.1%} < 80% threshold.\n"
            f"Missing from profiles: {missing_from_profiles[:20]}"
            f"{'...' if len(missing_from_profiles) > 20 else ''}\n"
            f"Unknown sectors: {unknown_sectors[:10]}"
            f"{'...' if len(unknown_sectors) > 10 else ''}\n"
            "Regenerate profiles.parquet or use --skip-sector-check (override is logged)."
        )

    print(f"[Startup] Sector data valid — coverage {coverage:.1%} ({len(known)} symbols in profiles)")
```

Then in `main()`, add the call at the very top of the function body (before the print banner and before `ThreadPoolExecutor`):

```python
def main():
    dry_run             = "--dry-run" in sys.argv
    skip_sector_check   = "--skip-sector-check" in sys.argv
    today               = date.today()

    # ── Startup validation: sector data must be present before agents spawn ───
    us_symbols = UniverseConfig().symbols
    validate_sector_data(us_symbols, skip=skip_sector_check)
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_validate_sector_data_raises_when_profiles_missing tests/test_phase1_hardening.py::test_validate_sector_data_raises_on_low_coverage tests/test_phase1_hardening.py::test_validate_sector_data_passes_on_good_coverage tests/test_phase1_hardening.py::test_validate_sector_data_skip_flag_logs_and_returns -v
```

Expected: all four `PASSED`

- [ ] **Step 5: Verify the file parses cleanly**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add run_all_agents.py tests/test_phase1_hardening.py
git commit -m "feat: validate_sector_data() startup check — aborts if profiles.parquet missing or coverage <80%"
```

---

## Task 6 — Fix 1c: Persistent halt state — write `halt_state.json` from debate runner

**Files:**
- Modify: `debate/debate_runner.py`
- Test: `tests/test_phase1_hardening.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_phase1_hardening.py`:

```python
def test_debate_runner_writes_halt_state_on_halt_verdict(tmp_path):
    """
    run_debate() must write execution/halt_state.json when verdict is halt_and_review.
    File must contain: halted=True, halt_date, reason, key_risks, verdict_path, requires_override.
    """
    from debate import debate_runner
    from unittest.mock import patch, MagicMock
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    verdict = {
        "recommendation": "halt_and_review",
        "confidence": 0.85,
        "key_risks": ["Energy concentration at 38%", "Oil shock risk"],
        "reasoning": "Too much energy exposure in volatile macro environment",
    }

    portfolio_state = {
        "date": "2026-04-15",
        "us_regime": "stressed",
        "macro_regime": "unknown",
        "n_positions": 12,
        "allocation": {},
        "weights": {"XLE": 0.20, "MPC": 0.18},
    }

    with patch.object(debate_runner, "HALT_STATE_PATH", halt_path), \
         patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull arg"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear arg"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil arg"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime arg"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant ok"), \
         patch("debate.debate_runner.run_judge", return_value=verdict), \
         patch("debate.debate_runner.DEBATE_LOG_DIR", tmp_path):
        result = debate_runner.run_debate(portfolio_state, run_date=date(2026, 4, 15))

    assert halt_path.exists(), "halt_state.json was not written"
    state = json.loads(halt_path.read_text())
    assert state["halted"] is True
    assert state["halt_date"] == "2026-04-15"
    assert state["requires_override"] is True
    assert "key_risks" in state
    assert len(state["key_risks"]) == 2


def test_debate_runner_does_not_write_halt_state_on_proceed(tmp_path):
    """run_debate() must NOT write halt_state.json when verdict is proceed."""
    from debate import debate_runner
    from unittest.mock import patch
    from datetime import date

    halt_path = tmp_path / "halt_state.json"
    verdict = {
        "recommendation": "proceed",
        "confidence": 0.70,
        "key_risks": [],
        "reasoning": "Portfolio looks fine",
    }
    portfolio_state = {
        "date": "2026-04-15",
        "us_regime": "calm_bull",
        "macro_regime": "unknown",
        "n_positions": 10,
        "allocation": {},
        "weights": {"AAPL": 0.10},
    }

    with patch.object(debate_runner, "HALT_STATE_PATH", halt_path), \
         patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull arg"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear arg"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil arg"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime arg"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant ok"), \
         patch("debate.debate_runner.run_judge", return_value=verdict), \
         patch("debate.debate_runner.DEBATE_LOG_DIR", tmp_path):
        debate_runner.run_debate(portfolio_state, run_date=date(2026, 4, 15))

    assert not halt_path.exists(), "halt_state.json should NOT be written for proceed verdict"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_debate_runner_writes_halt_state_on_halt_verdict tests/test_phase1_hardening.py::test_debate_runner_does_not_write_halt_state_on_proceed -v
```

Expected: both `FAILED` — `AttributeError: module 'debate.debate_runner' has no attribute 'HALT_STATE_PATH'`

- [ ] **Step 3: Add `HALT_STATE_PATH` constant and halt-write logic to `debate/debate_runner.py`**

Add the constant near the top of the file, after the existing `DEBATE_LOG_DIR` constant:

```python
DEBATE_LOG_DIR   = Path("outputs/debate_log")
MULTI_AGENT_LOG  = Path("logs/multi_agent_run.jsonl")
HALT_STATE_PATH  = Path("execution/halt_state.json")   # ADD THIS LINE
```

At the end of `run_debate()`, just before the `return verdict` line, add:

```python
    # Write persistent halt state if verdict requires human override
    if verdict.get("recommendation") == "halt_and_review":
        halt_record = {
            "halted":          True,
            "halt_date":       run_date.isoformat(),
            "reason":          verdict.get("reasoning", "")[:200],
            "key_risks":       verdict.get("key_risks", []),
            "verdict_path":    str(out_path),
            "requires_override": True,
            "created_at":      datetime.now().isoformat(),
        }
        HALT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        HALT_STATE_PATH.write_text(json.dumps(halt_record, indent=2))
        print(f"[Debate] HALT state written to {HALT_STATE_PATH} — "
              "create execution/halt_override.json to resume trading")

    return verdict
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_debate_runner_writes_halt_state_on_halt_verdict tests/test_phase1_hardening.py::test_debate_runner_does_not_write_halt_state_on_proceed -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add debate/debate_runner.py tests/test_phase1_hardening.py
git commit -m "feat: debate runner writes execution/halt_state.json on halt_and_review verdict"
```

---

## Task 7 — Fix 1c: Add `check_halt_state()` to `run_all_agents.py`

**Files:**
- Modify: `run_all_agents.py`
- Test: `tests/test_phase1_hardening.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_phase1_hardening.py`:

```python
def test_check_halt_state_returns_true_when_no_halt_file(tmp_path):
    """check_halt_state() returns True (proceed) when halt_state.json does not exist."""
    import run_all_agents

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date.today())

    assert result is True


def test_check_halt_state_returns_false_when_halted_no_override(tmp_path):
    """check_halt_state() returns False and prints a clear error when halted without override."""
    import run_all_agents

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"
    halt_path.write_text(json.dumps({
        "halted": True,
        "halt_date": "2026-04-15",
        "reason": "Too much energy exposure",
        "key_risks": ["Energy at 38%"],
        "verdict_path": "outputs/debate_log/verdict_2026-04-15.json",
        "requires_override": True,
        "created_at": "2026-04-15T13:47:22",
    }))

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date(2026, 4, 16))

    assert result is False


def test_check_halt_state_clears_files_on_valid_override(tmp_path):
    """check_halt_state() returns True and deletes both files when override is valid."""
    import run_all_agents

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"

    halt_path.write_text(json.dumps({
        "halted": True,
        "halt_date": "2026-04-15",
        "reason": "Too much energy",
        "key_risks": [],
        "verdict_path": "outputs/debate_log/verdict_2026-04-15.json",
        "requires_override": True,
        "created_at": "2026-04-15T13:47:22",
    }))
    override_path.write_text(json.dumps({
        "override_date": "2026-04-16",
        "override_by": "scott",
        "reason": "Reviewed — acceptable",
        "acknowledged_risks": [],
    }))

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date(2026, 4, 16))

    assert result is True
    assert not halt_path.exists(), "halt_state.json should be deleted after valid override"
    assert not override_path.exists(), "halt_override.json should be deleted after valid override"


def test_check_halt_state_blocks_override_predating_halt(tmp_path):
    """check_halt_state() returns False when override_date < halt_date."""
    import run_all_agents

    halt_path = tmp_path / "halt_state.json"
    override_path = tmp_path / "halt_override.json"

    halt_path.write_text(json.dumps({
        "halted": True,
        "halt_date": "2026-04-15",
        "reason": "energy",
        "key_risks": [],
        "verdict_path": "x",
        "requires_override": True,
        "created_at": "2026-04-15T13:47:22",
    }))
    override_path.write_text(json.dumps({
        "override_date": "2026-04-14",   # predates halt
        "override_by": "scott",
        "reason": "Stale override",
        "acknowledged_risks": [],
    }))

    with patch.object(run_all_agents, "HALT_STATE_PATH", halt_path), \
         patch.object(run_all_agents, "HALT_OVERRIDE_PATH", override_path):
        result = run_all_agents.check_halt_state(today=date(2026, 4, 16))

    assert result is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_check_halt_state_returns_true_when_no_halt_file tests/test_phase1_hardening.py::test_check_halt_state_returns_false_when_halted_no_override tests/test_phase1_hardening.py::test_check_halt_state_clears_files_on_valid_override tests/test_phase1_hardening.py::test_check_halt_state_blocks_override_predating_halt -v
```

Expected: all four `FAILED`

- [ ] **Step 3: Add constants and `check_halt_state()` to `run_all_agents.py`**

Add two path constants near the top of `run_all_agents.py` (with `SECTOR_OVERRIDE_LOG`):

```python
SECTOR_OVERRIDE_LOG = Path("logs/sector_override.jsonl")
HALT_STATE_PATH     = Path("execution/halt_state.json")
HALT_OVERRIDE_PATH  = Path("execution/halt_override.json")
```

Add the function before `main()`:

```python
def check_halt_state(today=None) -> bool:
    """
    Returns True if execution may proceed, False if halted.

    Halt is cleared only when a valid halt_override.json is present with
    override_date >= halt_date. Both files are deleted on successful clear.
    Agents and orchestrator still run during a halt — only execution is blocked.
    """
    from datetime import date as _date
    today = today or _date.today()

    if not HALT_STATE_PATH.exists():
        return True

    halt = json.loads(HALT_STATE_PATH.read_text())

    if not halt.get("requires_override", True):
        HALT_STATE_PATH.unlink(missing_ok=True)
        return True

    if not HALT_OVERRIDE_PATH.exists():
        print(
            f"[HALT] System halted since {halt['halt_date']}: {halt.get('reason', '')}\n"
            f"[HALT] Create execution/halt_override.json to resume trading.\n"
            f"[HALT] See verdict: {halt.get('verdict_path', 'outputs/debate_log/')}"
        )
        return False

    override = json.loads(HALT_OVERRIDE_PATH.read_text())

    if override.get("override_date", "") < halt.get("halt_date", ""):
        print(
            f"[HALT] Override date {override['override_date']} predates "
            f"halt date {halt['halt_date']} — invalid override. Recreate the file."
        )
        return False

    # Valid override — clear both files
    print(f"[HALT] Override accepted by {override.get('override_by', 'unknown')} — halt cleared. "
          "NOTE: today's debate may still issue a new halt.")
    HALT_STATE_PATH.unlink(missing_ok=True)
    HALT_OVERRIDE_PATH.unlink(missing_ok=True)
    return True
```

Then in `main()`, add the halt check on rebalance days, just before the debate block (after the checklist block and the merged weights write):

```python
    # ── Rebalance day: check for active halt before debating ─────────────────
    if is_rebalance:
        if not check_halt_state(today=today):
            print("[Runner] Halted — agents ran, weights updated, execution skipped.")
            print("[Runner] Create execution/halt_override.json to resume.")
            _log_run(today, merged_weights, agent_outputs, dry_run)
            return
```

This block must go after the merged weights write but before the `# ── Rebalance day: debate → execute ───` comment.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py::test_check_halt_state_returns_true_when_no_halt_file tests/test_phase1_hardening.py::test_check_halt_state_returns_false_when_halted_no_override tests/test_phase1_hardening.py::test_check_halt_state_clears_files_on_valid_override tests/test_phase1_hardening.py::test_check_halt_state_blocks_override_predating_halt -v
```

Expected: all four `PASSED`

- [ ] **Step 5: Verify the file parses cleanly**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add run_all_agents.py tests/test_phase1_hardening.py
git commit -m "feat: check_halt_state() gates rebalance execution — halt persists until human override file"
```

---

## Task 8 — Full test suite run and final verification

- [ ] **Step 1: Run all Phase 1 tests together**

```bash
.venv/bin/python -m pytest tests/test_phase1_hardening.py -v
```

Expected: all tests `PASSED`. Count should be 14 tests.

- [ ] **Step 2: Verify no existing tests broken**

```bash
.venv/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: no new failures introduced.

- [ ] **Step 3: Smoke test — verify run_all_agents.py imports without error**

```bash
.venv/bin/python -c "
import run_all_agents
print('validate_sector_data:', callable(run_all_agents.validate_sector_data))
print('check_halt_state:', callable(run_all_agents.check_halt_state))
print('HALT_STATE_PATH:', run_all_agents.HALT_STATE_PATH)
print('SECTOR_OVERRIDE_LOG:', run_all_agents.SECTOR_OVERRIDE_LOG)
print('OK')
"
```

Expected: all four lines print with correct values, final `OK`.

- [ ] **Step 4: Smoke test — verify optimizer raises on import**

```bash
.venv/bin/python -c "
from ascent.portfolio.optimizer import SectorDataError, sector_constrained_weighted
print('SectorDataError:', SectorDataError)
print('sector_constrained_weighted:', callable(sector_constrained_weighted))
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 5: Smoke test — verify debate runner has HALT_STATE_PATH**

```bash
.venv/bin/python -c "
from debate.debate_runner import HALT_STATE_PATH
print('HALT_STATE_PATH:', HALT_STATE_PATH)
print('OK')
"
```

Expected: `OK`.

- [ ] **Step 6: Final commit — update CLAUDE.md session log**

Add to CLAUDE.md session log:

```
### 2026-04-12
- Phase 1 system hardening complete: skill score staleness guard, sector constraint hard fail, persistent debate halt state
- Fix 1a: skill_tracker writes skill_score_as_of; orchestrator rejects scores >1 day stale; run_all_agents sequential dependency clarified
- Fix 1b: SectorDataError replaces silent fallback in optimizer; validate_sector_data() startup check added to run_all_agents.py
- Fix 1c: debate_runner writes execution/halt_state.json on halt_and_review; check_halt_state() in run_all_agents.py gates rebalance execution
- Files: ascent/monitoring/skill_tracker.py, orchestrator/central_intelligence.py, run_all_agents.py, ascent/portfolio/optimizer.py, debate/debate_runner.py, tests/test_phase1_hardening.py
- Open: Phase 2 (async approval gate, Almgren-Chriss cost model)
```

```bash
git add CLAUDE.md
git commit -m "docs: CLAUDE.md session log — Phase 1 hardening complete"
```
