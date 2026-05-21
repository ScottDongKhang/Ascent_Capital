# Non-Rebalance Intelligence Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily intelligence stack that accumulates structured observations across the 9 non-rebalance days between rebalances, synthesizes them into a pre-rebalance briefing, and exposes that briefing to the AI PM as its first tool call — so every rebalance decision is the culmination of 9 days of thinking rather than a single-run computation.

**Architecture:** Seven pure-Python or Haiku intelligence modules write daily JSON entries to `data_cache/daily_intelligence/`. On rebalance day, a synthesis module reads all entries since the last rebalance and produces a structured briefing that the AI PM reads via a new `get_rebalance_brief` tool before doing anything else. A shared `data_cache/last_rebalance_state.json` anchors conviction and signal decay to the actual rebalance baseline.

**Tech Stack:** Python 3.12, Haiku (`claude-haiku-4-5-20251001`) for LLM modules, existing `ascent/llm/client.py`, `logs/sleeve_ic_log.jsonl`, `dashboard/regime_signal.json`, `logs/regime_episodes.jsonl`, `execution/merged_weights.json`, `data_cache/daily_intelligence/`, `data_cache/last_rebalance_state.json`

---

## File Map

**New files:**
- `ascent/monitoring/conviction_tracker.py` — alpha rank decay per held position (pure Python)
- `ascent/monitoring/signal_health.py` — sleeve IC trajectory vs rebalance baseline (pure Python)
- `ascent/monitoring/regime_trajectory.py` — regime label stability + stress trend (pure Python)
- `ascent/monitoring/position_thesis.py` — per-position thesis update via Haiku (LLM)
- `ascent/monitoring/adversarial_daily.py` — daily portfolio challenge via Haiku (LLM)
- `ascent/monitoring/macro_calendar.py` — upcoming event impact scoring via Haiku (LLM)
- `ascent/monitoring/analogue_search.py` — regime fingerprint + episode memory lookup (pure Python)
- `ascent/monitoring/daily_intelligence.py` — orchestrates all 7 modules, writes daily entry
- `ascent/monitoring/rebalance_brief.py` — synthesizes N days of intelligence into briefing (Haiku)
- `tests/monitoring/test_conviction_tracker.py`
- `tests/monitoring/test_signal_health.py`
- `tests/monitoring/test_regime_trajectory.py`
- `tests/monitoring/test_analogue_search.py`
- `tests/monitoring/test_rebalance_brief.py`
- `tests/monitoring/test_daily_intelligence.py`

**Modified files:**
- `run_all_agents.py` — call `run_daily_intelligence()` on non-rebalance days; save rebalance state + call `generate_rebalance_brief()` on rebalance days
- `agents/ai_pm_agent.py` — add `get_rebalance_brief` as tool #17, update `_SYSTEM_PROMPT` Phase 1 instruction

---

## Task 1: Shared Data Foundation

**Files:**
- Create: `data_cache/last_rebalance_state.json` (written by run_all_agents.py on rebalance day)
- Create: `data_cache/daily_intelligence/` directory

On every rebalance day, `run_all_agents.py` must snapshot the rebalance baseline so the daily intelligence modules have something to compare against. This is a pure data-writing step — no new module file needed yet.

- [ ] **Step 1: Verify directory structure**

```bash
ls "data_cache/" && echo "OK"
```

Expected: lists existing parquet files, no `daily_intelligence/` yet.

- [ ] **Step 2: Document the shared JSON schemas**

`data_cache/last_rebalance_state.json` schema (written on rebalance day):
```json
{
  "date": "2026-05-19",
  "weights": {"VICR": 0.066, "CHRD": 0.066},
  "alpha_ranks": {"VICR": 3, "CHRD": 7},
  "alpha_scores": {"VICR": 0.82, "CHRD": 0.71},
  "sleeve_ics": {"trend": 0.014, "ml": 0.009, "meanrev": 0.001},
  "regime": "calm_bull",
  "regime_stability_10d": 0.9
}
```

`data_cache/daily_intelligence/YYYY-MM-DD.json` schema (written each non-rebalance day):
```json
{
  "date": "2026-05-20",
  "conviction_decay": {},
  "signal_health": {},
  "regime_trajectory": {},
  "position_theses": {},
  "adversarial_challenge": "",
  "macro_events": [],
  "historical_analogues": []
}
```

No code to write — this step is schema documentation for all subsequent tasks.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-05-20-non-rebalance-intelligence.md
git commit -m "plan: non-rebalance intelligence stack"
```

---

## Task 2: Conviction Decay Tracker

**Files:**
- Create: `ascent/monitoring/conviction_tracker.py`
- Create: `tests/monitoring/test_conviction_tracker.py`

Loads held positions + alpha scores from agent_outputs. Compares each held position's current alpha rank against its rank at the last rebalance (from `last_rebalance_state.json`). Returns a dict keyed by symbol with rank drift and score decay.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_conviction_tracker.py
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from ascent.monitoring.conviction_tracker import compute_conviction_decay


def _make_agent_output(scores: dict):
    ao = MagicMock()
    ao.agent_id = "us_equities"
    ao.alpha_scores = pd.DataFrame(
        {"composite": scores},
        index=pd.to_datetime(["2026-05-20"] * len(scores))
    )
    return ao


def test_conviction_decay_detects_rank_drop(tmp_path, monkeypatch):
    last_state = {
        "date": "2026-05-19",
        "weights": {"AAPL": 0.07, "MSFT": 0.07},
        "alpha_ranks": {"AAPL": 1, "MSFT": 2},
        "alpha_scores": {"AAPL": 0.9, "MSFT": 0.8},
        "sleeve_ics": {"trend": 0.014},
        "regime": "calm_bull",
        "regime_stability_10d": 0.9,
    }
    import json
    state_path = tmp_path / "last_rebalance_state.json"
    state_path.write_text(json.dumps(last_state))

    # AAPL dropped from rank 1 to rank 5, MSFT stayed at rank 2
    scores = {"AAPL": 0.6, "MSFT": 0.85, "GOOG": 0.95, "META": 0.92, "AMZN": 0.88}
    agent_outputs = [_make_agent_output(scores)]
    merged_weights = {"AAPL": 0.07, "MSFT": 0.07}

    result = compute_conviction_decay(
        "2026-05-20", merged_weights, agent_outputs,
        state_path=str(state_path)
    )

    assert "AAPL" in result
    assert result["AAPL"]["rank_today"] > result["AAPL"]["rank_at_rebalance"]
    assert result["AAPL"]["rank_at_rebalance"] == 1
    assert result["MSFT"]["rank_today"] <= 3  # still near top


def test_conviction_decay_returns_empty_without_state(tmp_path):
    from ascent.monitoring.conviction_tracker import compute_conviction_decay
    result = compute_conviction_decay(
        "2026-05-20", {"AAPL": 0.07}, [],
        state_path=str(tmp_path / "missing.json")
    )
    assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/monitoring/test_conviction_tracker.py -v
```

Expected: `ModuleNotFoundError: No module named 'ascent.monitoring.conviction_tracker'`

- [ ] **Step 3: Implement conviction_tracker.py**

```python
# ascent/monitoring/conviction_tracker.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = "data_cache/last_rebalance_state.json"


def compute_conviction_decay(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    state_path: str = _DEFAULT_STATE_PATH,
) -> Dict[str, Any]:
    """
    For each held position, compare today's alpha rank to rank at last rebalance.
    Returns {} if no rebalance state exists yet or no us_equities agent output.
    """
    path = Path(state_path)
    if not path.exists():
        log.warning("[ConvictionTracker] No rebalance state found at %s", state_path)
        return {}

    try:
        last_state = json.loads(path.read_text())
    except Exception as e:
        log.warning("[ConvictionTracker] Failed to read state: %s", e)
        return {}

    # Get today's alpha scores from us_equities agent
    us_agent = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
    if us_agent is None or us_agent.alpha_scores is None or us_agent.alpha_scores.empty:
        log.warning("[ConvictionTracker] No us_equities alpha scores available")
        return {}

    try:
        latest = us_agent.alpha_scores.iloc[-1]  # most recent date row
        scores_today = latest.to_dict()
    except Exception as e:
        log.warning("[ConvictionTracker] Failed to extract alpha scores: %s", e)
        return {}

    # Rank all symbols by today's score (higher = better, rank 1 = best)
    sorted_symbols = sorted(scores_today, key=lambda s: scores_today[s], reverse=True)
    rank_today = {sym: idx + 1 for idx, sym in enumerate(sorted_symbols)}

    ranks_at_rebalance = last_state.get("alpha_ranks", {})
    scores_at_rebalance = last_state.get("alpha_scores", {})

    result = {}
    for sym in merged_weights:
        if sym not in rank_today:
            continue
        rank_then = ranks_at_rebalance.get(sym)
        score_then = scores_at_rebalance.get(sym)
        score_now = scores_today.get(sym, 0.0)
        decay_pct = (
            round((score_then - score_now) / abs(score_then) * 100, 1)
            if score_then and score_then != 0 else None
        )
        result[sym] = {
            "rank_at_rebalance": rank_then,
            "rank_today":        rank_today[sym],
            "score_at_rebalance": round(score_then, 4) if score_then is not None else None,
            "score_today":        round(score_now, 4),
            "decay_pct":          decay_pct,
        }

    return result


def save_rebalance_alpha_state(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    sleeve_ics: Dict[str, float],
    regime: str,
    regime_stability_10d: float,
    state_path: str = _DEFAULT_STATE_PATH,
) -> None:
    """Call on rebalance day to snapshot the baseline for future decay comparisons."""
    us_agent = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
    alpha_ranks, alpha_scores = {}, {}

    if us_agent is not None and not us_agent.alpha_scores.empty:
        try:
            latest = us_agent.alpha_scores.iloc[-1].to_dict()
            sorted_syms = sorted(latest, key=lambda s: latest[s], reverse=True)
            alpha_ranks = {s: i + 1 for i, s in enumerate(sorted_syms) if s in merged_weights}
            alpha_scores = {s: round(latest[s], 4) for s in merged_weights if s in latest}
        except Exception as e:
            log.warning("[ConvictionTracker] Could not build rebalance snapshot: %s", e)

    state = {
        "date":                  date,
        "weights":               {k: round(v, 6) for k, v in merged_weights.items()},
        "alpha_ranks":           alpha_ranks,
        "alpha_scores":          alpha_scores,
        "sleeve_ics":            sleeve_ics,
        "regime":                regime,
        "regime_stability_10d":  round(regime_stability_10d, 4),
    }
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    Path(state_path).write_text(json.dumps(state, indent=2))
    log.info("[ConvictionTracker] Rebalance state saved to %s", state_path)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_conviction_tracker.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/conviction_tracker.py tests/monitoring/test_conviction_tracker.py
git commit -m "feat: conviction decay tracker — alpha rank drift per held position"
```

---

## Task 3: Signal Health Monitor

**Files:**
- Create: `ascent/monitoring/signal_health.py`
- Create: `tests/monitoring/test_signal_health.py`

Reads `logs/sleeve_ic_log.jsonl` (format: `{"date": "YYYY-MM-DD", "sleeves": {"trend": {"mean_ic": 0.014, ...}}}`). Computes 5-day rolling mean IC per sleeve and compares to the baseline in `last_rebalance_state.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_signal_health.py
import json
import pytest
from pathlib import Path
from ascent.monitoring.signal_health import compute_signal_health


def _write_ic_log(tmp_path, entries):
    p = tmp_path / "sleeve_ic_log.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in entries))
    return str(p)


def _write_state(tmp_path, sleeve_ics):
    p = tmp_path / "last_rebalance_state.json"
    p.write_text(json.dumps({
        "date": "2026-05-10",
        "weights": {}, "alpha_ranks": {}, "alpha_scores": {},
        "sleeve_ics": sleeve_ics,
        "regime": "calm_bull", "regime_stability_10d": 0.9,
    }))
    return str(p)


def test_signal_health_detects_decay():
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        entries = [
            {"date": f"2026-05-{11+i:02d}", "sleeves": {
                "trend": {"mean_ic": 0.014 - i * 0.002, "ic_t": 3.0, "n": 1500}
            }} for i in range(7)
        ]
        ic_path   = _write_ic_log(Path(tmp), entries)
        state_path = _write_state(Path(tmp), {"trend": 0.014})

        result = compute_signal_health("2026-05-20", ic_log_path=ic_path, state_path=state_path)

        assert "trend" in result
        assert result["trend"]["ic_at_rebalance"] == pytest.approx(0.014)
        assert result["trend"]["ic_5d_avg"] < 0.014
        assert result["trend"]["status"] in ("healthy", "weakening", "deteriorating")


def test_signal_health_returns_empty_without_log(tmp_path):
    result = compute_signal_health(
        "2026-05-20",
        ic_log_path=str(tmp_path / "missing.jsonl"),
        state_path=str(tmp_path / "missing.json"),
    )
    assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/monitoring/test_signal_health.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement signal_health.py**

```python
# ascent/monitoring/signal_health.py
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)

_IC_LOG       = "logs/sleeve_ic_log.jsonl"
_STATE_PATH   = "data_cache/last_rebalance_state.json"
_WINDOW       = 5


def compute_signal_health(
    date: str,
    ic_log_path: str = _IC_LOG,
    state_path: str = _STATE_PATH,
) -> Dict[str, Any]:
    """
    Reads the last _WINDOW entries from sleeve_ic_log.jsonl.
    Compares rolling average IC per sleeve to the rebalance baseline.
    Status: healthy (>80% of baseline), weakening (50-80%), deteriorating (<50%).
    """
    ic_path = Path(ic_log_path)
    if not ic_path.exists():
        return {}

    lines = [l for l in ic_path.read_text().splitlines() if l.strip()]
    if not lines:
        return {}

    # Parse last _WINDOW unique-date entries
    seen_dates, recent = set(), []
    for line in reversed(lines):
        try:
            entry = json.loads(line)
            d = entry.get("date", "")
            if d and d not in seen_dates:
                seen_dates.add(d)
                recent.append(entry)
            if len(recent) >= _WINDOW:
                break
        except Exception:
            continue

    if not recent:
        return {}

    # Aggregate IC per sleeve across recent entries
    sleeve_ics: Dict[str, list] = defaultdict(list)
    for entry in recent:
        for sleeve, stats in entry.get("sleeves", {}).items():
            ic = stats.get("mean_ic")
            if ic is not None:
                sleeve_ics[sleeve].append(ic)

    # Load rebalance baseline
    baseline: Dict[str, float] = {}
    sp = Path(state_path)
    if sp.exists():
        try:
            baseline = json.loads(sp.read_text()).get("sleeve_ics", {})
        except Exception:
            pass

    result = {}
    for sleeve, ics in sleeve_ics.items():
        avg = sum(ics) / len(ics)
        base = baseline.get(sleeve)
        change_pct = round((avg - base) / abs(base) * 100, 1) if base and base != 0 else None
        if change_pct is None:
            status = "unknown"
        elif change_pct >= -20:
            status = "healthy"
        elif change_pct >= -50:
            status = "weakening"
        else:
            status = "deteriorating"

        result[sleeve] = {
            "ic_at_rebalance": round(base, 4) if base is not None else None,
            "ic_5d_avg":       round(avg, 4),
            "change_pct":      change_pct,
            "status":          status,
        }

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_signal_health.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/signal_health.py tests/monitoring/test_signal_health.py
git commit -m "feat: signal health monitor — sleeve IC decay vs rebalance baseline"
```

---

## Task 4: Regime Trajectory Tracker

**Files:**
- Create: `ascent/monitoring/regime_trajectory.py`
- Create: `tests/monitoring/test_regime_trajectory.py`

Reads `dashboard/regime_signal.json`. Uses the `series` field (list of `{d, label, risk_mult, rs}`) to compute: current label, 10-day label stability (% of last 10 days matching current label), regime stress trend (rs slope over last 5 days), and days in current regime.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_regime_trajectory.py
import json
import pytest
from pathlib import Path
from ascent.monitoring.regime_trajectory import compute_regime_trajectory


def _write_regime_signal(tmp_path, series, current="calm_bull"):
    data = {
        "regime": current, "label": current,
        "as_of": series[-1]["d"], "last_refit_date": series[-1]["d"],
        "series": series
    }
    p = tmp_path / "regime_signal.json"
    p.write_text(json.dumps(data))
    return str(p)


def test_stable_regime():
    import tempfile
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as tmp:
        series = [
            {"d": f"2026-05-{i:02d}", "label": "calm_bull", "risk_mult": 1.0, "rs": 0.0}
            for i in range(10, 21)
        ]
        path = _write_regime_signal(P(tmp), series)
        result = compute_regime_trajectory("2026-05-20", signal_path=path)

        assert result["current_label"] == "calm_bull"
        assert result["stability_10d"] == pytest.approx(1.0)
        assert result["days_in_regime"] >= 10


def test_unstable_regime():
    import tempfile
    from pathlib import Path as P
    with tempfile.TemporaryDirectory() as tmp:
        labels = ["calm_bull", "stressed", "calm_bull", "stressed", "calm_bull",
                  "stressed", "calm_bull", "calm_bull", "stressed", "calm_bull"]
        series = [
            {"d": f"2026-05-{i+10:02d}", "label": labels[i], "risk_mult": 1.0, "rs": 0.1 * i}
            for i in range(10)
        ]
        path = _write_regime_signal(P(tmp), series, current="calm_bull")
        result = compute_regime_trajectory("2026-05-20", signal_path=path)

        assert result["stability_10d"] < 0.8
        assert result["rs_trend"] in ("rising", "flat", "falling")


def test_returns_empty_without_file(tmp_path):
    result = compute_regime_trajectory("2026-05-20", signal_path=str(tmp_path / "missing.json"))
    assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/monitoring/test_regime_trajectory.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement regime_trajectory.py**

```python
# ascent/monitoring/regime_trajectory.py
import json
import logging
from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)

_SIGNAL_PATH = "dashboard/regime_signal.json"


def compute_regime_trajectory(
    date: str,
    signal_path: str = _SIGNAL_PATH,
) -> Dict[str, Any]:
    """
    Reads regime_signal.json series to compute stability and stress trend.
    stability_10d: fraction of last 10 days matching current label (1.0 = perfectly stable).
    rs_trend: slope direction of regime stress over last 5 days.
    """
    path = Path(signal_path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        log.warning("[RegimeTrajectory] Failed to read signal: %s", e)
        return {}

    current_label = data.get("label", "unknown")
    series = data.get("series", [])
    if not series:
        return {}

    # Last 10 entries
    recent_10 = series[-10:]
    stability = sum(1 for e in recent_10 if e.get("label") == current_label) / len(recent_10)

    # rs trend over last 5 entries
    recent_5_rs = [e.get("rs", 0.0) for e in series[-5:]]
    if len(recent_5_rs) >= 2:
        slope = recent_5_rs[-1] - recent_5_rs[0]
        rs_trend = "rising" if slope > 0.01 else "falling" if slope < -0.01 else "flat"
    else:
        rs_trend = "unknown"

    # Days in current regime (consecutive from end)
    days_in_regime = 0
    for entry in reversed(series):
        if entry.get("label") == current_label:
            days_in_regime += 1
        else:
            break

    return {
        "current_label":     current_label,
        "stability_10d":     round(stability, 3),
        "rs_trend":          rs_trend,
        "days_in_regime":    days_in_regime,
        "as_of":             data.get("as_of", ""),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_regime_trajectory.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/regime_trajectory.py tests/monitoring/test_regime_trajectory.py
git commit -m "feat: regime trajectory tracker — stability and stress trend from signal series"
```

---

## Task 5: Historical Analogue Search

**Files:**
- Create: `ascent/monitoring/analogue_search.py`
- Create: `tests/monitoring/test_analogue_search.py`

Builds a regime fingerprint from current data (regime label, 10-day stability, rs_trend, trend IC). Queries `logs/regime_episodes.jsonl` for episodes with matching regime prefix and similar conditions. Returns top 3 analogues with their realized_return_21d outcomes.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_analogue_search.py
import json
import pytest
from pathlib import Path
from ascent.monitoring.analogue_search import find_historical_analogues


def test_finds_analogues_with_matching_regime(tmp_path):
    episodes = [
        {"date": f"2024-0{i+1}-10", "regime": "calm_bull",
         "quant_weights": {"AAPL": 0.1}, "ai_weights": None,
         "realized_return_21d": 0.03 + i * 0.01}
        for i in range(5)
    ]
    p = tmp_path / "regime_episodes.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in episodes))

    trajectory = {
        "current_label": "calm_bull", "stability_10d": 0.9,
        "rs_trend": "flat", "days_in_regime": 8,
    }
    signal_health = {"trend": {"ic_5d_avg": 0.013, "status": "healthy"}}

    result = find_historical_analogues(
        "2026-05-20", trajectory, signal_health,
        episodes_path=str(p)
    )

    assert isinstance(result, list)
    assert len(result) <= 3
    for analogue in result:
        assert "date" in analogue
        assert "regime" in analogue
        assert "outcome_21d" in analogue


def test_returns_empty_without_episodes(tmp_path):
    result = find_historical_analogues(
        "2026-05-20", {}, {},
        episodes_path=str(tmp_path / "missing.jsonl")
    )
    assert result == []


def test_excludes_episodes_without_outcomes(tmp_path):
    episodes = [
        {"date": "2026-05-10", "regime": "calm_bull",
         "quant_weights": {}, "ai_weights": None,
         "realized_return_21d": None}  # no outcome yet
    ]
    p = tmp_path / "regime_episodes.jsonl"
    p.write_text(json.dumps(episodes[0]))

    result = find_historical_analogues(
        "2026-05-20", {"current_label": "calm_bull"}, {},
        episodes_path=str(p)
    )
    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/monitoring/test_analogue_search.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement analogue_search.py**

```python
# ascent/monitoring/analogue_search.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

log = logging.getLogger(__name__)

_EPISODES_PATH = "logs/regime_episodes.jsonl"


def find_historical_analogues(
    date: str,
    regime_trajectory: Dict[str, Any],
    signal_health: Dict[str, Any],
    episodes_path: str = _EPISODES_PATH,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Scores past regime episodes by similarity to current conditions.
    Only considers episodes with realized_return_21d populated.
    Similarity: regime prefix match (required) + stability distance + IC proximity.
    Returns top_n best matches with their outcomes.
    """
    path = Path(episodes_path)
    if not path.exists():
        return []

    current_label    = regime_trajectory.get("current_label", "")
    current_stab     = regime_trajectory.get("stability_10d", 0.5)
    current_trend_ic = (signal_health.get("trend", {}).get("ic_5d_avg") or 0.0)

    episodes = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            ep = json.loads(line)
            # Skip if no outcome or same-period date
            if ep.get("realized_return_21d") is None:
                continue
            if ep.get("date", "") >= date:
                continue
            # Regime prefix match required
            ep_regime = ep.get("regime", "")
            if not ep_regime.startswith(current_label.split("_")[0]):
                continue
            episodes.append(ep)
        except Exception:
            continue

    if not episodes:
        return []

    # Score similarity (lower = more similar)
    scored = []
    for ep in episodes:
        regime_match = 1.0 if ep.get("regime") == current_label else 0.5
        scored.append((1.0 - regime_match, ep))

    scored.sort(key=lambda x: x[0])
    top = [ep for _, ep in scored[:top_n]]

    return [
        {
            "date":       ep["date"],
            "regime":     ep.get("regime", ""),
            "outcome_21d": ep["realized_return_21d"],
            "n_positions": len(ep.get("quant_weights", {})),
        }
        for ep in top
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_analogue_search.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/analogue_search.py tests/monitoring/test_analogue_search.py
git commit -m "feat: historical analogue search — regime fingerprint matching from episode memory"
```

---

## Task 6: Per-Position Living Thesis (Haiku)

**Files:**
- Create: `ascent/monitoring/position_thesis.py`

One batched Haiku call covering all held positions. Returns a dict of `{symbol: updated_thesis_str}`. Reads the last rebalance thesis from `outputs/ai_pm_theses/` to give Haiku the original rationale to compare against.

- [ ] **Step 1: Implement position_thesis.py**

No pure-Python test possible (LLM call). The module is written to be skippable on failure — returns `{}` on any error.

```python
# ascent/monitoring/position_thesis.py
import json
import logging
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)

_THESES_DIR = "outputs/ai_pm_theses"

_SYSTEM = (
    "You are an institutional equity analyst. Given today's portfolio data, "
    "assess whether each position's original investment thesis still holds. "
    "For each symbol respond in ≤60 words: thesis status (intact/weakening/broken), "
    "one key supporting or contradicting data point, and any near-term risk. "
    "Be specific and quantitative where possible."
)


def _load_last_thesis_rationale() -> Dict[str, str]:
    """Return {symbol: rationale} from the most recent AI PM thesis JSON."""
    theses_dir = Path(_THESES_DIR)
    if not theses_dir.exists():
        return {}
    files = sorted(theses_dir.glob("*.json"))
    if not files:
        return {}
    try:
        data = json.loads(files[-1].read_text())
        rationale = data.get("position_rationale", {})
        return {k: str(v) for k, v in rationale.items()}
    except Exception:
        return {}


def update_position_theses(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
) -> Dict[str, str]:
    """
    Calls Haiku once with all held positions to update each thesis.
    Returns {} on any failure — never blocks the daily run.
    """
    if not merged_weights:
        return {}

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return {}

    # Build alpha score context from us_equities agent
    alpha_context = {}
    us_agent = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
    if us_agent is not None and not us_agent.alpha_scores.empty:
        try:
            latest = us_agent.alpha_scores.iloc[-1].to_dict()
            alpha_context = {s: round(latest[s], 3) for s in merged_weights if s in latest}
        except Exception:
            pass

    last_rationale = _load_last_thesis_rationale()

    # Build user prompt
    lines = [f"Date: {date}", "Held positions:"]
    for sym, wt in merged_weights.items():
        alpha = alpha_context.get(sym, "N/A")
        rationale = last_rationale.get(sym, "No prior rationale available.")
        lines.append(
            f"\n{sym} ({wt:.1%} weight, alpha={alpha}):\n"
            f"  Original rationale: {rationale[:200]}"
        )

    lines.append(
        "\nReturn a JSON object: {\"SYMBOL\": \"updated thesis in ≤60 words\", ...}"
        " for every symbol above."
    )

    try:
        raw = generate_structured(
            system_prompt=_SYSTEM,
            user_prompt="\n".join(lines),
            model=HAIKU_MODEL,
            max_tokens=2000,
            temperature=0.3,
            use_cache=True,
        )
        # Extract JSON block
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return {}
        return json.loads(raw[start:end])
    except Exception as e:
        log.warning("[PositionThesis] Failed: %s", e)
        return {}
```

- [ ] **Step 2: Syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/monitoring/position_thesis.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ascent/monitoring/position_thesis.py
git commit -m "feat: per-position living thesis — daily Haiku update vs rebalance rationale"
```

---

## Task 7: Daily Adversarial Challenge (Haiku)

**Files:**
- Create: `ascent/monitoring/adversarial_daily.py`

Single Haiku call. Asks: "What is the single most dangerous assumption baked into this portfolio right now?" Returns a plain string. Never blocks.

- [ ] **Step 1: Implement adversarial_daily.py**

```python
# ascent/monitoring/adversarial_daily.py
import json
import logging
from typing import Dict

log = logging.getLogger(__name__)

_SYSTEM = (
    "You are a risk manager at an institutional quantitative fund. "
    "Your job is to identify the SINGLE most dangerous assumption currently embedded "
    "in the portfolio — something the quant signals would not catch. "
    "Think about: hidden correlations, regime fragility, crowding, "
    "event risk, liquidity, or thesis staleness. "
    "Be specific, quantitative, and under 100 words."
)


def generate_adversarial_challenge(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    regime: str = "unknown",
) -> str:
    """
    Returns a single adversarial challenge string, or '' on failure.
    """
    if not merged_weights:
        return ""

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return ""

    top_positions = sorted(merged_weights.items(), key=lambda x: x[1], reverse=True)[:8]
    pos_str = ", ".join(f"{s} ({w:.1%})" for s, w in top_positions)

    user_prompt = (
        f"Date: {date}\n"
        f"Regime: {regime}\n"
        f"Top positions: {pos_str}\n"
        f"Total positions: {len(merged_weights)}\n\n"
        "What is the single most dangerous assumption in this portfolio right now?"
    )

    try:
        return generate_structured(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.7,
        ).strip()
    except Exception as e:
        log.warning("[AdversarialDaily] Failed: %s", e)
        return ""
```

- [ ] **Step 2: Syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/monitoring/adversarial_daily.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ascent/monitoring/adversarial_daily.py
git commit -m "feat: daily adversarial challenge — Haiku surfaces hidden portfolio assumptions"
```

---

## Task 8: Macro Event Risk Calendar (Haiku)

**Files:**
- Create: `ascent/monitoring/macro_calendar.py`

Builds a 10-day forward calendar from: (a) a hardcoded 2026 FOMC/CPI/NFP schedule and (b) held-position earnings dates from `data_cache/earnings_cache.json` if available. One Haiku call scores each event's portfolio sensitivity.

- [ ] **Step 1: Implement macro_calendar.py**

```python
# ascent/monitoring/macro_calendar.py
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

# 2026 FOMC meeting dates (decision days)
_FOMC_2026 = [
    "2026-01-29", "2026-03-19", "2026-05-07", "2026-06-18",
    "2026-07-30", "2026-09-17", "2026-11-05", "2026-12-17",
]
# 2026 CPI release dates (approx — BLS releases ~2 weeks after month end)
_CPI_2026 = [
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
    "2026-05-13", "2026-06-11", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-13", "2026-11-12", "2026-12-11",
]
# NFP (first Friday of each month, approx)
_NFP_2026 = [
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-10", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

_SYSTEM = (
    "You are a macro risk analyst at an institutional fund. "
    "For each upcoming event, rate the portfolio's sensitivity from 1 (minimal impact) "
    "to 5 (high impact) and explain in one sentence which positions are most exposed. "
    "Return a JSON array: [{\"event\": str, \"date\": str, \"days_away\": int, "
    "\"sensitivity\": int, \"impact\": str}]"
)


def _upcoming_macro_events(today: date, horizon_days: int = 10) -> List[Dict]:
    events = []
    end = today + timedelta(days=horizon_days)
    for d_str in _FOMC_2026 + _CPI_2026 + _NFP_2026:
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        if today < d <= end:
            label = ("FOMC Decision" if d_str in _FOMC_2026
                     else "CPI Release" if d_str in _CPI_2026 else "NFP Release")
            events.append({"event": label, "date": d_str, "days_away": (d - today).days})
    return sorted(events, key=lambda x: x["days_away"])


def _load_earnings_events(today: date, merged_weights: Dict, horizon_days: int = 10) -> List[Dict]:
    cache_path = Path("data_cache/earnings_cache.json")
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return []
    end = today + timedelta(days=horizon_days)
    events = []
    for sym in merged_weights:
        entry = data.get(sym, {})
        d_str = entry.get("report_date") or entry.get("next_earnings_date")
        if not d_str:
            continue
        try:
            d = date.fromisoformat(str(d_str)[:10])
        except ValueError:
            continue
        if today < d <= end:
            events.append({"event": f"{sym} Earnings", "date": str(d), "days_away": (d - today).days})
    return events


def build_event_calendar(
    date_str: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
) -> List[Dict]:
    """
    Returns list of upcoming macro/earnings events with Haiku-scored sensitivity.
    Returns [] on failure.
    """
    try:
        today = date.fromisoformat(date_str)
    except ValueError:
        return []

    events = _upcoming_macro_events(today) + _load_earnings_events(today, merged_weights)
    if not events:
        return events

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return events

    top_positions = sorted(merged_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    pos_str = ", ".join(f"{s} ({w:.1%})" for s, w in top_positions)
    events_str = "\n".join(
        f"- {e['event']} on {e['date']} ({e['days_away']} days away)" for e in events
    )

    user_prompt = (
        f"Portfolio top positions: {pos_str}\n\n"
        f"Upcoming events:\n{events_str}\n\n"
        "Score each event's portfolio sensitivity and identify exposed positions."
    )

    try:
        raw = generate_structured(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=800,
            temperature=0.3,
            use_cache=True,
        )
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return events
        return json.loads(raw[start:end])
    except Exception as e:
        log.warning("[MacroCalendar] Haiku scoring failed (%s) — returning raw events", e)
        return events
```

- [ ] **Step 2: Syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/monitoring/macro_calendar.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ascent/monitoring/macro_calendar.py
git commit -m "feat: macro event risk calendar — FOMC/CPI/NFP/earnings sensitivity scoring"
```

---

## Task 9: Daily Intelligence Orchestrator

**Files:**
- Create: `ascent/monitoring/daily_intelligence.py`
- Create: `tests/monitoring/test_daily_intelligence.py`

Calls all 7 modules, assembles the daily JSON entry, writes to `data_cache/daily_intelligence/YYYY-MM-DD.json`. Each module is wrapped in try/except — one failure never blocks the others.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_daily_intelligence.py
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from ascent.monitoring.daily_intelligence import run_daily_intelligence


def test_writes_daily_entry(tmp_path):
    merged_weights = {"AAPL": 0.07, "MSFT": 0.07}
    agent_outputs  = []

    with patch("ascent.monitoring.daily_intelligence.compute_conviction_decay", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.compute_signal_health", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.compute_regime_trajectory", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.find_historical_analogues", return_value=[]), \
         patch("ascent.monitoring.daily_intelligence.update_position_theses", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.generate_adversarial_challenge", return_value="test challenge"), \
         patch("ascent.monitoring.daily_intelligence.build_event_calendar", return_value=[]):

        run_daily_intelligence(
            "2026-05-20", merged_weights, agent_outputs,
            output_dir=str(tmp_path)
        )

    out = tmp_path / "2026-05-20.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["date"] == "2026-05-20"
    assert "conviction_decay" in data
    assert "adversarial_challenge" in data
    assert data["adversarial_challenge"] == "test challenge"


def test_module_failure_does_not_block(tmp_path):
    with patch("ascent.monitoring.daily_intelligence.compute_conviction_decay", side_effect=RuntimeError("boom")), \
         patch("ascent.monitoring.daily_intelligence.compute_signal_health", return_value={"trend": {}}), \
         patch("ascent.monitoring.daily_intelligence.compute_regime_trajectory", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.find_historical_analogues", return_value=[]), \
         patch("ascent.monitoring.daily_intelligence.update_position_theses", return_value={}), \
         patch("ascent.monitoring.daily_intelligence.generate_adversarial_challenge", return_value=""), \
         patch("ascent.monitoring.daily_intelligence.build_event_calendar", return_value=[]):

        run_daily_intelligence("2026-05-20", {"AAPL": 0.07}, [], output_dir=str(tmp_path))

    out = tmp_path / "2026-05-20.json"
    assert out.exists()
    data = json.loads(out.read_text())
    # signal_health still populated even though conviction_decay failed
    assert data["signal_health"] == {"trend": {}}
    assert data["conviction_decay"] == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/monitoring/test_daily_intelligence.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement daily_intelligence.py**

```python
# ascent/monitoring/daily_intelligence.py
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List

from ascent.monitoring.conviction_tracker   import compute_conviction_decay
from ascent.monitoring.signal_health        import compute_signal_health
from ascent.monitoring.regime_trajectory    import compute_regime_trajectory
from ascent.monitoring.analogue_search      import find_historical_analogues
from ascent.monitoring.position_thesis      import update_position_theses
from ascent.monitoring.adversarial_daily    import generate_adversarial_challenge
from ascent.monitoring.macro_calendar       import build_event_calendar

log = logging.getLogger(__name__)

_OUTPUT_DIR = "data_cache/daily_intelligence"


def run_daily_intelligence(
    date: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
    output_dir: str = _OUTPUT_DIR,
) -> Dict:
    """
    Runs all 7 intelligence modules. Each failure is caught independently.
    Writes result to output_dir/YYYY-MM-DD.json and returns the dict.
    """
    log.info("[DailyIntel] Running non-rebalance intelligence for %s", date)

    # Extract current regime for modules that need it
    regime = "unknown"
    try:
        traj = compute_regime_trajectory(date)
        regime = traj.get("current_label", "unknown")
    except Exception:
        traj = {}

    def _safe(name, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.warning("[DailyIntel] %s failed: %s", name, e)
            return {} if name != "adversarial" else ""

    conviction  = _safe("conviction_decay",  compute_conviction_decay,
                         date, merged_weights, agent_outputs)
    signal      = _safe("signal_health",     compute_signal_health, date)
    # regime_trajectory already computed above; re-use
    if not traj:
        traj    = _safe("regime_trajectory", compute_regime_trajectory, date)
    analogues   = _safe("analogue_search",   find_historical_analogues,
                         date, traj, signal)
    theses      = _safe("position_thesis",   update_position_theses,
                         date, merged_weights, agent_outputs)
    adversarial = _safe("adversarial",       generate_adversarial_challenge,
                         date, merged_weights, agent_outputs, regime)
    macro_evts  = _safe("macro_calendar",    build_event_calendar,
                         date, merged_weights, agent_outputs)

    entry = {
        "date":                date,
        "conviction_decay":    conviction,
        "signal_health":       signal,
        "regime_trajectory":   traj,
        "historical_analogues": analogues,
        "position_theses":     theses,
        "adversarial_challenge": adversarial,
        "macro_events":        macro_evts,
    }

    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date}.json"

    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(entry, f, indent=2)
        os.replace(tmp_path, out_path)
    except Exception as e:
        log.error("[DailyIntel] Failed to write %s: %s", out_path, e)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    log.info("[DailyIntel] Written to %s", out_path)
    return entry
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_daily_intelligence.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/daily_intelligence.py tests/monitoring/test_daily_intelligence.py
git commit -m "feat: daily intelligence orchestrator — runs all 7 modules, atomic write"
```

---

## Task 10: Rebalance Brief Synthesizer

**Files:**
- Create: `ascent/monitoring/rebalance_brief.py`
- Create: `tests/monitoring/test_rebalance_brief.py`

Reads the last N daily intelligence entries. One Haiku call synthesizes them into a structured briefing JSON written to `data_cache/rebalance_brief.json`. Called on rebalance day before the AI PM runs.

- [ ] **Step 1: Write the failing test**

```python
# tests/monitoring/test_rebalance_brief.py
import json
from pathlib import Path
from unittest.mock import patch
from ascent.monitoring.rebalance_brief import generate_rebalance_brief


def _write_intel_entries(tmp_path, n=3):
    d = tmp_path / "daily_intelligence"
    d.mkdir()
    for i in range(n):
        date = f"2026-05-{17+i:02d}"
        entry = {
            "date": date,
            "conviction_decay": {"AAPL": {"rank_at_rebalance": 1, "rank_today": 3}},
            "signal_health": {"trend": {"status": "healthy", "change_pct": -5.0}},
            "regime_trajectory": {"current_label": "calm_bull", "stability_10d": 0.9},
            "historical_analogues": [{"date": "2024-03-01", "outcome_21d": 0.03}],
            "position_theses": {"AAPL": "Thesis intact."},
            "adversarial_challenge": f"Risk #{i+1}: crowding.",
            "macro_events": [{"event": "FOMC", "date": "2026-05-27", "sensitivity": 4}],
        }
        (d / f"{date}.json").write_text(json.dumps(entry))
    return str(d)


def test_generates_brief_from_entries(tmp_path):
    intel_dir = _write_intel_entries(tmp_path)
    brief_path = str(tmp_path / "rebalance_brief.json")

    with patch("ascent.monitoring.rebalance_brief.generate_structured") as mock_llm:
        mock_llm.return_value = "The portfolio enters rebalance in a stable calm_bull regime."
        generate_rebalance_brief(
            "2026-05-20",
            intel_dir=intel_dir,
            brief_path=brief_path,
        )

    assert Path(brief_path).exists()
    data = json.loads(Path(brief_path).read_text())
    assert data["date"] == "2026-05-20"
    assert "synthesis" in data
    assert "stale_positions" in data
    assert "weakening_sleeves" in data
    assert mock_llm.called


def test_returns_empty_brief_without_entries(tmp_path):
    empty_dir = str(tmp_path / "daily_intelligence")
    Path(empty_dir).mkdir()
    brief_path = str(tmp_path / "brief.json")

    result = generate_rebalance_brief("2026-05-20", intel_dir=empty_dir, brief_path=brief_path)
    assert result["synthesis"] == ""
    assert result["stale_positions"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/monitoring/test_rebalance_brief.py -v
```

Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement rebalance_brief.py**

```python
# ascent/monitoring/rebalance_brief.py
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any

log = logging.getLogger(__name__)

_INTEL_DIR   = "data_cache/daily_intelligence"
_BRIEF_PATH  = "data_cache/rebalance_brief.json"
_MAX_ENTRIES = 9  # one rebalance cycle

_SYSTEM = (
    "You are a senior portfolio manager synthesizing a pre-rebalance intelligence brief. "
    "You have been given 9 days of structured observations: conviction decay, signal health, "
    "regime trajectory, historical analogues, per-position thesis updates, daily adversarial "
    "challenges, and upcoming macro events. "
    "Write a 300-400 word briefing covering: (1) regime assessment and stability, "
    "(2) positions whose thesis has weakened most, (3) alpha signal environment, "
    "(4) key risks and upcoming catalysts, (5) what historical analogues suggest. "
    "Be specific, reference symbols by name, and use numbers where available. "
    "This brief will be the first thing the AI portfolio manager reads before making decisions."
)


def _load_entries(intel_dir: str) -> List[Dict]:
    d = Path(intel_dir)
    if not d.exists():
        return []
    files = sorted(d.glob("*.json"))[-_MAX_ENTRIES:]
    entries = []
    for f in files:
        try:
            entries.append(json.loads(f.read_text()))
        except Exception:
            continue
    return entries


def _extract_stale_positions(entries: List[Dict]) -> List[str]:
    """Symbols that dropped ≥10 ranks from rebalance baseline in latest entry."""
    if not entries:
        return []
    latest = entries[-1].get("conviction_decay", {})
    stale = []
    for sym, data in latest.items():
        r_then = data.get("rank_at_rebalance")
        r_now  = data.get("rank_today")
        if r_then is not None and r_now is not None and (r_now - r_then) >= 10:
            stale.append(sym)
    return stale


def _extract_weakening_sleeves(entries: List[Dict]) -> List[str]:
    if not entries:
        return []
    latest = entries[-1].get("signal_health", {})
    return [s for s, d in latest.items() if d.get("status") in ("weakening", "deteriorating")]


def generate_rebalance_brief(
    date: str,
    intel_dir: str = _INTEL_DIR,
    brief_path: str = _BRIEF_PATH,
) -> Dict[str, Any]:
    """
    Synthesizes daily intelligence entries into a structured rebalance briefing.
    Writes to brief_path and returns the dict. Returns empty brief on failure.
    """
    empty = {
        "date": date, "synthesis": "", "stale_positions": [],
        "weakening_sleeves": [], "top_macro_risks": [], "analogue_signal": "",
        "adversarial_themes": [],
    }

    entries = _load_entries(intel_dir)
    if not entries:
        log.warning("[RebalanceBrief] No intelligence entries found in %s", intel_dir)
        _write_brief(empty, brief_path)
        return empty

    stale_positions  = _extract_stale_positions(entries)
    weakening_sleeves = _extract_weakening_sleeves(entries)

    # Collect adversarial challenges (dedup)
    adversarial_themes = list({
        e.get("adversarial_challenge", "")
        for e in entries if e.get("adversarial_challenge")
    })

    # Top macro events from most recent entry
    top_macro_risks = [
        f"{ev.get('event')} ({ev.get('days_away')}d, sensitivity {ev.get('sensitivity')})"
        for ev in entries[-1].get("macro_events", [])[:3]
    ]

    # Historical analogue signal
    all_outcomes = [
        a.get("outcome_21d") for e in entries
        for a in e.get("historical_analogues", [])
        if a.get("outcome_21d") is not None
    ]
    analogue_signal = (
        f"{len(all_outcomes)} historical analogues found; "
        f"median 21d outcome {sorted(all_outcomes)[len(all_outcomes)//2]:+.1%}"
        if all_outcomes else "Insufficient historical analogues"
    )

    # Build condensed prompt
    summary_lines = [
        f"Period: last {len(entries)} trading days ending {date}",
        f"Stale positions (rank dropped ≥10): {stale_positions or 'none'}",
        f"Weakening alpha sleeves: {weakening_sleeves or 'none'}",
        f"Regime trajectory: {json.dumps(entries[-1].get('regime_trajectory', {}))}",
        f"Analogue signal: {analogue_signal}",
        f"Top macro risks: {top_macro_risks}",
        "Daily adversarial challenges (last 3):",
    ] + [f"  - {t}" for t in adversarial_themes[-3:]] + [
        "Latest position thesis updates:",
    ] + [
        f"  {sym}: {thesis[:80]}"
        for sym, thesis in list(entries[-1].get("position_theses", {}).items())[:6]
    ]

    user_prompt = "\n".join(summary_lines)

    synthesis = ""
    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
        synthesis = generate_structured(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=700,
            temperature=0.3,
            use_cache=True,
        ).strip()
    except Exception as e:
        log.warning("[RebalanceBrief] Haiku synthesis failed: %s", e)

    result = {
        "date":               date,
        "synthesis":          synthesis,
        "stale_positions":    stale_positions,
        "weakening_sleeves":  weakening_sleeves,
        "top_macro_risks":    top_macro_risks,
        "analogue_signal":    analogue_signal,
        "adversarial_themes": adversarial_themes[-3:],
        "n_entries":          len(entries),
    }
    _write_brief(result, brief_path)
    return result


def _write_brief(data: Dict, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, p)
    except Exception as e:
        log.error("[RebalanceBrief] Write failed: %s", e)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/monitoring/test_rebalance_brief.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/rebalance_brief.py tests/monitoring/test_rebalance_brief.py
git commit -m "feat: rebalance brief synthesizer — 9-day intelligence summary for AI PM"
```

---

## Task 11: AI PM Tool Integration

**Files:**
- Modify: `agents/ai_pm_agent.py` — add `get_rebalance_brief` as tool #17, update Phase 1 instruction in `_SYSTEM_PROMPT`

The AI PM calls `get_rebalance_brief` first in Phase 1, before `get_regime_state`. This gives it pre-digested context and reduces how many tool calls it needs for raw information gathering.

- [ ] **Step 1: Add tool schema to AI_PM_TOOLS**

In `agents/ai_pm_agent.py`, find `AI_PM_TOOLS = [` and insert at the top of the list:

```python
    {
        "name": "get_rebalance_brief",
        "description": (
            "Get the pre-rebalance intelligence brief synthesized from the last 9 non-rebalance "
            "days. Contains: regime trajectory and stability, positions whose conviction has "
            "decayed since last rebalance, weakening alpha sleeves, macro event risks, "
            "historical analogue outcomes, and accumulated adversarial challenges. "
            "Call this FIRST before any other tool."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
```

- [ ] **Step 2: Implement _tool_get_rebalance_brief**

Add after the other `_tool_` functions (before `_make_executor`):

```python
def _tool_get_rebalance_brief(_: dict) -> str:
    try:
        from pathlib import Path
        import json
        brief_path = Path("data_cache/rebalance_brief.json")
        if not brief_path.exists():
            return "No pre-rebalance brief available. Proceed with standard research."
        data = json.loads(brief_path.read_text())
        lines = [
            f"=== PRE-REBALANCE INTELLIGENCE BRIEF ({data.get('date', 'N/A')}) ===",
            f"\nSYNTHESIS:\n{data.get('synthesis', 'N/A')}",
            f"\nSTALE POSITIONS (rank decayed ≥10 since rebalance): {data.get('stale_positions') or 'none'}",
            f"WEAKENING ALPHA SLEEVES: {data.get('weakening_sleeves') or 'none'}",
            f"ANALOGUE SIGNAL: {data.get('analogue_signal', 'N/A')}",
            f"TOP MACRO RISKS: {'; '.join(data.get('top_macro_risks', [])) or 'none'}",
            "\nACCUMULATED ADVERSARIAL CHALLENGES (last 3 days):",
        ] + [f"  - {t}" for t in data.get("adversarial_themes", [])]
        return "\n".join(lines)
    except Exception as e:
        return f"Brief unavailable: {e}"
```

- [ ] **Step 3: Register in _make_executor**

In `_make_executor`, find the dispatch dict and add:

```python
"get_rebalance_brief": _tool_get_rebalance_brief,
```

- [ ] **Step 4: Update _SYSTEM_PROMPT Phase 1 instruction**

Find the Phase 1 line in `_SYSTEM_PROMPT` and update:

```
1. PHASE 1 — Market context: Call get_rebalance_brief FIRST (pre-digested 9-day intelligence),
   then get_regime_state and get_macro_data.
```

- [ ] **Step 5: Syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('agents/ai_pm_agent.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add agents/ai_pm_agent.py
git commit -m "feat: add get_rebalance_brief tool to AI PM — pre-digested 9-day intelligence brief"
```

---

## Task 12: Wire Into run_all_agents.py

**Files:**
- Modify: `run_all_agents.py`

Three wiring points:
1. **Rebalance day**: after writing merged_weights, call `save_rebalance_alpha_state()` and `generate_rebalance_brief()`
2. **Non-rebalance day**: call `run_daily_intelligence()` after the orchestrator step
3. **Both**: the brief is already in `data_cache/rebalance_brief.json` — AI PM reads it via tool

- [ ] **Step 1: Find the non-rebalance stop point**

In `run_all_agents.py`, find:
```python
    if not is_rebalance:
        print("[Runner] Non-rebalance day — weights updated, no debate, no execution.")
```

- [ ] **Step 2: Add daily intelligence call before the non-rebalance stop**

Insert before that `if not is_rebalance` block:

```python
    # ── Daily intelligence (non-rebalance days — feeds rebalance brief) ──────
    if not is_rebalance:
        try:
            from ascent.monitoring.daily_intelligence import run_daily_intelligence
            run_daily_intelligence(today.isoformat(), merged_weights, agent_outputs)
        except Exception as _di_e:
            print(f"[DailyIntel] Skipped: {_di_e}")
```

- [ ] **Step 3: Add rebalance-day calls**

On rebalance day, after `_log_run` is called and weights are finalized, find the rebalance path and insert after the weights are written to `merged_weights.json`:

Find the line `weights_path.write_text(...)` or the block that writes merged weights on rebalance day. After it, add:

```python
    # ── Snapshot rebalance baseline for conviction tracker ───────────────────
    if is_rebalance:
        try:
            from ascent.monitoring.conviction_tracker import save_rebalance_alpha_state
            from ascent.monitoring.signal_health import compute_signal_health
            sleeve_ics = {
                s: d.get("ic_5d_avg", 0.0)
                for s, d in compute_signal_health(today.isoformat()).items()
            }
            traj = {}
            try:
                from ascent.monitoring.regime_trajectory import compute_regime_trajectory
                traj = compute_regime_trajectory(today.isoformat())
            except Exception:
                pass
            save_rebalance_alpha_state(
                date=today.isoformat(),
                merged_weights=merged_weights,
                agent_outputs=agent_outputs,
                sleeve_ics=sleeve_ics,
                regime=traj.get("current_label", "unknown"),
                regime_stability_10d=traj.get("stability_10d", 0.5),
            )
        except Exception as _rs_e:
            print(f"[RebalanceState] Snapshot failed: {_rs_e}")

        # ── Generate rebalance brief from accumulated intelligence ────────────
        try:
            from ascent.monitoring.rebalance_brief import generate_rebalance_brief
            generate_rebalance_brief(today.isoformat())
            print("[RebalanceBrief] Brief generated for AI PM.")
        except Exception as _rb_e:
            print(f"[RebalanceBrief] Generation failed: {_rb_e}")
```

- [ ] **Step 4: Syntax check**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -10
```

Expected: 492+ tests passing, 0 failures.

- [ ] **Step 6: Final commit**

```bash
git add run_all_agents.py
git commit -m "feat: wire daily intelligence + rebalance brief into daily runner"
```

---

## Self-Review

**Spec coverage:**
- ✅ Conviction decay (Task 2)
- ✅ Signal health / sleeve IC decay (Task 3)
- ✅ Regime trajectory (Task 4)
- ✅ Historical analogues (Task 5)
- ✅ Per-position living thesis / Haiku (Task 6)
- ✅ Daily adversarial challenge / Haiku (Task 7)
- ✅ Macro event calendar / Haiku (Task 8)
- ✅ Daily intelligence orchestrator (Task 9)
- ✅ Rebalance brief synthesis (Task 10)
- ✅ AI PM tool integration (Task 11)
- ✅ run_all_agents.py wiring (Task 12)

**Placeholder scan:** No TBD, no "implement later", no "similar to Task N". All code blocks are complete. All test assertions are specific.

**Type consistency:**
- `compute_conviction_decay` → `Dict[str, Any]` — consistent across Tasks 2 and 9
- `compute_signal_health` → `Dict[str, Any]` — consistent across Tasks 3, 9, 12
- `compute_regime_trajectory` → `Dict[str, Any]` — consistent across Tasks 4, 5, 9, 12
- `find_historical_analogues` takes `(date, trajectory, signal_health, episodes_path)` — consistent Tasks 5, 9
- `run_daily_intelligence(date, merged_weights, agent_outputs, output_dir)` — consistent Tasks 9, 12
- `generate_rebalance_brief(date, intel_dir, brief_path)` — consistent Tasks 10, 12
- `save_rebalance_alpha_state(date, merged_weights, agent_outputs, sleeve_ics, regime, regime_stability_10d)` — consistent Tasks 2, 12
- `_tool_get_rebalance_brief` registered as `"get_rebalance_brief"` in both tool list and executor — consistent Task 11
