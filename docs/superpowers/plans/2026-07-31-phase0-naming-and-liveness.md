# Phase 0: Canonical Naming and Governance Liveness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the AI PM decision log duplicating rows, repair the 8 existing duplicates, add an alarm that fires when the earned-authority buffers stop filling, and give the five counterfactual tracks one canonical name each.

**Architecture:** Four independent changes, no new subsystems. Two reuse an idempotency pattern that already exists in `ascent/monitoring/ai_pm_counterfactual.py`; one adds a check to the existing `verify_docs.py` registry; one adds a small read-only series registry under `ascent/analyst/catalog/`. No LLM calls, no code generation, no adapters for the 41 log schemas. Nothing touches order submission.

**Tech Stack:** Python 3.12.13 at `.venv/bin/python`, pytest, pandas, stdlib `json`/`pathlib`.

## Global Constraints

- Always use `.venv/bin/python`. Never bare `python`.
- `import logging`; **never** `from loguru import logger` — loguru is not installed.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch to a Python file.
- Run `.venv/bin/python scripts/verify_docs.py` before each commit. It must stay at 0 failures.
- Data repairs are **dry-run by default** and write a `.bak` backup, following `scripts/rebuild_counterfactual_log.py`.
- Market dates come from `ascent/utils/market_time.py`, never `date.today()`.
- Nothing in this plan may import from `ascent.execution` or touch `execution/merged_weights.json`.
- This plan does **not** change `ai_weight`, remove any write path, or alter `PROMOTION_CONFIG`. The §7 authority decision in the spec is explicitly deferred.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `run_all_agents.py` (modify `_write_decision_log`, add `_upsert_decision_log`) | One decision-log row per `(date, phase)` | 1 |
| `tests/test_decision_log_idempotency.py` (create) | Pins the upsert behaviour | 1 |
| `scripts/dedupe_decision_log.py` (create) | One-off repair of the 8 existing duplicate rows | 2 |
| `tests/scripts/test_dedupe_decision_log.py` (create) | Pins dedup semantics on a fixture | 2 |
| `scripts/verify_docs.py` (add `check_authority_buffers_live`, register in `CHECKS`) | Liveness alarm on the authority ladder | 3 |
| `tests/test_authority_liveness_check.py` (create) | Pins the check's pass/fail boundary | 3 |
| `ascent/analyst/__init__.py`, `ascent/analyst/catalog/__init__.py`, `ascent/analyst/catalog/registry.py` (create) | Canonical name → series descriptor + loader | 4 |
| `tests/analyst/test_catalog_registry.py` (create) | Pins the registry contract | 4 |

---

## Task 1: Decision-log upsert

**Why:** `_write_decision_log` at `run_all_agents.py:186` appends unconditionally with `open(AI_PM_DECISION_LOG, "a")`. The 2026-06-10 overnight rerun therefore wrote **8 rows for one date**, with `fallback` flipping True/False between them. `ascent/monitoring/ai_pm_counterfactual.py::_upsert_daily` already solved the identical problem for the daily counterfactual log — its docstring even names the same rerun ("the June-10 overnight rerun wrote ~9 rows"). This log never got the same treatment.

**Files:**
- Modify: `run_all_agents.py` — add `_upsert_decision_log` next to `_write_decision_log` (~line 144), change the write inside `_write_decision_log` (~line 186), add `"phase": 2` to `entry`
- Test: `tests/test_decision_log_idempotency.py`

**Interfaces:**
- Consumes: `AI_PM_DECISION_LOG` (module-level `Path` constant, already defined ~line 43-54)
- Produces: `_upsert_decision_log(entry: dict) -> None`. Keys on `(entry["date"], entry["phase"])`, last-wins. Task 2 relies on the `phase` field existing on new rows.

- [ ] **Step 1: Write the failing test**

Create `tests/test_decision_log_idempotency.py`:

```python
"""One decision-log row per (date, phase). The 2026-06-10 rerun wrote 8."""
import json
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def runner(tmp_path, monkeypatch):
    mod = importlib.import_module("run_all_agents")
    monkeypatch.setattr(mod, "AI_PM_DECISION_LOG", tmp_path / "ai_pm_decision_log.jsonl")
    return mod


def _rows(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_rerun_same_date_replaces_row(runner):
    path = runner.AI_PM_DECISION_LOG
    runner._upsert_decision_log({"date": "2026-06-10", "phase": 2, "fallback": True})
    runner._upsert_decision_log({"date": "2026-06-10", "phase": 2, "fallback": False})

    rows = _rows(path)
    assert len(rows) == 1, f"expected 1 row after rerun, got {len(rows)}"
    assert rows[0]["fallback"] is False, "last write should win"


def test_distinct_dates_both_kept(runner):
    path = runner.AI_PM_DECISION_LOG
    runner._upsert_decision_log({"date": "2026-06-10", "phase": 2})
    runner._upsert_decision_log({"date": "2026-06-24", "phase": 2})

    assert len(_rows(path)) == 2


def test_distinct_phases_same_date_both_kept(runner):
    path = runner.AI_PM_DECISION_LOG
    runner._upsert_decision_log({"date": "2026-06-10", "phase": 1})
    runner._upsert_decision_log({"date": "2026-06-10", "phase": 2})

    rows = _rows(path)
    assert len(rows) == 2
    assert {r["phase"] for r in rows} == {1, 2}


def test_malformed_existing_line_is_dropped_not_fatal(runner):
    path = runner.AI_PM_DECISION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"date": "2026-06-09", "phase": 2}\nNOT JSON\n')

    runner._upsert_decision_log({"date": "2026-06-10", "phase": 2})

    rows = _rows(path)
    assert [r["date"] for r in rows] == ["2026-06-09", "2026-06-10"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_decision_log_idempotency.py -v`
Expected: FAIL with `AttributeError: module 'run_all_agents' has no attribute '_upsert_decision_log'`

- [ ] **Step 3: Add the upsert helper**

In `run_all_agents.py`, immediately **above** `def _write_decision_log(...)` (~line 144):

```python
def _upsert_decision_log(entry: dict) -> None:
    """Write one row per (date, phase). A rerun REPLACES the prior row.

    This log was append-only, so the 2026-06-10 overnight rerun left 8 rows for
    one date with `fallback` flipping between them -- and this file feeds
    override scoring and the authority ladder. `ai_pm_counterfactual._upsert_daily`
    already fixed the identical failure for the daily counterfactual log.
    """
    AI_PM_DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    key = (entry.get("date"), entry.get("phase"))
    rows = []
    if AI_PM_DECISION_LOG.exists():
        for line in AI_PM_DECISION_LOG.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if (r.get("date"), r.get("phase")) != key:
                rows.append(r)
    rows.append(entry)
    with open(AI_PM_DECISION_LOG, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
```

- [ ] **Step 4: Use it, and stamp the phase**

In `_write_decision_log`, add `"phase": 2,` to the `entry` dict immediately after the `"date"` line (~line 165):

```python
            "date":                  today.isoformat(),
            "phase":                 2,
```

Then replace the append block (~line 186-187):

```python
        with open(AI_PM_DECISION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

with:

```python
        _upsert_decision_log(entry)
```

- [ ] **Step 5: Verify syntax and run the tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read())"
.venv/bin/python -m pytest tests/test_decision_log_idempotency.py -v
.venv/bin/python scripts/verify_docs.py --quiet
```

Expected: `ast.parse` silent, 4 tests PASS, verify_docs reports 0 failures.

- [ ] **Step 6: Commit**

```bash
git add run_all_agents.py tests/test_decision_log_idempotency.py
git commit -m "fix(logging): upsert AI PM decision log on (date, phase)

The log was append-only, so the 2026-06-10 rerun wrote 8 rows for one date
with fallback flipping between them. This file feeds override scoring and the
authority ladder. Mirrors ai_pm_counterfactual._upsert_daily, which already
fixed the same failure for the daily counterfactual log.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Repair the existing duplicates

**Why:** Task 1 stops new duplicates. `logs/ai_pm_decision_log.jsonl` still holds 9 rows for 2 real decisions (8 × 2026-06-10, 1 × 2026-06-24), and none carry a `phase` field.

**Files:**
- Create: `scripts/dedupe_decision_log.py`
- Test: `tests/scripts/test_dedupe_decision_log.py`

**Interfaces:**
- Consumes: `_upsert_decision_log` semantics from Task 1 (last-wins on `(date, phase)`)
- Produces: `dedupe(rows: list[dict]) -> tuple[list[dict], list[dict]]` returning `(kept, dropped)`. Pure function — no file IO — so it is testable without fixtures on disk.

- [ ] **Step 1: Write the failing test**

Create `tests/scripts/test_dedupe_decision_log.py`:

```python
"""Dedup keeps the last row per (date, phase) and backfills a missing phase."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location(
    "dedupe_decision_log", ROOT / "scripts" / "dedupe_decision_log.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_last_row_wins_per_date():
    rows = [
        {"date": "2026-06-10", "fallback": True},
        {"date": "2026-06-10", "fallback": False},
        {"date": "2026-06-10", "fallback": True},
    ]
    kept, dropped = mod.dedupe(rows)
    assert len(kept) == 1
    assert kept[0]["fallback"] is True
    assert len(dropped) == 2


def test_missing_phase_is_backfilled_to_2():
    kept, _ = mod.dedupe([{"date": "2026-06-24"}])
    assert kept[0]["phase"] == 2


def test_existing_phase_is_preserved():
    kept, _ = mod.dedupe([{"date": "2026-06-24", "phase": 1}])
    assert kept[0]["phase"] == 1


def test_distinct_dates_and_phases_all_kept():
    rows = [
        {"date": "2026-06-10", "phase": 1},
        {"date": "2026-06-10", "phase": 2},
        {"date": "2026-06-24", "phase": 2},
    ]
    kept, dropped = mod.dedupe(rows)
    assert len(kept) == 3
    assert dropped == []


def test_chronological_order_preserved():
    rows = [
        {"date": "2026-06-24", "phase": 2},
        {"date": "2026-06-10", "phase": 2},
    ]
    kept, _ = mod.dedupe(rows)
    assert [r["date"] for r in kept] == ["2026-06-10", "2026-06-24"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/scripts/test_dedupe_decision_log.py -v`
Expected: FAIL — `scripts/dedupe_decision_log.py` does not exist.

- [ ] **Step 3: Write the script**

Create `scripts/dedupe_decision_log.py`:

```python
#!/usr/bin/env python
"""
Dedupe logs/ai_pm_decision_log.jsonl to one row per (date, phase).

The log was append-only until the upsert fix, so the 2026-06-10 overnight rerun
left 8 rows for one date with `fallback` flipping between them. This file feeds
override scoring and the earned-authority ladder, so duplicates are not cosmetic.

Rows written before the fix carry no `phase`; they are all Phase 2 and are
backfilled as such. Last row per key wins, matching `_upsert_decision_log`.

    .venv/bin/python scripts/dedupe_decision_log.py            # dry run
    .venv/bin/python scripts/dedupe_decision_log.py --write    # apply
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "logs" / "ai_pm_decision_log.jsonl"
DEFAULT_PHASE = 2


def dedupe(rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Return (kept, dropped). One row per (date, phase), last wins.

    Missing `phase` is backfilled to 2 -- every row written before the upsert
    fix was a Phase 2 synthesis. Output is sorted by (date, phase).
    """
    winners: Dict[tuple, dict] = {}
    dropped: List[dict] = []
    for row in rows:
        r = dict(row)
        r.setdefault("phase", DEFAULT_PHASE)
        key = (r.get("date"), r.get("phase"))
        if key in winners:
            dropped.append(winners[key])
        winners[key] = r
    kept = sorted(winners.values(), key=lambda r: (str(r.get("date")), r.get("phase")))
    return kept, dropped


def _load(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="apply changes (default: dry run)")
    args = ap.parse_args()

    rows = _load(LOG)
    kept, dropped = dedupe(rows)

    print(f"{LOG.relative_to(ROOT)}: {len(rows)} rows -> {len(kept)} kept, {len(dropped)} dropped")
    for r in kept:
        print(f"  KEEP {r.get('date')} phase={r.get('phase')} fallback={r.get('fallback')}")
    for r in dropped:
        print(f"  DROP {r.get('date')} phase={r.get('phase')} fallback={r.get('fallback')}")

    if not args.write:
        print("\nDry run. Re-run with --write to apply.")
        return 0

    if not rows:
        print("Nothing to do.")
        return 0

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = LOG.with_suffix(f".pre_dedupe.{stamp}.bak.jsonl")
    shutil.copy2(LOG, backup)
    with open(LOG, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(kept)} rows. Backup: {backup.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and the dry run**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/dedupe_decision_log.py').read())"
.venv/bin/python -m pytest tests/scripts/test_dedupe_decision_log.py -v
.venv/bin/python scripts/dedupe_decision_log.py
```

Expected: 5 tests PASS. Dry run prints `9 rows -> 2 kept, 7 dropped`, with KEEP lines for 2026-06-10 and 2026-06-24.

**Stop and read the dry-run output before continuing.** If the kept count is not 2, do not proceed — report the discrepancy instead.

- [ ] **Step 5: Apply the repair**

```bash
.venv/bin/python scripts/dedupe_decision_log.py --write
.venv/bin/python -c "
import json
rows=[json.loads(l) for l in open('logs/ai_pm_decision_log.jsonl') if l.strip()]
print(len(rows),'rows;',sorted((r['date'],r['phase']) for r in rows))
"
```

Expected: `2 rows; [('2026-06-10', 2), ('2026-06-24', 2)]`

- [ ] **Step 6: Commit**

```bash
git add scripts/dedupe_decision_log.py tests/scripts/test_dedupe_decision_log.py logs/ai_pm_decision_log.jsonl
git commit -m "fix(logging): dedupe AI PM decision log to one row per (date, phase)

9 rows for 2 real decisions (8x 2026-06-10 from an overnight rerun). Backfills
the phase field on pre-fix rows and keeps the last row per key. Dry-run by
default; backup written alongside.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: Earned-authority liveness check

**Why:** `data_cache/earned_authority.json` shows `days_at_level: 19` with `track_d_returns: []` and `track_astar_returns: []`. Promotion is gated on `len(d_buf) >= 21` in `ascent/strategy/earned_authority.py`, so with empty buffers the gate dict is never constructed — promotion was never *evaluated*, only demotion could fire. Nothing alarms on this today. The mechanism is visible in `logs/counterfactual_daily.jsonl`, whose recent rows carry `track_astar_return: null` and `track_d_return: null`; `update_authority` skips the buffer append whenever either is `None`.

This check is an alarm, not a fix. It does not change authority.

**Files:**
- Modify: `scripts/verify_docs.py` — add `check_authority_buffers_live` before the `# registry` divider (~line 660), register in `CHECKS`
- Test: `tests/test_authority_liveness_check.py`

**Interfaces:**
- Consumes: `ROOT` (`Path`, line 38) and `Result` (`Tuple[bool, str]`, line 41), both already defined in `verify_docs.py`
- Produces: `check_authority_buffers_live() -> Result`, and module constant `AUTHORITY_LIVENESS_MAX_DAYS = 10`

- [ ] **Step 1: Write the failing test**

Create `tests/test_authority_liveness_check.py`:

```python
"""The authority ladder must be observably alive, not just configured."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "verify_docs", ROOT / "scripts" / "verify_docs.py"
)
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(vd, "ROOT", tmp_path)
    p = tmp_path / "data_cache" / "earned_authority.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def test_fails_when_buffers_empty_past_threshold(state_file):
    state_file.write_text(json.dumps({
        "days_at_level": 19, "track_d_returns": [], "track_astar_returns": [],
    }))
    ok, detail = vd.check_authority_buffers_live()
    assert ok is False
    assert "19" in detail


def test_passes_when_buffers_filling(state_file):
    state_file.write_text(json.dumps({
        "days_at_level": 19,
        "track_d_returns": [0.001] * 5,
        "track_astar_returns": [0.002] * 5,
    }))
    ok, _ = vd.check_authority_buffers_live()
    assert ok is True


def test_passes_below_threshold_even_when_empty(state_file):
    """A freshly promoted level legitimately has empty buffers."""
    state_file.write_text(json.dumps({
        "days_at_level": 2, "track_d_returns": [], "track_astar_returns": [],
    }))
    ok, _ = vd.check_authority_buffers_live()
    assert ok is True


def test_fails_when_only_one_buffer_fills(state_file):
    state_file.write_text(json.dumps({
        "days_at_level": 19,
        "track_d_returns": [0.001] * 5,
        "track_astar_returns": [],
    }))
    ok, _ = vd.check_authority_buffers_live()
    assert ok is False


def test_passes_when_state_file_absent(state_file):
    assert vd.check_authority_buffers_live()[0] is True


def test_registered_in_checks():
    names = [name for name, _claim, _fn in vd.CHECKS]
    assert "authority_buffers_live" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_authority_liveness_check.py -v`
Expected: FAIL with `AttributeError: module 'verify_docs' has no attribute 'check_authority_buffers_live'`

- [ ] **Step 3: Add the check**

In `scripts/verify_docs.py`, immediately **above** the `# ------------------------------------------------------------ registry ----` divider:

```python
#: Update cycles the authority state may run with empty return buffers before
#: this is treated as a dead mechanism rather than a warm-up. Promotion needs 21
#: buffered days, so 10 empty cycles already means the gate cannot fire soon.
AUTHORITY_LIVENESS_MAX_DAYS = 10


def check_authority_buffers_live() -> Result:
    """The earned-authority ladder must be observably accumulating evidence.

    Promotion is gated on `len(track_d_returns) >= window` in
    ascent/strategy/earned_authority.py. With empty buffers that expression is
    never reached, so the gates are never EVALUATED -- only demotion, which
    needs a single bad day, can fire. On 2026-07-31 the state showed 19 update
    cycles with zero buffer appends and nothing alarmed.

    Buffers fill only when both track_d_return and track_astar_return are
    non-None; nulls in logs/counterfactual_daily.jsonl silently stall the ladder.
    """
    path = ROOT / "data_cache" / "earned_authority.json"
    if not path.exists():
        return True, "no earned_authority.json yet"
    try:
        state = json.loads(path.read_text())
    except Exception as exc:
        return False, f"earned_authority.json unreadable: {exc}"

    days = int(state.get("days_at_level", 0) or 0)
    d_buf = state.get("track_d_returns") or []
    a_buf = state.get("track_astar_returns") or []

    if days >= AUTHORITY_LIVENESS_MAX_DAYS and not (d_buf and a_buf):
        return False, (
            f"{days} update cycles at level with buffers "
            f"track_d={len(d_buf)} track_astar={len(a_buf)} -- promotion gate "
            f"cannot evaluate; only demotion can fire"
        )
    return True, f"days_at_level={days}, buffers d={len(d_buf)} astar={len(a_buf)}"
```

- [ ] **Step 4: Register it**

In the `CHECKS` list, add as the final entry after `("no_unsourced_sharpe", ...)`:

```python
    ("authority_buffers_live", "governance must be observably alive, not just configured", check_authority_buffers_live),
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('scripts/verify_docs.py').read())"
.venv/bin/python -m pytest tests/test_authority_liveness_check.py -v
.venv/bin/python scripts/verify_docs.py
```

Expected: 6 tests PASS. `verify_docs.py` now reports **26** checks, and `authority_buffers_live` **FAILS** against the real state file — that is correct and is the point of the task. Record the failure text in the commit body.

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_docs.py tests/test_authority_liveness_check.py
git commit -m "feat(guards): alarm when the authority ladder stops accumulating

earned_authority.json showed 19 update cycles with both return buffers empty.
Promotion is gated on len(buffer) >= 21, so the gates were never evaluated,
only demotion could fire, and nothing alarmed. This check fails loudly on that
state. It is an alarm only -- it does not change ai_weight or PROMOTION_CONFIG.

Currently RED by design.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 4: Canonical series registry

**Why:** Five artifacts answer one counterfactual question because no canonical *name* for that question exists — each new reader hand-wrote its own path and column handling. This task gives the five tracks one address each. It is a read-only registry: no adapters for the other 41 log schemas, no compiler, no codegen.

The five tracks are columns inside `logs/counterfactual_daily.jsonl`, not five files: `track_astar_return`, `track_a_return`, `track_b_return`, `track_c_return`, `track_d_return`.

**Files:**
- Create: `ascent/analyst/__init__.py`, `ascent/analyst/catalog/__init__.py`, `ascent/analyst/catalog/registry.py`
- Test: `tests/analyst/test_catalog_registry.py`

**Interfaces:**
- Consumes: nothing from earlier tasks
- Produces:
  - `Series` dataclass with fields `name, description, source, column, index_kind, coverage, provenance`
  - `SERIES: dict[str, Series]`
  - `describe(name: str) -> Series` — raises `KeyError` on unknown name
  - `load(name: str) -> pandas.Series` — float values indexed by `datetime.date`, sorted, nulls dropped
  - `names() -> list[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/analyst/test_catalog_registry.py`:

```python
"""One canonical name per question. Five artifacts answered one question before."""
import dataclasses
import json

import pytest

from ascent.analyst.catalog import registry


EXPECTED = {
    "counterfactual.track_astar",
    "counterfactual.track_a",
    "counterfactual.track_b",
    "counterfactual.track_c",
    "counterfactual.track_d",
}


def test_five_tracks_registered():
    assert EXPECTED.issubset(set(registry.names()))


def test_every_descriptor_is_complete():
    for name in registry.names():
        s = registry.describe(name)
        assert s.name == name
        assert s.description
        assert s.index_kind == "market_trading_day"
        assert s.coverage
        assert s.provenance


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        registry.describe("counterfactual.does_not_exist")


def test_names_are_unique_per_source_column():
    seen = {}
    for name in registry.names():
        s = registry.describe(name)
        key = (str(s.source), s.column)
        assert key not in seen, f"{name} and {seen[key]} both address {key}"
        seen[key] = name


def test_load_returns_dated_floats(tmp_path, monkeypatch):
    log = tmp_path / "counterfactual_daily.jsonl"
    log.write_text(
        json.dumps({"date": "2026-06-11", "track_b_return": 0.002}) + "\n"
        + json.dumps({"date": "2026-06-10", "track_b_return": 0.001}) + "\n"
        + json.dumps({"date": "2026-06-12", "track_b_return": None}) + "\n"
    )
    monkeypatch.setitem(
        registry.SERIES, "counterfactual.track_b",
        dataclasses.replace(registry.SERIES["counterfactual.track_b"], source=log),
    )
    out = registry.load("counterfactual.track_b")

    assert list(out.index) == sorted(out.index), "index must be sorted"
    assert len(out) == 2, "null values are dropped"
    assert out.iloc[0] == pytest.approx(0.001)


def test_real_sources_resolve():
    """Every registered source file must exist in the repo as it stands."""
    missing = [n for n in registry.names() if not registry.describe(n).source.exists()]
    assert not missing, f"unresolvable sources: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/analyst/test_catalog_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ascent.analyst'`

- [ ] **Step 3: Create the package**

```bash
mkdir -p ascent/analyst/catalog tests/analyst
touch ascent/analyst/__init__.py ascent/analyst/catalog/__init__.py tests/analyst/__init__.py
```

- [ ] **Step 4: Write the registry**

Create `ascent/analyst/catalog/registry.py`:

```python
"""Canonical series registry.

One name, one series, forever. Five separate artifacts came to answer the single
counterfactual question -- ai_pm_counterfactual, counterfactual_tracker,
counterfactual_rebuild, and two scripts -- because the question had no canonical
address. A new reader hand-wrote its own path and column handling every time.

This is a read-only lens over files that already exist. It moves no bytes and
owns no data. Registering a series here does not change how it is written.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_DAILY = _REPO / "logs" / "counterfactual_daily.jsonl"


@dataclass(frozen=True)
class Series:
    """A named series and everything needed to read and trust it."""
    name: str
    description: str
    source: Path
    column: str
    index_kind: str   # market_trading_day | calendar_day | utc_timestamp
    coverage: str     # the completeness invariant this series must satisfy
    provenance: str   # where the values ultimately come from


_COVERAGE = "every NYSE session from first to last row"

SERIES = {
    s.name: s for s in [
        Series(
            name="counterfactual.track_astar",
            description="Pure quant daily return, zero Phase 1 influence.",
            source=_DAILY, column="track_astar_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="reconstruction: snapshot weights priced on prices_live",
        ),
        Series(
            name="counterfactual.track_a",
            description=(
                "Quant plus Phase 1 sleeve priors. KNOWN DEFECT: structurally "
                "identical to track_astar -- both read the same post-orchestrator "
                "merged_weights, so this series measures nothing. Do not cite it."
            ),
            source=_DAILY, column="track_a_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="reconstruction: snapshot weights priced on prices_live",
        ),
        Series(
            name="counterfactual.track_b",
            description="Actual traded book daily return.",
            source=_DAILY, column="track_b_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="settled Alpaca 1D bars; total-return basis",
        ),
        Series(
            name="counterfactual.track_c",
            description="SPY benchmark daily return.",
            source=_DAILY, column="track_c_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="prices_live closes; split-only basis",
        ),
        Series(
            name="counterfactual.track_d",
            description="Pure AI PM portfolio daily return, pre-blend.",
            source=_DAILY, column="track_d_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="reconstruction: snapshot weights priced on prices_live",
        ),
    ]
}


def names() -> List[str]:
    """Every registered canonical name, sorted."""
    return sorted(SERIES)


def describe(name: str) -> Series:
    """The descriptor for one series. Raises KeyError on an unknown name."""
    if name not in SERIES:
        raise KeyError(f"unknown series {name!r}; known: {names()}")
    return SERIES[name]


def load(name: str) -> pd.Series:
    """Read one series as float values indexed by date, sorted, nulls dropped."""
    s = describe(name)
    if not s.source.exists():
        raise FileNotFoundError(f"{name}: source missing at {s.source}")

    values = {}
    for line in s.source.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        raw = row.get(s.column)
        if raw is None:
            continue
        try:
            values[date.fromisoformat(row["date"])] = float(raw)
        except (KeyError, ValueError, TypeError):
            continue

    out = pd.Series(values, dtype="float64", name=name)
    return out.sort_index()
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/catalog/registry.py').read())"
.venv/bin/python -m pytest tests/analyst/test_catalog_registry.py -v
.venv/bin/python -c "
from ascent.analyst.catalog import registry
for n in registry.names():
    print(f'{n:34s} {len(registry.load(n)):3d} pts')
"
```

Expected: 6 tests PASS. The loader prints non-zero point counts for `track_b` and `track_c`; `track_a`, `track_astar` and `track_d` may be lower, since recent rows carry nulls — that is the stalled-buffer mechanism from Task 3, now visible by name.

- [ ] **Step 6: Commit**

```bash
git add ascent/analyst tests/analyst
git commit -m "feat(catalog): canonical names for the five counterfactual tracks

Five artifacts came to answer one question because the question had no
canonical address. This is a read-only lens over logs/counterfactual_daily.jsonl:
one name per series, each carrying index_kind, coverage and provenance. It moves
no bytes and changes no writer. track_a carries its known defect in its own
description so a reader cannot cite it by accident.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Done criteria

```bash
.venv/bin/python -m pytest tests/test_decision_log_idempotency.py \
    tests/scripts/test_dedupe_decision_log.py \
    tests/test_authority_liveness_check.py \
    tests/analyst/test_catalog_registry.py -v
.venv/bin/python scripts/verify_docs.py
```

- 21 new tests pass.
- `logs/ai_pm_decision_log.jsonl` holds exactly 2 rows.
- `verify_docs.py` reports 26 checks, with `authority_buffers_live` **RED** — expected until the buffers fill or the §7 decision is made.
- `.venv/bin/python -c "import run_all_agents"` still succeeds.

## Explicitly out of scope

- Any change to `ai_weight`, `PROMOTION_CONFIG`, or the three write paths. Spec §7 is deferred pending the 2026-08-05 rebalance.
- Deleting or merging `counterfactual_tracker.py` / `counterfactual_rebuild.py` / the two backfill scripts. Registering canonical names is reversible; deleting readers is not, and several are cited in `CURRENT_VERIFIED_NUMBERS.md`.
- Adapters for the remaining 41 log schemas, the planner, codegen, verify, runtime.
- Repointing `scripts/generate_performance_page.py` at the registry.
