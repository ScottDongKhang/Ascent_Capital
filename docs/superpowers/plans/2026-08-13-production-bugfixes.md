# Production Bugfixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two real production bugs found during the proof-audit extension —
`feature_defs.py`'s missing `signal_date` midnight normalization, and `parquet.py::save_parquet`
dropping the index on wide-format caches — then repair the three corrupted price caches via a
real re-fetch, unlocking real measurement for `earnings`/`analyst` and all three specialist
agents in the proof-audit scorecard.

**Architecture:** Two independent bugfixes in shared production code, TDD'd against the existing
test suites for each file, then a data-repair step (force-refresh three caches over the network)
that depends on both fixes being in place first.

**Tech Stack:** Python 3.12.13, `.venv/bin/python`, pandas, pyarrow, pytest, yfinance (for the
cache-repair step only).

## Global Constraints

- Always use `.venv/bin/python`. Never bare `python`.
- `import logging`; never `from loguru import logger`.
- Run `.venv/bin/python -c "import ast; ast.parse(open('<file>').read())"` after each patch.
- Live paper trading stays paused (`com.ascentcapital.eod`/`.heartbeat`) — this plan does not
  reload it.
- This plan touches shared production code (`ascent/features/feature_defs.py`,
  `ascent/data/store/parquet.py`, and possibly the three agent `_fetch_*_prices` functions if
  Task 2's investigation finds the correct fix location is there) — every change must be
  TDD'd against the EXISTING test suite for that file, run in full, not just new tests.
- Task 2 (the `save_parquet` fix) starts from an investigation, not a locked design — the
  obvious fix (branch the final `to_parquet(..., index=False)` call) is insufficient, because
  `save_parquet`'s read-modify-write does `pd.concat([old, df], ignore_index=True)`, which
  discards any index BEFORE the write step is reached, and its dedup-key logic is built around
  finding a `date` *column* (with excellent existing calendar-day dedup logic already there for
  exactly this purpose). The real fix likely converts a wide-format `DatetimeIndex` input into a
  `date` column at the TOP of `save_parquet` (reusing 100% of the existing dedup machinery for
  free) rather than special-casing the final write call — but confirm this in Task 2 before
  committing to it, the way Task 14's tz investigation in the prior sub-project falsified its
  own starting hypothesis before landing on the real fix.
- If the correct fix requires restoring a `DatetimeIndex` after `load_parquet` for the three
  affected agents specifically, doing so in the three `_fetch_*_prices` call sites
  (`agents/macro_agent.py`, `international_agent.py`, `alternatives_agent.py`) — one line each —
  is IN SCOPE and preferred over adding wide-format-detection heuristics to `load_parquet`
  itself, which is a shared, generic function used by every cache in the system. Keep
  `load_parquet`'s behavior unchanged for every other caller.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `ascent/features/feature_defs.py` (modify 3 functions) | Add missing `.dt.normalize()` | 1 |
| `tests/test_earnings_alpha.py`, `tests/test_analyst_alpha.py` (modify) | Non-midnight regression tests | 1 |
| `tests/test_insider_alpha.py` or similar (create if absent) | Insider panel regression test | 1 |
| `ascent/data/store/parquet.py` (modify `save_parquet`, maybe `load_parquet`) | Preserve wide-format index | 2 |
| `agents/macro_agent.py`, `international_agent.py`, `alternatives_agent.py` (modify, if Task 2 finds this is the right place) | Restore index after load | 2 |
| `tests/test_parquet_store_dedup.py` (modify) | Round-trip regression tests | 2 |
| (data operation, no file) | Force-refresh 3 caches via live re-fetch | 3 |
| `outputs/analyst/proof_audit_2026-08-12.json` or a new dated file (regenerate) | Final extended scorecard | 4 |

---

## Task 1: Fix `signal_date` midnight normalization

**Why:** `build_earnings_panel`, `build_analyst_panel`, `build_insider_panel` in
`ascent/features/feature_defs.py` all tz-strip `signal_date` but never normalize it to midnight,
unlike their siblings `build_options_panel`/`build_short_panel` which already do both. Since the
price index they're reindexed against IS midnight-normalized, any non-midnight `signal_date`
never matches, producing all-NaN panels — the root cause of `earnings`/`analyst` scoring "need
at least 10 points, got 0" in the proof-audit's real-data run.

**Files:**
- Modify: `ascent/features/feature_defs.py` — lines ~299-302 (`build_earnings_panel`),
  ~356-358 (`build_analyst_panel`), ~455-457 (`build_insider_panel`)
- Test: `tests/test_earnings_alpha.py`, `tests/test_analyst_alpha.py`, and a test covering
  `build_insider_panel` (check `tests/test_phase6_signals.py` first for existing indirect
  coverage; create `tests/test_insider_alpha.py` only if no direct coverage exists)

**Interfaces:**
- Consumes: nothing new
- Produces: no interface change — `build_earnings_panel`/`build_analyst_panel`/
  `build_insider_panel` keep their existing signatures and return shapes; only their internal
  date handling changes.

- [ ] **Step 1: Read the current exact code**

```bash
sed -n '269,325p' ascent/features/feature_defs.py   # build_earnings_panel
sed -n '325,385p' ascent/features/feature_defs.py   # build_analyst_panel
sed -n '405,420p' ascent/features/feature_defs.py   # build_options_panel (reference pattern)
sed -n '428,477p' ascent/features/feature_defs.py   # build_insider_panel
sed -n '498,510p' ascent/features/feature_defs.py   # build_short_panel (reference pattern)
```

Confirm the exact reference pattern in `build_options_panel`/`build_short_panel` (should be
something like `pd.to_datetime(sub["signal_date"]).dt.normalize().dt.tz_localize(None)` or
`.dt.tz_localize(None).dt.normalize()` — read the ACTUAL order used, and match it exactly in the
three broken functions, don't guess the order).

- [ ] **Step 2: Write the failing tests**

In `tests/test_earnings_alpha.py`, find the existing test for `build_earnings_panel` (e.g.
`test_build_earnings_panel_strips_timezone`) and add, following that test's existing fixture
style:

```python
def test_build_earnings_panel_normalizes_non_midnight_signal_date():
    """A signal_date with a nonzero time-of-day must still align with a midnight price index."""
    df = pd.DataFrame({
        "symbol": ["AAPL"],
        "signal_date": [pd.Timestamp("2025-01-02 16:30:00")],  # non-midnight, tz-naive
        "surprise_pct": [0.05],
    })
    price_index = pd.date_range("2025-01-01", "2025-01-10", freq="D")
    panel = build_earnings_panel(df, price_index)  # match the real function signature
    assert panel.loc[pd.Timestamp("2025-01-02"), "AAPL"] == pytest.approx(0.05)
```

Adapt to the function's REAL signature and return shape (read Step 1's output first — don't
invent a signature). Add an analogous test to `tests/test_analyst_alpha.py` for
`build_analyst_panel`, and a third test (new file or existing) for `build_insider_panel`,
each using that function's real column names (`score` vs `surprise_pct`, etc.).

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_earnings_alpha.py tests/test_analyst_alpha.py -k normalize -v
```

Expected: FAIL — the panel has no row at `2025-01-02` (or is all-NaN there) because
`16:30:00` never matched the midnight index.

- [ ] **Step 4: Apply the fix**

In each of the three functions, change:
```python
        if sub["signal_date"].dt.tz is not None:
            sub["signal_date"] = sub["signal_date"].dt.tz_localize(None)
```
to also normalize — matching whatever exact order `build_options_panel`/`build_short_panel` use
(confirmed in Step 1), e.g.:
```python
        if sub["signal_date"].dt.tz is not None:
            sub["signal_date"] = sub["signal_date"].dt.tz_localize(None)
        sub["signal_date"] = sub["signal_date"].dt.normalize()
```

- [ ] **Step 5: Run tests to verify they pass, then the full file's existing suite**

```bash
.venv/bin/python -m pytest tests/test_earnings_alpha.py tests/test_analyst_alpha.py -v
.venv/bin/python -m pytest tests/test_phase6_signals.py -v   # or wherever insider coverage lives
```

Expected: new tests PASS, and every pre-existing test in these files still PASSES (this fix must
not change behavior for already-midnight `signal_date` values, which is the common case).

- [ ] **Step 6: Syntax check and commit**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/features/feature_defs.py').read())"
git add ascent/features/feature_defs.py tests/test_earnings_alpha.py tests/test_analyst_alpha.py <insider test file>
git commit -m "fix(features): normalize signal_date to midnight in 3 panel builders

build_earnings_panel/build_analyst_panel/build_insider_panel tz-stripped
signal_date but never normalized it to midnight, unlike their siblings
build_options_panel/build_short_panel. Since price data is reindexed
against a midnight-normalized index, any non-midnight signal_date never
matched -- the root cause of earnings/analyst scoring all-NaN in the
proof-audit's real-data run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: Fix `save_parquet` dropping the index on wide-format caches

**Why:** `save_parquet` (`ascent/data/store/parquet.py`) does `pd.concat([old, df],
ignore_index=True)` then `df.to_parquet(path, index=False)` — both steps discard any index. For
`prices_macro`/`prices_international`/`prices_alternatives` (wide format: date lives only in the
index, no `date` column), every save permanently loses all date information. Confirmed on disk:
`RangeIndex` on reload, implausible row counts (176k/151k/150k rows for 9-13 symbols).

**Files:**
- Modify: `ascent/data/store/parquet.py` (`save_parquet`, possibly `load_parquet`)
- Modify (if Task 2's investigation finds this is the correct restore point):
  `agents/macro_agent.py`, `agents/international_agent.py`, `agents/alternatives_agent.py`
- Test: `tests/test_parquet_store_dedup.py`

**Interfaces:**
- Consumes: nothing new
- Produces: `save_parquet(df, name)` and `load_parquet(name)` keep their existing signatures.

- [ ] **Step 1: Read the current exact implementation and existing tests**

```bash
sed -n '1,160p' ascent/data/store/parquet.py
cat tests/test_parquet_store_dedup.py
```

- [ ] **Step 2: Investigate — do not assume the fix is "branch the final `to_parquet` call"**

Reproduce the bug directly:
```bash
.venv/bin/python -c "
import pandas as pd
from ascent.data.store.parquet import save_parquet, load_parquet
df = pd.DataFrame({'AAPL': [100.0, 101.0], 'MSFT': [200.0, 201.0]},
                   index=pd.date_range('2025-01-01', periods=2, freq='D'))
save_parquet(df, '_test_wide_repro')
back = load_parquet('_test_wide_repro')
print(type(back.index), back.index[:3])
"
```

Confirm this reproduces a `RangeIndex` on reload (or similar index loss) EVEN IF you mentally
patch the final `to_parquet(..., index=False)` line to `index=True` — because
`pd.concat([old, df], ignore_index=True)` on the SECOND call (once `path.exists()`) would
already have discarded the index before reaching that line. Confirm this concat-level loss is
real by tracing the code path, not by assumption.

Decide the correct minimal fix based on what you find. The most likely correct approach,
matching this repo's own established pattern of reusing existing machinery rather than adding a
parallel path: at the TOP of `save_parquet`, if `isinstance(df.index, pd.DatetimeIndex)` and no
`date`/`symbol`-style id column is already present, convert to a `date`-column frame via
`df = df.reset_index().rename(columns={df.index.name or "index": "date"})` — this makes the
EXISTING `id_cols`/calendar-day-dedup/concat logic handle it correctly with zero further changes
to that logic, since `date` will now be picked up by the existing `if "date" in
combined.columns:` calendar-day-key branch. Then, since the rest of `save_parquet` already
writes with `index=False` (now correctly, since `date` is a real column), NO change to the final
`to_parquet` line may be needed at all — verify this by testing, don't assume.

If this approach is correct, `load_parquet` does NOT need format-detection heuristics added
(risky for a shared, generic function used by every cache in the system) — instead, the three
`_fetch_*_prices` functions in `agents/macro_agent.py`/`international_agent.py`/
`alternatives_agent.py` need one line each after their `load_parquet(...)` call to restore the
`DatetimeIndex` from the now-preserved `date` column, e.g. `cached =
cached.set_index("date")` (match the exact column name your fix uses). Confirm each of these
three functions' current code to find the exact right insertion point before editing.

If your investigation finds a DIFFERENT correct fix location than described above, use it —
this section is a strong hypothesis from prior research, not a mandate. Explain your reasoning
in your report either way.

- [ ] **Step 3: Write the failing tests**

Add to `tests/test_parquet_store_dedup.py`, matching its existing fixture/tmp-cache-dir style:

```python
def test_datetime_index_survives_round_trip(...):
    """A wide-format DataFrame's DatetimeIndex must not be lost across save/load."""
    df = pd.DataFrame({"AAPL": [100.0, 101.0], "MSFT": [200.0, 201.0]},
                       index=pd.date_range("2025-01-01", periods=2, freq="D"))
    save_parquet(df, "test_wide_cache")
    back = load_parquet("test_wide_cache")
    # Assert however your fix actually restores it -- e.g. a `date` column present,
    # or (if you added restore logic elsewhere) confirm what load_parquet alone returns here.


def test_rangeindex_dataframe_still_saves_without_index_column():
    """Regression guard: existing long-format callers must see byte-identical behavior."""
    df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "date": [...], "close": [100.0, 200.0]})
    save_parquet(df, "test_long_cache")
    back = load_parquet("test_long_cache")
    assert isinstance(back.index, pd.RangeIndex)  # unchanged from today
    assert "date" in back.columns  # the column was already there, not index-derived


def test_second_save_of_wide_dataframe_appends_and_dedupes_correctly():
    """The read-modify-write path (path.exists() branch) must also preserve the index."""
    df1 = pd.DataFrame({"AAPL": [100.0]}, index=pd.date_range("2025-01-01", periods=1))
    df2 = pd.DataFrame({"AAPL": [101.0]}, index=pd.date_range("2025-01-02", periods=1))
    save_parquet(df1, "test_wide_append")
    save_parquet(df2, "test_wide_append")
    back = load_parquet("test_wide_append")
    # Assert 2 rows survive with both dates present, not 1 or 0 or duplicated -- this is the
    # test that would have caught the ignore_index=True concat bug specifically.
```

Write these against a temp cache directory (check how the existing tests in this file isolate
`DATA_DIR` — reuse that fixture, don't invent a new isolation mechanism).

- [ ] **Step 4: Run tests to verify they fail, apply the fix, verify they pass**

```bash
.venv/bin/python -m pytest tests/test_parquet_store_dedup.py -v
```

RED first, then implement, then GREEN. Include the full existing test file's other tests in this
run every time — this fix must not break the dedup logic that's already correct for long-format
caches.

- [ ] **Step 5: If the fix requires touching the three agent files, apply those one-line changes**

Re-run each agent's own existing test suite if one exists (`tests/test_macro_agent.py` or
similar — check for one; if none exists, at minimum re-run the full
`.venv/bin/python -c "import ast; ast.parse(...)"` syntax check and note the gap in your report).

- [ ] **Step 6: Syntax check, full-suite run, commit**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/data/store/parquet.py').read())"
.venv/bin/python -m pytest tests/test_parquet_store_dedup.py -v
git add ascent/data/store/parquet.py tests/test_parquet_store_dedup.py <agent files if touched>
git commit -m "fix(data): preserve DatetimeIndex through save_parquet/load_parquet round-trip

save_parquet's pd.concat(..., ignore_index=True) plus its final
to_parquet(index=False) together discarded any DatetimeIndex before ever
reaching disk, silently corrupting the three wide-format specialist-agent
caches (prices_macro/international/alternatives) on every save. <describe
your actual fix here -- reset_index-to-date-column approach or whatever
you actually implemented>.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Repair the three corrupted caches via live re-fetch

**Why:** The corrupted parquet files on disk cannot be repaired after the fact — the date
information is gone, not just misread. `_fetch_macro_prices`/`_fetch_international_prices`/
`_fetch_alternatives_prices` all do a full historical `yf.download(start="2020-01-01", ...)` on
cache miss (confirmed by research, not incremental), so deleting the corrupted caches and letting
the fetch functions run (now fixed by Task 1... Task 2) fully recovers history from Yahoo Finance.

**Files:** none (data operation only)

- [ ] **Step 1: Back up the corrupted caches first** (in case anything about this step needs to
be undone)

```bash
mkdir -p data_cache/.pre_repair_backup_2026-08-13
cp data_cache/prices_macro.parquet data_cache/.pre_repair_backup_2026-08-13/
cp data_cache/prices_international.parquet data_cache/.pre_repair_backup_2026-08-13/
cp data_cache/prices_alternatives.parquet data_cache/.pre_repair_backup_2026-08-13/
```

- [ ] **Step 2: Delete the corrupted caches and trigger a fresh fetch**

Read `agents/macro_agent.py::_fetch_macro_prices` (and the other two) to confirm the exact
trigger condition for a live fetch (likely: cache missing OR `has_data(...)` false OR a staleness
check) — call the real function directly rather than guessing a CLI flag:

```bash
rm data_cache/prices_macro.parquet data_cache/prices_international.parquet data_cache/prices_alternatives.parquet
.venv/bin/python -c "
from agents.macro_agent import _fetch_macro_prices
from agents.international_agent import _fetch_international_prices
from agents.alternatives_agent import _fetch_alternatives_prices
import ascent.config as cfg  # or however these functions get their config/symbol list -- check real signature
print(_fetch_macro_prices(...).shape)   # match real args
print(_fetch_international_prices(...).shape)
print(_fetch_alternatives_prices(...).shape)
"
```

- [ ] **Step 3: Verify the repair**

```bash
.venv/bin/python -c "
from ascent.data.store.parquet import load_parquet
for name in ['prices_macro', 'prices_international', 'prices_alternatives']:
    df = load_parquet(name)
    print(name, type(df.index), len(df), df.index.min() if hasattr(df.index, 'min') else 'n/a', df.index.max() if hasattr(df.index, 'max') else 'n/a')
"
```

Expected: each shows a real `DatetimeIndex` (or a `date` column, depending on Task 2's chosen
restore point) spanning roughly 2020-01-01 to today, with a plausible row count (~2000 rows per
symbol-day, not 150k+).

**Stop and report the actual shapes/date ranges before continuing** — if any cache still looks
wrong, the Task 1/2 fixes may be incomplete; investigate rather than proceeding to Task 4.

- [ ] **Step 4: Remove the backup once verified good, or leave it — controller's call, note
either way in your report**

---

## Task 4: Re-run the full proof audit and report the final extended scorecard

**Files:**
- Regenerate: `outputs/analyst/proof_audit_2026-08-12.json` (or a new dated file if this runs on
  2026-08-13 — check today's date and match the CLI's existing `date.today().isoformat()`
  naming convention, don't hardcode the old filename)

- [ ] **Step 1: Run the real-data CLI**

```bash
PYTHONPATH=. .venv/bin/python scripts/run_proof_audit.py
```

- [ ] **Step 2: Read the full table.** Report every row's verdict, metric, p-value, sample size,
and reason. Compare against the pre-this-plan scorecard (KEEP=2, CUT=8, INSUFFICIENT_DATA=13).
Expect `earnings`, `analyst` to move off `INSUFFICIENT_DATA`, and
`macro_agent`/`international_agent`/`alternatives_agent` to be scored on real, distinct universes
for the first time. If any of these five is STILL `INSUFFICIENT_DATA`, read its `reason` and
confirm it's legitimate (e.g. genuinely sparse real coverage) rather than a remaining wiring gap.

- [ ] **Step 3: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/ -q -W error
```

This is the FULL suite (not just `tests/analyst/`) since Task 1/2 touched shared production
code used well beyond the proof-audit tool — confirm no regression anywhere.

- [ ] **Step 4: Commit the regenerated scorecard**

```bash
git add outputs/analyst/
git commit -m "chore(proof-audit): regenerate scorecard after production bugfixes

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Done criteria

```bash
.venv/bin/python -m pytest tests/ -q -W error
PYTHONPATH=. .venv/bin/python scripts/run_proof_audit.py
```

- Full test suite passes clean under `-W error` (not just `tests/analyst/` — this plan touches
  shared production code).
- `earnings`/`analyst` sleeves show a real verdict or a legitimate (non-bug) `INSUFFICIENT_DATA`
  reason.
- `macro_agent`/`international_agent`/`alternatives_agent` are scored on their real universes,
  not falling back to the shared US-equity matrix.
- `prices_macro`/`prices_international`/`prices_alternatives` parquet files show a real,
  plausible date range and row count on disk.

## Explicitly out of scope

- Any change to sleeve/agent/subsystem scoring math, the verdict rule, or the component fixture
  in `ascent/analyst/proof_audit/`.
- Fetching `altdata_reddit.parquet` or any other missing altdata source.
- Sub-project 2 (target architecture) — this only extends the scorecard it will read.
- Resuming `com.ascentcapital.eod`/`.heartbeat`.
- Any `save_parquet` call site not confirmed broken by this plan's research (the 7 long-format
  sites listed in the design spec must remain byte-identical in behavior).
