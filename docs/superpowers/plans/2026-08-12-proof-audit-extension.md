# Proof Audit Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve as many of the 13 `INSUFFICIENT_DATA` proof-audit rows as real on-disk data
allows, by wiring the six optional `FeatureBuilder` inputs and giving each specialist agent its
own real price universe — closing the measurement gap before sub-project 2 reads the scorecard.

**Architecture:** Two independent, additive changes to `scripts/run_proof_audit.py` and
`ascent/analyst/proof_audit/run.py`'s `__main__` block (kept in sync, matching sub-project 1's
established pattern for these two files). No change to scoring math, the verdict rule, the
component fixture, or the `DegenerateSignalError`/duplicate-agent guards.

**Tech Stack:** Python 3.12.13, `.venv/bin/python`, pandas, pytest.

## Global Constraints

- Always use `.venv/bin/python`. Never bare `python`.
- No new production write paths — this is read-only analysis tooling, same as sub-project 1.
- Every optional data load is independently guarded with `has_data(name)` from
  `ascent/data/store/parquet.py:29` — a missing cache degrades that one sleeve/agent to
  `INSUFFICIENT_DATA` via the existing guard machinery, never a crash.
- `_dedupe_prices_by_calendar_day` (already defined once in `ascent/analyst/proof_audit/run.py`,
  imported by `scripts/run_proof_audit.py`) must be reused for every price matrix this plan
  loads — do not write a second copy.
- `run.py` and `scripts/run_proof_audit.py` must stay in sync on this logic, mirroring how
  sub-project 1 kept their `__main__`/`main()` bodies equivalent.
- Report the real final scorecard totals precisely — do not round or approximate.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `ascent/analyst/proof_audit/run.py` (modify `__main__`) | Load 6 optional `FeatureBuilder` inputs | 1 |
| `scripts/run_proof_audit.py` (modify `main`) | Same, mirrored | 1 |
| `ascent/analyst/proof_audit/run.py` (modify `__main__`, possibly `run()`) | Per-agent real price matrices | 2 |
| `scripts/run_proof_audit.py` (modify `main`) | Same, mirrored | 2 |
| `outputs/analyst/proof_audit_2026-08-12.json` (regenerate) | Final real scorecard | 3 |

---

## Task 1: Wire the six optional FeatureBuilder inputs

**Why:** `fundamental`, `earnings`, `analyst`, `options_flow`, `insider`, `short_interest` all
need a `features` dict key that `FeatureBuilder` only produces when given the matching optional
DataFrame. All six source caches exist on disk (`data_cache/fundamentals.parquet`,
`earnings.parquet`, `analyst_revisions.parquet`, `options_flow.parquet`,
`insider_transactions.parquet`, `short_interest.parquet`), confirmed populated during this
sub-project's research phase.

**Files:**
- Modify: `ascent/analyst/proof_audit/run.py` (`__main__` block)
- Modify: `scripts/run_proof_audit.py` (`main()`)
- No new test file — this is CLI data-loading glue, consistent with how sub-project 1's Task 8
  and its dedup fix were verified (real-data run, not a unit test asserting a specific dataset).

**Interfaces:**
- Consumes: `has_data(name: str) -> bool` and `load_parquet(name: str) -> pd.DataFrame`
  (`ascent/data/store/parquet.py`), `FeatureBuilder.__init__`'s existing optional kwargs
  (`ascent/features/build_features.py:16`) — no interface changes, only new call-site arguments.

- [ ] **Step 1: Update `ascent/analyst/proof_audit/run.py`'s `__main__` block**

Find the current block (added in sub-project 1, modified in its final-review fix wave — read the
file first to see its exact current state, since the dedup-fix and dates-hoisting changes already
touched this block). Add, before the `FeatureBuilder(...)` call:

```python
    from ascent.data.store.parquet import has_data

    fundamentals_df = load_parquet("fundamentals") if has_data("fundamentals") else None
    earnings_df = load_parquet("earnings") if has_data("earnings") else None
    analyst_df = load_parquet("analyst_revisions") if has_data("analyst_revisions") else None
    options_df = load_parquet("options_flow") if has_data("options_flow") else None
    insider_df = load_parquet("insider_transactions") if has_data("insider_transactions") else None
    short_df = load_parquet("short_interest") if has_data("short_interest") else None
```

Then change the `FeatureBuilder(price_df).compute_features()` call to:

```python
    features = FeatureBuilder(
        price_df,
        fundamentals_df=fundamentals_df,
        earnings_df=earnings_df,
        analyst_df=analyst_df,
        options_df=options_df,
        insider_df=insider_df,
        short_df=short_df,
    ).compute_features()
```

- [ ] **Step 2: Mirror the identical change in `scripts/run_proof_audit.py`'s `main()`**

Same six-line load block and the same `FeatureBuilder(...)` call signature change.

- [ ] **Step 3: Syntax check both files**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/run.py').read())"
.venv/bin/python -c "import ast; ast.parse(open('scripts/run_proof_audit.py').read())"
```

- [ ] **Step 4: Run the full `tests/analyst/` suite — confirm no regression**

```bash
.venv/bin/python -m pytest tests/analyst/ -v -W error
```

Expected: same 56 tests pass as before this task (this task touches no code any unit test
exercises directly — the `__main__`/`main()` blocks are integration-only, verified by the
real-data run in Task 3, not unit tests).

- [ ] **Step 5: Commit**

```bash
git add ascent/analyst/proof_audit/run.py scripts/run_proof_audit.py
git commit -m "feat(proof-audit): wire fundamentals/earnings/analyst/options/insider/short data

Six alpha sleeves (fundamental, earnings, analyst, options_flow, insider,
short_interest) needed FeatureBuilder inputs the CLI never loaded. All six
source caches exist on disk; wiring them is a pure load_parquet swap-in,
matching ascent/main.py's own has_data-guarded pattern.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 2: Give each specialist agent its own real price universe

**Why:** `macro_agent`, `international_agent`, and `alternatives_agent` were all fed the shared
938-symbol US-equity `prices` matrix in sub-project 1, producing bit-identical or all-NaN
signals (caught and correctly flagged `INSUFFICIENT_DATA` by the duplicate-agent guard added in
sub-project 1's final review) instead of ever seeing their real universes. `score_agent(name,
prices)` (`ascent/analyst/proof_audit/wf_scorer.py`) already takes `prices` per call — this is a
caller-side fix in `run.py`'s dispatch, not a change to `score_agent` or `agent_signals.py`
themselves.

**Files:**
- Modify: `ascent/analyst/proof_audit/run.py` (`run()`'s `kind == "agent"` dispatch branch, and
  its `__main__` block to build the per-agent prices)
- Modify: `scripts/run_proof_audit.py` (`main()`, mirrored)

**Interfaces:**
- Consumes: `run()`'s existing signature. This task ADDS an optional parameter —
  `run(features, prices, out_path=None, agent_prices=None)` where `agent_prices: dict[str,
  pd.DataFrame] | None` maps agent component name to that agent's own deduped, pivoted price
  matrix. When `agent_prices` is `None` or missing a key, `run()` falls back to the shared
  `prices` (preserves the old behavior for any direct caller/test that doesn't pass it — do not
  break `tests/analyst/proof_audit/test_run.py`'s existing calls to `run()`).
- Produces: the `kind == "agent"` branch calls `score_agent(c.name, (agent_prices or {}).get(c.name, prices))` instead of always `score_agent(c.name, prices)`.

- [ ] **Step 1: Read `run.py`'s current `run()` function and `__main__` block in full**

Confirm the exact current signature and dispatch logic before editing (it was modified twice
already in sub-project 1 — by the dates-hoisting fix and the duplicate-agent-score fix — read the
live file, don't assume the plan's Task 7/Critical-1-fix descriptions are still verbatim-current).

- [ ] **Step 2: Add the `agent_prices` parameter to `run()`**

```python
def run(features: dict, prices, out_path: Path | None = None, agent_prices: dict | None = None) -> list[ScorecardRow]:
    ...
    elif c.kind == "agent":
        this_agent_prices = (agent_prices or {}).get(c.name, prices)
        rows.append(_row_from_result(c.name, c.kind, c.method, score_agent(c.name, this_agent_prices, ...)))
```

Match this against the ACTUAL current dispatch code in the file (it now includes the
`dates=`-threading and duplicate-agent-flagging logic from sub-project 1's final review — extend
that code, don't replace it wholesale). The duplicate-agent-score check that runs after the loop
should still apply to agent rows regardless of which prices matrix scored them.

- [ ] **Step 3: Build and pass `agent_prices` in both `__main__` blocks**

In `ascent/analyst/proof_audit/run.py`'s `__main__` block and `scripts/run_proof_audit.py`'s
`main()`:

```python
    agent_price_caches = {
        "macro_agent": "prices_macro",
        "international_agent": "prices_international",
        "alternatives_agent": "prices_alternatives",
    }
    agent_prices = {}
    for agent_name, cache_name in agent_price_caches.items():
        if has_data(cache_name):
            agent_price_df = load_parquet(cache_name)
            agent_price_df = _dedupe_prices_by_calendar_day(agent_price_df)
            agent_prices[agent_name] = pivot_prices(agent_price_df, field="close")
```

Then pass `agent_prices=agent_prices` into the `run(...)` call.

- [ ] **Step 4: Syntax check both files**

```bash
.venv/bin/python -c "import ast; ast.parse(open('ascent/analyst/proof_audit/run.py').read())"
.venv/bin/python -c "import ast; ast.parse(open('scripts/run_proof_audit.py').read())"
```

- [ ] **Step 5: Add a regression test for the fallback behavior**

Add to `tests/analyst/proof_audit/test_run.py`: a test that calls `run(features, prices)`
WITHOUT `agent_prices` (matching every pre-existing call in this test file) and confirms it
still works exactly as before — this pins that the new parameter is additive, not breaking.

```python
def test_run_without_agent_prices_falls_back_to_shared_prices(...):
    # reuse this file's existing fixture pattern for features/prices
    rows = run(features, prices)  # no agent_prices kwarg
    assert rows  # doesn't crash, produces the same 23-row shape as before
```

Write this using whatever fixture helpers already exist in `test_run.py` — read the file first
to match its established synthetic-data pattern (real variance, no near-degenerate scipy inputs,
consistent with this codebase's established testing discipline).

- [ ] **Step 6: Run the full test suite**

```bash
.venv/bin/python -m pytest tests/analyst/ -v -W error
```

Expected: all prior tests plus the new one pass, clean under `-W error`.

- [ ] **Step 7: Commit**

```bash
git add ascent/analyst/proof_audit/run.py scripts/run_proof_audit.py tests/analyst/proof_audit/test_run.py
git commit -m "feat(proof-audit): score each specialist agent on its own real universe

macro_agent/international_agent/alternatives_agent were all fed the shared
US-equity prices matrix, producing bit-identical or all-NaN signals (caught
by the duplicate-agent guard, but never a real measurement). score_agent
already took prices per-call -- this is a caller-side fix: run() now
accepts an optional agent_prices dict and the CLI builds it from each
agent's real production cache (prices_macro/international/alternatives).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Task 3: Re-run the full audit and commit the final scorecard

**Files:**
- Regenerate: `outputs/analyst/proof_audit_2026-08-12.json`
- No code changes in this task — verification and artifact only.

- [ ] **Step 1: Run the real-data CLI**

```bash
PYTHONPATH=. .venv/bin/python scripts/run_proof_audit.py
```

- [ ] **Step 2: Read the full printed table.** Report every row's verdict, metric, p-value,
sample size, and reason (where present). Compare against sub-project 1's scorecard (KEEP=2,
CUT=6, INSUFFICIENT_DATA=15) — expect most of the 13 originally-`INSUFFICIENT_DATA` rows to now
carry a real verdict. If any of the 6 newly-wired sleeves or 3 re-universed agents is STILL
`INSUFFICIENT_DATA`, read its `reason` field and confirm it's legitimate (e.g. genuinely sparse
real coverage for that sleeve) rather than another wiring gap — investigate before treating the
run as done.

- [ ] **Step 3: Run the full test suite once more**

```bash
.venv/bin/python -m pytest tests/analyst/ -v -W error
```

- [ ] **Step 4: Commit the regenerated scorecard**

```bash
git add outputs/analyst/proof_audit_2026-08-12.json
git commit -m "chore(proof-audit): regenerate scorecard with extended data coverage

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Done criteria

```bash
.venv/bin/python -m pytest tests/analyst/ -v -W error
PYTHONPATH=. .venv/bin/python scripts/run_proof_audit.py
```

- All tests pass clean under `-W error`.
- The printed scorecard shows a real (not `INSUFFICIENT_DATA`) verdict for as many of the 23
  components as the on-disk data genuinely supports — report the exact final counts precisely,
  don't round.
- No component that previously had a real measurement (meanrev, statarb, trend, volatility, the
  4 subsystems) changed verdict as a side effect of this extension — this task only ADDS
  measurement coverage, it must not perturb sleeves/subsystems that were already scored.

## Explicitly out of scope

- Fetching or fixing `altdata_reddit.parquet` (missing) — `altdata_alpha` degrades gracefully on
  its own, unrelated to this plan.
- Any change to `ml`, `llm_fundamental`, `narrative` (still deferred) or to the verdict rule,
  component fixture, or scoring math.
- Sub-project 2 (target architecture) — this plan only produces the scorecard it will read.
