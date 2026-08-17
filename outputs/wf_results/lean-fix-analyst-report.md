# analyst/ prototype fixes — report

Commit: `e9f2ffb` on `main` (first-ever commit of `analyst/`, plus the three fixes
and the test-directory rename, all in one commit per the task's own guidance that
committing the whole untracked package alongside the fix was acceptable).

Scope respected: nothing under `ascent/`, `agents/`, `debate/`, or the rest of the
main trading pipeline was touched. (`ascent/alpha/stack.py`, `logs/*`, and
`outputs/debate_log/verdict_2026-04-12.json` showed as modified in `git status` at
session start/during the session — not touched by me, not staged, not committed;
almost certainly the background dashboard-automation sweep CLAUDE.md warns about.)

---

## Task 1 — ticker-substitution silent failure

**Root cause, precisely.** `analyst/toolkit.py::load_prices` already raised
`ValueError` loudly on a genuinely empty yfinance result — that path was never the
hole. The hole was one layer up, in the *repair loop*
(`analyst/execute.py::execute_node` → `analyst/codegen.py::generate_one`). When
attempt 1's code called `toolkit.load_prices("NOTAREALTICKER123", ...)` and got a
`ValueError`, the traceback was handed to the LLM as `prior_error` and it was asked
to return corrected code. Nothing in the contract required the retry to *keep*
asking for `NOTAREALTICKER123` — a plausible "fix" is to just swap in a ticker that
works. The regenerated code called `toolkit.load_prices("CL=F", ...)`, got back a
perfectly-shaped, perfectly real `DataFrame`, and `validate_output`
(`execute.py:33-59` at the time) checked only `isinstance`, non-empty, and declared
columns — all of which a *different real asset's* data satisfies. The node reported
`done` in 2 attempts, silently answering the question with the wrong asset's prices.
This matches the HANDOFF anecdote exactly (`NOTAREALTICKER123` → `CL=F`, 2 attempts,
shape-perfect).

**Fix.**
- `analyst/toolkit.py::load_prices` now stamps `df.attrs["ticker"] = ticker` after
  every successful load — on the fresh-fetch path *and* the cache-hit path (parquet
  does not persist `.attrs`, so the cache-hit path needs its own re-stamp, sourced
  from the same ticker argument that built the cache key).
- `analyst/types.py::Task` gained `expected_ticker: str | None = None`.
  `analyst/plans.py::build_event_move_plan` sets it on both LOAD tasks
  (`load_asset_a`, `load_asset_b`) to the actual ticker each one is supposed to
  fetch.
- `analyst/execute.py::validate_output` gained an identity postcondition: for any
  task with `expected_ticker` set, it checks `value.attrs.get("ticker") ==
  task.expected_ticker` *before* the shape checks, and raises a new
  `IdentityError(ValidationError)` naming both the expected and actual ticker on
  mismatch — including when a DataFrame carries no ticker stamp at all (e.g. a
  generated task that bypassed `toolkit` entirely).

This closes the hole at the boundary the HANDOFF suggested ("the safer fix is at
the cache/fetch boundary") without requiring semantic validation of prices
themselves — it only asserts that generated code actually used the ticker the plan
named.

**Test**: `tests/analyst/test_execute.py`. Nine tests, all passing:
- `validate_output` accepts a matching ticker, rejects a substituted one (message
  names the requested `NOTAREALTICKER123`), rejects missing ticker metadata.
- `IdentityError` is a `ValidationError` subclass (so existing `except
  ValidationError` handling in the repair loop still catches it).
- `toolkit.load_prices` raises `ValueError` naming the ticker on an empty
  yfinance result (mocked `yf.download`, no network).
- `toolkit.load_prices` stamps `attrs["ticker"]` on both the fresh-fetch and
  cache-hit paths (mocked `yf.download` + `tmp_path` cache dir, no network).
- **The exact reproduction**: `test_execute_node_fails_loudly_instead_of_silently_substituting`
  drives `execute_node` with attempt-1 code that requests `NOTAREALTICKER123`
  (raises `ValueError`, mocked, no network) and a mocked `codegen.generate_one`
  that reproduces the original bug's "fix" — regenerated code that requests
  `CL=F` instead. Before this fix that would have reported `NodeState.DONE`.
  After it: `NodeState.FAILED`, 2 attempts, `repair_history` has 2 entries (the
  genuine fetch failure, then the identity mismatch), and the node never accepts
  the substituted data.

```
$ .venv/bin/python -m pytest tests/analyst/test_execute.py -v
9 passed in 0.83s
```

---

## Task 2 — preserve repair history

`analyst/types.py::NodeResult` gained `repair_history: list[str] =
field(default_factory=list)`. `execute.py::execute_node` appends each attempt's
formatted traceback to it on every failure (success or not), and no longer treats
a successful repair as erasing the past — `result.error` is still cleared to `""`
on success (it documents only the latest attempt), but `repair_history` is never
cleared.

`analyst/report.py::write_report` was extended with a small `repaired_note` block:
when `res.attempts > 1 and res.repair_history`, the figure/table sections now
render a one-line note (`*Self-healed after N attempts. Earlier failures: ...*`)
using the last line of each prior traceback. `run.json`'s per-task summary also
gained a `repair_history` array (truncated to 800 chars per entry, matching the
existing `error` truncation convention). This was a small, in-place addition, not a
new report section — per the task's own "don't over-build" instruction.

**Test**: `tests/analyst/test_repair_history.py`, 2 tests:
- A node that fails once (`RuntimeError('boom')`) then succeeds on repair reports
  `DONE`, `attempts == 2`, `error == ""`, and `repair_history == [<the boom
  traceback>]`.
- A node that succeeds on the first attempt has `repair_history == []` (no
  spurious entries).

```
$ .venv/bin/python -m pytest tests/analyst/test_repair_history.py -v
2 passed in <0.1s>
```

---

## Task 3 — `tests/analyst/` naming collision

Verified before touching anything: `tests/analyst/test_catalog_registry.py`
imports `from ascent.analyst.catalog import registry` and asserts on the 5
counterfactual tracks (`counterfactual.track_astar` etc). `tests/analyst/proof_audit/`
(9 files) imports from `ascent.analyst.proof_audit.*` (components, stats,
forward_returns, sleeve_signals, wf_scorer, agent_signals, counterfactual_scorer,
scorecard, run). None of the 12 files import anything from the standalone
`analyst/` package this task fixes. Confirmed collision as described.

Renamed `tests/analyst/` → `tests/ascent_analyst/` (`git mv`, both the top-level
file and the `proof_audit/` subdirectory), matching the actual package path
(`ascent/analyst/...`) it covers. Grepped the whole repo (excluding `.venv`) for
`tests.analyst` / `tests/analyst` references in `.py`/`.cfg`/`.ini`/`.toml`/`.yml`
files — none found; `pyproject.toml`'s `testpaths = ["tests"]` is generic and
needed no change. The only hits were prose in historical
`docs/superpowers/plans/*.md` describing already-completed past work — left
alone as a historical record, not live config.

`tests/analyst/` was then recreated fresh for the two new test files
(`test_execute.py`, `test_repair_history.py`) covering the standalone package —
now unambiguous, since the collision is resolved.

```
$ .venv/bin/python -m pytest tests/ascent_analyst/ -q
64 passed in 0.87s
```

---

## Other verification

- `ast.parse` run on every touched `.py` file after patching — clean.
- `analyst_cache/` added to `.gitignore` (next to the existing `data_cache/`
  pattern) and not committed; the 6 pre-existing parquet files under it stay
  untracked.
- `python -m analyst.cli --help` smoke check: prints usage correctly, no import
  errors.
- Ran the full test suite outside `tests/analyst/` and `tests/ascent_analyst/`
  (`pytest tests/ -q --ignore=tests/analyst --ignore=tests/ascent_analyst`): first
  failure hit was `tests/data/test_new_ingest.py::test_fetch_ff_factors_returns_dataframe`
  (`AttributeError: ... famafrench_factors ... does not have the attribute
  '_get_obb'`) — unrelated to `analyst/`, not touched by this session, pre-existing.
  Did not chase further since it's out of scope; flagging it here rather than
  silently ignoring it.

## Files touched

- `analyst/types.py` — `Task.expected_ticker`, `NodeResult.repair_history`
- `analyst/toolkit.py` — ticker identity stamping (fresh + cache-hit paths)
- `analyst/plans.py` — sets `expected_ticker` on both LOAD tasks
- `analyst/execute.py` — `IdentityError`, postcondition in `validate_output`,
  repair-history accumulation in `execute_node`
- `analyst/report.py` — self-heal note in markdown, `repair_history` in `run.json`
- `.gitignore` — added `analyst_cache/`
- `tests/analyst/test_execute.py`, `tests/analyst/test_repair_history.py` — new
- `tests/analyst/` → `tests/ascent_analyst/` (rename, 12 files)
