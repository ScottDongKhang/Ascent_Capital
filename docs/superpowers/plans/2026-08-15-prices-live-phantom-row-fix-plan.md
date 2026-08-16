# Plan — fix `prices_live` phantom-row corruption

Spec: `docs/superpowers/specs/2026-08-15-prices-live-phantom-row-fix-design.md` — read it first,
it is the binding authority for this plan.

## Global Constraints

- Read-only exploration before any write; back up `data_cache/prices_live.parquet` (copy to
  `data_cache/prices_live.parquet.pre_phantom_repair_bak` or similar, do not overwrite the
  original until the repair script is verified) before modifying it.
- Do not change `_EVENING_ROLLOVER_HOUR` / `_calendar_day_key` in
  `ascent/data/store/parquet.py` — out of scope.
- Do not touch `prices_macro`/`prices_international`/`prices_alternatives` caches.
- Never run `git stash` / `git stash pop` in this repo.
- Do not reload `com.ascentcapital.eod`/`.heartbeat` or take any action that resumes live
  trading.
- `.venv` in a fresh worktree needs
  `rm -rf .venv && ln -s /Users/scott/IdeaProjects/ascent-capital/.venv .venv` before running
  anything; `git checkout -- .venv` before merging.
- `data_cache/` and `logs/` are gitignored and worktree-local — this plan's tasks need the real
  `prices_live.parquet`, so do this work directly against the main checkout's
  `data_cache/`, not inside an isolated worktree that wouldn't see it. (Code changes still go
  through normal commits on a feature branch off `main`.)

## Task 1 — Diagnose phantom-row data-loss risk (no writes)

Read `data_cache/prices_live.parquet` with `ascent.data.store.parquet.load_parquet`. Split rows
into "real" (date time-of-day == 00:00:00) and "phantom" (time-of-day != 00:00:00). For every
`(symbol, date.dt.normalize())` pair present in the phantom set, check whether a real row exists
for the same `(symbol, normalized date)`. Report:

- Total phantom rows, total real rows, total distinct normalized dates covered by each.
- Count and list (sample of 20, plus a saved CSV of the full list at
  `outputs/wf_results/phantom_only_cells_2026-08-15.csv`) of `(symbol, date)` pairs where a
  phantom row exists but NO real row exists for that same normalized date — these are the cells
  where dropping the phantom row would destroy otherwise-unrecoverable price data.
- Whether phantom-only cells cluster on particular symbols/date ranges (helps decide if a
  targeted yfinance re-fetch, like checkpoint-4's cache repair, is warranted for those cells).

Write findings to `outputs/wf_results/phantom-row-diagnosis-2026-08-15.md`. This is a read-only
analysis script (put it at `scripts/maintenance/diagnose_phantom_rows.py`, following the pattern
of `scripts/maintenance/repair_mixed_basis_symbols.py`) plus the report. No cache file is
modified in this task.

**Report contract**: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED, commit hash, one-line
summary of phantom-only cell count, path to the full report.

## Task 2 — Repair `prices_live.parquet`

Depends on Task 1's report (read it first — it is in
`outputs/wf_results/phantom-row-diagnosis-2026-08-15.md`).

Write `scripts/maintenance/repair_prices_live_phantom_rows.py`:

1. Back up the current `data_cache/prices_live.parquet` to
   `data_cache/prices_live.parquet.pre_phantom_repair_bak` (fail loudly if the backup already
   exists — don't silently overwrite a prior backup).
2. Load the cache. Normalize every row's `date` to midnight
   (`pd.to_datetime(df["date"]).dt.normalize()`) as the merge key, matching what
   `normalize_prices()` already does on every write path.
3. For `(symbol, normalized_date)` groups with more than one row (i.e. a phantom/real pair or,
   per Task 1, a phantom-only cell): keep the real (00:00) row's OHLCV values when a real row
   exists; when only a phantom row exists for that cell (Task 1's phantom-only list), keep the
   phantom row's values but rewrite its `date` to the normalized midnight timestamp so it
   becomes a normal row going forward.
4. Write the repaired frame back through `save_parquet(df, "prices_live")` (not a raw
   `to_parquet`) so it goes through the existing dedup path and any downstream invariants it
   enforces.
5. Verify post-repair: `pivot_prices(load_parquet("prices_live"), "close").shape` has ~1,650
   rows (real NYSE trading days in the cache's date range), not ~3,298. Assert this in the
   script and print the before/after row counts and pivot shapes.
6. Run `scripts/reconcile_numbers.py`'s data-integrity section (or the specific check it runs
   for `prices_live` duplicates) and record whether it reports 0 non-midnight rows after repair.

**Report contract**: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED, commit hash,
before/after row counts and pivot shape, confirmation the backup file exists, path to the repair
script's log output.

## Task 3 — Prevent recurrence: integrity guard + read-boundary defense

Two independent, small changes — same task, same file family, dispatch together:

1. In `ascent/data/store/parquet.py::validate_cache()`, add a check (only when `date_col`'s
   dtype is datetime-like): report non-OK with a reason string like
   `"N rows have non-midnight time-of-day (phantom-row corruption)"` if any row's
   `date.dt.time != time(0,0)`. Follow the existing return-tuple convention
   (`return False, reason`) used elsewhere in this function — but this is diagnostic-only, so
   gate it: only fail validation when the count exceeds a small tolerance (e.g. 0, since after
   Task 2's repair there should be none) — do not break existing callers that tolerate a handful
   of legitimately-tz-shifted rows elsewhere; read the whole function first to confirm this
   doesn't regress any existing passing check.
2. In `ascent/data/normalize/prices.py::pivot_prices()`, normalize the `date` column
   (`df["date"].dt.normalize()`) before calling `df.pivot_table(...)`, keeping the existing
   `aggfunc="last"` so any remaining same-day collision (from any future corruption) resolves
   deterministically instead of silently fragmenting the index. Do not change `pivot_macro` or
   any other function unless it shares the exact same bug — check first, don't assume.

Add or extend a test (find the existing test file for `ascent/data/normalize/prices.py` or
`ascent/data/store/parquet.py` first — do not create a new test file if one already covers
these) that constructs a small DataFrame with one real (00:00) row and one same-day phantom
(19:00, different symbol) row and asserts `pivot_prices` produces a single index entry for that
date, not two.

**Report contract**: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED, commit hash, test file
and pass/fail output, confirmation that `validate_cache`'s existing callers were checked for
regressions (name which ones).

## Task 4 — Re-run walk-forward validation and report

Depends on Tasks 2 and 3 being merged (their commits must be in the branch this task runs on).

Run `ascent/research/walk_forward_runner.py` the same way the two prior validation runs did
(check `outputs/wf_results/wf_run_target_architecture_2026-08-14_fixed_still_broken.log` for the
invocation command/args used — replicate it exactly so results are comparable). This is a
multi-minute run; let it complete, don't background-and-abandon it (see handoff gotcha: a
subagent that launches a long-running process and ends its turn before it finishes never
reports the result — this task's implementer must wait for the run to actually finish before
writing its report).

Save the full log to `outputs/wf_results/wf_run_target_architecture_2026-08-15_post_phantom_fix.log`.

Write `outputs/wf_results/vc-task-4-post-phantom-fix-report.md` covering:

- Sharpe, hit rate, CAGR, volatility, max drawdown, benchmark CAGR/beta — compare directly
  against the two prior (broken) runs' numbers.
- Whether benchmark CAGR is now non-zero and beta is non-zero (the specific tell that was wrong
  before).
- Rebalance cadence (fold-to-fold date gaps) — should now be consistent with the configured
  `rebalance_freq_days`, not roughly half of it.
- Whether the `[WF] targets injected, ..., valid rows: 0` anomaly is gone or reduced.
- An explicit verdict: do these numbers look like a plausible (not necessarily good) real
  2-sleeve equity strategy, or is something still obviously broken? This verdict does not
  authorize resuming live trading — that decision is the controller's alone, after reading this
  report — but it must be stated clearly so the controller can act on it.

**Report contract**: DONE / DONE_WITH_CONCERNS / NEEDS_CONTEXT / BLOCKED, commit hash (this task
only adds docs/logs, so the commit should contain only the new report + log file), the four key
numbers (Sharpe/hit-rate/CAGR/benchmark-beta) inline in the report-back so the controller doesn't
have to open the file to get the headline result.

## Final whole-branch review

Covers Tasks 1-4 together: script quality, whether the repair script's merge logic actually
matches what Task 1's diagnosis found (not a generic drop-all-phantom-rows script if phantom-only
cells existed), whether `validate_cache`'s new check could break any existing caller, and whether
the Task 4 report's verdict is adequately supported by the numbers it cites.

After a clean final review: update `CHECKPOINTS.md`'s BLOCKER section to reflect resolution (or,
if Task 4's numbers are still implausible, update it with the new finding instead of resolving
it — do not mark the blocker resolved unless Task 4's verdict actually supports it), then use
`superpowers:finishing-a-development-branch`.
