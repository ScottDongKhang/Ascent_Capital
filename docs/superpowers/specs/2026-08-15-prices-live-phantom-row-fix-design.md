# Spec — fix `prices_live` phantom-row corruption breaking walk-forward validation

Date: 2026-08-15
Sub-project: "4b continued" — root-causing the open blocker from
`docs/superpowers/handoffs/2026-08-14-strip-down-rebuild-handoff.md`.

## Problem

Walk-forward validation of the rebuilt 2-sleeve (`meanrev`+`statarb`) alpha stack via
`ascent/research/walk_forward_runner.py` reports Sharpe ≈ -0.3 to -0.43, hit rate ≈ 4.6-4.7%
over 3291 "trading days" / 330 folds — not a real strategy's signature. This persisted after
the alpha-weight runtime-override bug (commits `6628948`, `31d49ee`) was fixed, proving that
bug was real but not the cause of the bad backtest.

## Root cause (established by read-only investigation, evidence below)

`data_cache/prices_live.parquet` contains ~1,648 legacy phantom duplicate-calendar-day rows
alongside ~1,650 real trading-day rows (3298 total date entries where ~1650 is correct):

- Real rows: timestamped 00:00, ~867/938 symbols populated per day.
- Phantom rows: timestamped 19:00/20:00 on (after `_calendar_day_key`'s evening-rollover)
  dates that collide with the *following* day's real row for only the symbols present in the
  phantom row — not the *same* day's real row. Only ~54/938 symbols populated per phantom row.

Verified directly against the live cache (read-only, `ascent/data/store/parquet.py::load_parquet`
+ `ascent/data/normalize/prices.py::pivot_prices`):
```
df['date'].dt.time.value_counts() → 00:00:00: 1,430,450 rows; 19:00/20:00: 88,841 rows
pivot_prices(df, 'close').shape → (3298, 938)   # should be ~1650
```

Mechanism: `ascent/features/build_features.py:27` (`pivot_prices`) pivots on the raw `date`
value with no normalization, so real and phantom rows for the same trading day become distinct
index entries. `walk_forward_runner.py` never normalizes dates either (only strips tz, lines
104-105). The phantom rows flow straight into `close_full`/`all_dates`/`close_prices` and into
`ascent/backtest/engine.py::BacktestEngine.run()`.

`ascent/backtest/engine.py:51-54` — `daily_returns = close.pct_change().fillna(0)`. For any
symbol not in a phantom row's sparse subset, that cell is NaN → `pct_change` from the prior real
close is NaN → filled to 0, and the *next* transition (phantom → next real close) is
NaN → also filled to 0. The real day's price move is silently deleted and replaced by two
fabricated zero-return days, for roughly half of all evaluated days. This crushes measured
volatility (annualized vol reported at 1.93-2.26% in the run logs) and dilutes hit-rate
(`hit_rate = (returns > 0).mean()`) with a flood of exact-0.0 non-hit days.

Corroborating evidence: `SPY` rows exist only at 00:00 (never phantom-populated), so
`bm_data.reindex(common_dates)` injects NaN on every phantom date and the same mechanism zeroes
almost the entire benchmark series — explaining `Benchmark CAGR: +0.00%`, `Beta: 0.000` in both
run logs (`outputs/wf_results/wf_run_target_architecture_2026-08-14_BROKEN.log:2620-2622`,
`outputs/wf_results/wf_run_target_architecture_2026-08-14_fixed_still_broken.log:1960-1962`) — a
benchmark that traded for 6.5 years cannot have a literal 0% CAGR.

Secondary effect: `rebal_dates_set` (`walk_forward_runner.py:186-190`) samples every Nth entry
of the corrupted `all_dates`, roughly halving the intended ~monthly rebalance cadence.

The separately-flagged "valid rows: 0" anomaly in the `ml` sleeve's target construction
(`walk_forward_runner.py:349`) shares this root cause: `forward_return`'s row-based
`close.shift(-21)` almost never lands on two populated cells per symbol when ~half the rows are
near-empty phantom duplicates. Not an independent bug; a canary of the same corruption.

**Why this is legacy, not an active bug**: every current writer of `prices_live`
(`ascent/data/hub.py:227-228`, `ascent/main.py:256-257`) calls
`ascent/data/normalize/prices.py::normalize_prices()` before `save_parquet`, which does
`df["date"] = pd.to_datetime(df["date"]).dt.normalize()` — unconditionally zeroing time-of-day.
No current write path can produce a new phantom row. The ~1,648 rows already on disk are cruft
from before this normalization was universal (or from a blend that bypassed it), and
`save_parquet`'s dedup (`_calendar_day_key`'s evening-rollover, designed for a different,
legitimate case — a hub bar honestly stamped 19:00 for a bar that belongs to the *next* trading
day) does not collide these same-day phantom/real pairs, so dedup has never removed them.

## What this spec does NOT do

- Does not change `_calendar_day_key`'s rollover logic — it exists for a real, different,
  still-relevant case (documented at `ascent/data/store/parquet.py:53-72`) and current writers
  no longer produce non-midnight timestamps, so there is nothing live for it to mis-handle
  going forward.
- Does not touch `prices_macro`/`prices_international`/`prices_alternatives` — already known
  corrupt and already tracked separately (`save-parquet-wide-cache-corruption` memory,
  `CLAUDE.md` "Wide-format caches" gotcha). Out of scope here.
- Does not re-enable live trading. That remains gated on a plausible walk-forward result,
  produced by this fix, per the standing rule in
  `docs/superpowers/handoffs/2026-08-14-strip-down-rebuild-handoff.md`.

## Required work

1. **Diagnose data-loss risk before deleting anything.** For every (symbol, calendar day) where
   a phantom row exists, check whether a same-day real (00:00) row also exists for that symbol.
   If a phantom row is the *only* data for a (symbol, day) pair, dropping it destroys real
   price history for that cell — must be quantified and reported, not assumed away.
2. **Repair `data_cache/prices_live.parquet`**: remove phantom rows. Back up the pre-repair file
   first (mirrors the precedent in checkpoint-4's cache repair). If task 1 finds phantom-only
   cells, decide (and document) how to handle them — most likely: keep the phantom row's value
   for that specific (symbol, day) cell only (rewrite as a 00:00 row), don't blanket-drop it.
3. **Prevent recurrence / silent trust**: add an integrity check (script or `validate_cache`
   extension — follow the existing pattern of `scripts/reconcile_numbers.py`'s data-integrity
   section, which already reports `prices_live` duplicate counts) that reports any `prices_live`
   row with a non-midnight time-of-day, so this class of corruption is measured on every run
   rather than assumed clean. This is the same "measure it, don't trust it" rule already in
   `CLAUDE.md`'s Data / caching section for the existing duplicate-row issue.
4. **Defense-in-depth at the read boundary**: `ascent/data/normalize/prices.py::pivot_prices()`
   should normalize (`.dt.normalize()`) the `date` column before pivoting, with `aggfunc="last"`
   already in place to resolve any remaining same-day collisions deterministically. This makes
   the read path robust to any future cache that (despite task 3's guard) still contains
   non-midnight rows, matching this project's general posture of not trusting caches on faith.
5. **Re-run walk-forward validation** (`walk_forward_runner.py`, same invocation as the two
   prior runs) after the repair and confirm the numbers are plausible for a real (if modest)
   2-sleeve strategy — not proof the strategy is good, just proof the harness itself is no
   longer fabricating a broken signature. Compare benchmark CAGR/beta against known SPY
   performance over the same window as a sanity check (should not be ~0%).
6. **Document**: update `CHECKPOINTS.md`'s BLOCKER section (resolve or replace it with the new
   finding), and leave the live-trading resume gate exactly where it is unless the re-run in
   step 5 actually clears it — that determination belongs to the controller after reading the
   final numbers, not to any implementer subagent.

## Global constraints

- Read-only exploration before any write; back up `prices_live.parquet` before modifying it.
- Do not touch `_EVENING_ROLLOVER_HOUR` / `_calendar_day_key`'s rollover semantics — out of
  scope, still needed for its original documented case.
- Do not touch `prices_macro`/`prices_international`/`prices_alternatives`.
- Do not reload `com.ascentcapital.eod`/`.heartbeat` or otherwise resume live trading.
- Never use `git stash` in this repo (shared stash stack across worktrees/sessions — see
  handoff gotchas).
- Follow the project's standard cycle: this spec → plan → subagent-driven implementation →
  task review → final whole-branch review → merge, per
  `docs/superpowers/handoffs/2026-08-14-strip-down-rebuild-handoff.md`.
