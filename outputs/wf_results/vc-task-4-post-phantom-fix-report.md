# Task 4 — Walk-forward re-validation after `prices_live` phantom-row fix

**Date**: 2026-08-15
**Invocation** (identical across all three runs, per `wf_run_target_architecture_2026-08-14_fixed_still_broken.log`):
```
.venv/bin/python -c "from ascent.research.walk_forward_runner import walk_forward_pipeline; walk_forward_pipeline()"
```
**New run log**: `outputs/wf_results/wf_run_target_architecture_2026-08-15_post_phantom_fix.log` (32,951 lines)

Prior commits merged into this run: `b628885` (diagnosis), `9fd74ea` (repair `prices_live.parquet`),
`9f145fc` (defense-in-depth hardening of `pivot_prices`/`validate_cache`).

## Headline numbers, side by side

| Metric | BROKEN (08-14, before repair) | fixed_still_broken (08-14, alpha-fix only) | **post_phantom_fix (08-15, this run)** |
|---|---|---|---|
| Total Return | -10.40% | -8.65% | **+88.26%** |
| CAGR | -0.84% | -0.69% | **+10.20%** |
| Volatility | 1.93% | 2.26% | **24.58%** |
| Sharpe Ratio | -0.433 | -0.305 | **0.415** |
| Sortino Ratio | -0.184 | -0.129 | 0.551 |
| Max Drawdown | -13.13% | -11.45% | **-45.65%** |
| Calmar Ratio | -0.064 | -0.060 | 0.223 |
| Hit Rate | 4.6% | 4.7% | **52.3%** |
| Profit Factor | 0.79 | 0.84 | 1.11 |
| Trading Days | 3291 | 3291 | 1641 |
| Benchmark CAGR | +0.00% | +0.00% | **+13.82%** |
| Alpha | -0.84% | -0.69% | -3.62% |
| Beta | 0.000 | 0.000 | **0.947** |
| Excess Sharpe | -0.433 | -0.305 | -0.222 |
| Avg Turnover | 3.14%/day | 3.49%/day | 10.01%/day |
| Avg Positions | 11.9 | 11.9 | 10.9 |
| Total folds | 330 | 330 | 165 |

## The specific tell: benchmark CAGR / beta

Both prior runs reported **Benchmark CAGR: +0.00%** and **Beta: 0.000** — a dead giveaway that the
benchmark return series was empty/degenerate (a direct consequence of the phantom-duplicate rows
corrupting `prices_live`'s date index, which fed a benchmark series with effectively zero valid
observations). This run reports **Benchmark CAGR: +13.82%, Beta: 0.947** — both non-zero and in a
plausible range for a long-biased ~11-name equity book benchmarked against a broad index. **This
anomaly is resolved.**

## Rebalance cadence

`ascent/config/settings.py` sets `rebalance_freq_days: int = 10` (trading days).

- **Prior (broken) runs**: consecutive `Date` lines ~7-8 calendar days apart (e.g.
  2020-01-01 → 01-08 → 01-15 → 01-23 → 01-30 → 02-06 → 02-13 → 02-23 → 03-01 → 03-08), roughly
  weekly — about half the intended ~14-calendar-day (10-trading-day) cadence. 330 folds total.
- **This run**: consecutive `Date` lines are consistently ~14-15 calendar days apart (e.g.
  2020-01-02 → 01-16 → 01-31 → 02-14 → 03-02 → 03-16 → 03-30 → 04-14 → 04-28 → 05-12 → 05-27 →
  06-10), matching the intended 10-trading-day cadence. 165 folds total — almost exactly half
  the prior fold count, consistent with the interval having doubled to the correct value.

**Cadence anomaly is resolved.**

## `valid rows: 0` anomaly

Exact-substring count of `valid rows: 0` in each log:

| Run | Count |
|---|---|
| BROKEN (08-14) | 327 |
| fixed_still_broken (08-14) | 327 |
| **post_phantom_fix (08-15)** | **2** |

Previously nearly every fold (327 of 330, ~99%) hit `valid rows: 0` when targets were injected —
i.e. the target panel was present in shape but empty of usable rows, because the phantom-duplicate
corruption broke the date alignment used to build training targets. Now only 2 of 165 folds show
it (both early folds still in the "partial history"/warm-up window at the very start of the
backtest, where a small valid-row count is expected regardless of data quality). **This anomaly is
effectively gone** (99% → ~1%, and the remainder is explainable by legitimate warm-up).

## Other observations

- All 165 folds succeeded (0 skipped, 0 failed), same "clean" fold-completion behavior as before —
  the difference is in the *content* of those folds, not their success/failure status.
- A `ModuleNotFoundError: No module named 'ascent.dashboard.export_dashboard_data'` traceback
  appears in this run's log (and gives the outer Python process a non-zero exit code). This is
  **not new** — the identical traceback at the identical point (dashboard export, which runs after
  the OOS performance report has already been fully computed and printed) is present verbatim in
  both prior "broken" logs too. It is a pre-existing, unrelated dead import in the post-report
  dashboard-export step; it does not affect the walk-forward numbers above, all of which were
  printed to the log before that import was reached. Not in scope for this task, flagging for
  awareness only.
- Avg turnover roughly tripled (3.1-3.5%/day → 10.0%/day) and volatility roughly 10x'd
  (1.9-2.3% → 24.6%) versus the prior broken runs. Given the prior runs' near-zero volatility and
  4.6-4.7% hit rate were themselves artifacts of a benchmark/target series that was mostly empty
  (not a real low-turnover strategy), this is expected: the corrupted runs were not measuring a
  real quiet strategy, they were measuring noise from starved data. The new volatility (24.6% ann.)
  and turnover are within normal ranges for a concentrated ~11-name long equity book trading
  roughly biweekly.

## Verdict

**These numbers look like a plausible real 2-sleeve equity strategy result, not an obviously
broken pipeline.** Sharpe 0.415, hit rate 52.3%, non-zero benchmark CAGR/beta with beta ≈ 0.95
(sensible for a long-only concentrated book), negative, benchmark-lagging alpha (-3.62%,
consistent with CLAUDE.md's documented raw-return lag vs SPY from defensive sleeves/vol-targeting/
200MA cuts — though note this specific run is the *equity-only* 2-sleeve walk-forward, not the
full multi-sleeve production stack), and a max drawdown (-45.65%) that is large but falls within
the range a leveraged/concentrated equity strategy could plausibly show over a multi-year window
including 2020 and 2022 drawdowns. The specific corruption tells (benchmark CAGR/beta pinned at
zero, hit rate near-random at 4.6-4.7%, `valid rows: 0` on 99% of folds, half-cadence rebalancing)
are all gone or resolved to expected/negligible levels.

**This verdict does not authorize resuming live trading.** That decision belongs to the
controller, who should weigh the -45.65% max drawdown, the -3.62% alpha vs benchmark, and any
other production-readiness criteria not covered by this walk-forward check alone.

---
**Report-back headline (verbatim for controller)**: Sharpe 0.415 / Hit Rate 52.3% / CAGR +10.20%
/ Benchmark Beta 0.947. Verdict: numbers look like a plausible real 2-sleeve equity strategy — the
prior corruption tells (zero benchmark CAGR/beta, near-random hit rate, 99%-of-folds
`valid rows: 0`, half-cadence rebalancing) are resolved. Not an authorization to resume live
trading.
