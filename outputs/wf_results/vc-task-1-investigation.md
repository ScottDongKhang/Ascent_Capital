# Task 1 Investigation — Validation & Cutover Plan

Investigation only, no code changes. All commands run with `.venv/bin/python`.

## 1. `walk_forward_pipeline()` signature and defaults

File: `ascent/research/walk_forward_runner.py`, function starts at line 45.

```python
def walk_forward_pipeline(
    train_days=None,
    purge_days=None,
    top_n=None,
    max_weight=None,
    max_per_sector=1,
    regime_engine=None,      # accepted but ignored — a fresh regime engine is
                              # fit per fold on training data only (no look-ahead)
    spy_prices=None,
    univ_prices=None,
    vix_prices=None,
    price_df=None,            # caller can pass data directly
    macro_df=None,            # caller can pass data directly
    prices_cache_name=None,   # override cache name if needed
):
```

`None` defaults for `train_days`, `purge_days`, `top_n`, `max_weight` are resolved from
`get_config()` inside the function body (lines 67–70):

```python
if train_days is None: train_days = cfg.walk_forward.train_days
if purge_days is None: purge_days = cfg.walk_forward.purge_days
if top_n      is None: top_n      = cfg.backtest.top_n
if max_weight is None: max_weight = cfg.backtest.max_weight
```

There is also a hard floor: if `purge_days < TARGET_HORIZON` (21), it is forced up to 21 with a
printed warning, to prevent label leakage.

Calling `walk_forward_pipeline()` with no args (as Task 2's plan does) is exactly the intended
invocation — it always falls through to `get_config()` for these four values, and calls
`build_alpha_stack(hist_features, agent_id="us_equities")` with no weight override, so it uses
the live `DEFAULT_ALPHA_WEIGHTS` (`{"meanrev": 0.50, "statarb": 0.50}`) rather than a
grid-searched blend.

## 2. Config values (measured, not assumed)

Command run:
```
.venv/bin/python -c "
from ascent.config.settings import get_config
c = get_config()
print('train_days', c.walk_forward.train_days)
print('purge_days', c.walk_forward.purge_days)
print('top_n', c.backtest.top_n)
print('max_weight', c.backtest.max_weight)
"
```

Output:
```
train_days 252
purge_days 5
top_n 15
max_weight 0.1
```

Note: `purge_days=5` is below `TARGET_HORIZON=21`, so at runtime the pipeline will print the
`WARNING: purge_days=5 < TARGET_HORIZON=21. Forcing purge_days=21` line and actually run with
`purge_days=21`. Report Task 2's actual effective value from the printed log, not this raw
config value, if there's any doubt.

Additional config used but not asked for, worth noting: `cfg.backtest.rebalance_freq_days`
(drives both fold cadence and the `BacktestEngine`'s rebalance schedule),
`cfg.backtest.initial_capital`, `cfg.backtest.spread_bps`, `cfg.backtest.impact_bps`,
`cfg.backtest.execution_delay_days`, `cfg.universe.benchmark` (SPY).

## 3. Smoke-test / fast-mode option

**None exists.** Checked for CLI args, argparse, a `--smoke` flag, a fold-count limiter, or any
truncation knob:

```
grep -n "argparse\|--smoke\|n_folds\|max_folds\|limit" ascent/research/walk_forward_runner.py
```

No matches except the trailing `if __name__ == "__main__": walk_forward_pipeline()` (line
576–577) — bare call, no args, no CLI surface at all.

**Workaround available but not built-in**: the function accepts `price_df=` and `macro_df=`
directly. A caller could load `prices_live`/`macro_live` from cache, slice both to a short date
range (e.g. the most recent ~400 calendar days, enough for one `train_days=252` fold plus a
handful of OOS rebalances), and pass the truncated frames in to get an end-to-end smoke run in
a couple of minutes instead of the full ~6.5-year history. This is a manual construction, not a
supported flag — worth doing before the full Task 2 run given there is no other pre-flight
check, but it is extra work Task 2 should budget for if it wants one.

**Data availability check** (the other pre-flight risk — missing cache would abort immediately
with `ERROR: No cached data (...)`):

```
.venv/bin/python -c "
from ascent.data.store.parquet import has_data
for name in ['prices_live','macro_live','macro_simulated','profiles','fundamentals','earnings']:
    print(name, has_data(name))
"
```
Output: all `True`. So the pipeline will not abort on missing cache data; that specific risk is
cleared.

**Risk to note for Task 2** since no smoke mode exists: the only way to sanity-check
end-to-end correctness before the full run is (a) the manual truncated-date-range workaround
above, or (b) accepting the risk and watching the first few printed fold lines closely once the
full run starts (the loop prints one line per rebalance date as it goes, so a crash or
all-zero-weights pattern would surface within the first few folds, well before 30 minutes
elapse).

## 4. Metrics helper: `ascent/research/wf_framework/metrics.py::PerformanceAnalyzer`

Class interface (verified by reading `metrics.py` in full):

```python
class PerformanceAnalyzer:
    def __init__(self, rf_annual: float = 0.0, periods_per_year: int = 252): ...

    def cagr(self, returns: pd.Series) -> float
    def volatility(self, returns: pd.Series) -> float
    def sharpe(self, returns: pd.Series) -> float
    def sortino(self, returns: pd.Series) -> float
    def max_drawdown(self, returns: pd.Series) -> float
    def win_rate(self, returns: pd.Series) -> float
    def alpha_beta(self, returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]
    def walk_forward_efficiency(self, fold_results: list[FoldResult]) -> float
    def full_report(self, oos_returns: pd.Series, benchmark: pd.Series | None,
                     fold_results: list[FoldResult]) -> dict
```

`FoldResult` is a small dataclass: `fold_id: int, is_sharpe: float, oos_returns: pd.Series`.

**Input shape needed**: every metric except WFE just needs a plain `pd.Series` of *period
(daily) returns*, ideally datetime-indexed (`cagr()` uses calendar span from the index when
available, falling back to `n / periods_per_year` otherwise). `alpha_beta()` additionally wants
a benchmark returns series aligned by index intersection.

**Can it be built from `walk_forward_pipeline()`'s output without modification? Mostly yes,
with one gap:**

- `walk_forward_pipeline()` writes `ascent_daily_ledger.csv` (`result.daily_ledger.to_csv(...)`
  at line ~552) with `date` as the index and a `net_return` column — exactly a daily returns
  series, directly usable as `PerformanceAnalyzer(...).cagr(net_return_series)`,
  `.sharpe(...)`, `.sortino(...)`, `.max_drawdown(...)`, `.win_rate(...)`. Confirmed by reading
  `ascent/backtest/engine.py` lines 148–199: `daily_rows.append({..., "net_return": ...})` →
  `daily_ledger = pd.DataFrame(daily_rows).set_index("date")`.
- A benchmark series for `alpha_beta()` is *not* written to the CSV, but is available
  in-memory inside `walk_forward_pipeline()` as `bm_data` (SPY close, pct-changed) and is also
  passed into `engine.run(benchmark_prices=bm_data)`; the returned `BacktestResult` exposes it
  as `result.benchmark_returns`. If working only from the CSV artifact (not modifying the
  function), the benchmark series would need to be reconstructed separately from
  `prices_live`/SPY — trivial (`close.pct_change()`), just not sitting in the CSV.
- **WFE is the one metric that cannot be assembled from this pipeline's output as-is.**
  `walk_forward_efficiency()` requires a `list[FoldResult]`, each with an **in-sample Sharpe**
  per fold. `walk_forward_pipeline()`'s own `fold_results` list (built inside its loop, lines
  ~247–256 and ~380–390 `.venv/bin/python`-confirmed by reading the loop) only records
  `{"date", "train", "test_days", "daily_ret", "status"}` — a single OOS day's return per fold,
  never an in-sample Sharpe computed on the training slice. There is no IS backtest step in this
  loop at all (only feature/alpha computation and one OOS weight draw per fold). So
  `PerformanceAnalyzer.walk_forward_efficiency()` cannot be called against this pipeline's
  native fold data without new code to compute an in-sample Sharpe per fold (e.g. by
  backtesting `hist_alpha`/`test_weights` over the training window before scoring OOS) — that
  is genuinely new metric code, not reuse. Task 2 should either add that small amount of
  IS-Sharpe-per-fold code, or report WFE as "not computable from this framework's native
  output" and rely on the other five metrics (CAGR, vol, Sharpe, Sortino, max drawdown) plus
  win rate, alpha, beta — all of which come for free from `ascent_daily_ledger.csv`.

**Practical recipe for Task 2**:
```python
import pandas as pd
from ascent.research.wf_framework.metrics import PerformanceAnalyzer

ledger = pd.read_csv("ascent_daily_ledger.csv", index_col="date", parse_dates=True)
pa = PerformanceAnalyzer()
report = {
    "cagr": pa.cagr(ledger["net_return"]),
    "volatility": pa.volatility(ledger["net_return"]),
    "sharpe": pa.sharpe(ledger["net_return"]),
    "sortino": pa.sortino(ledger["net_return"]),
    "max_drawdown": pa.max_drawdown(ledger["net_return"]),
    "win_rate": pa.win_rate(ledger["net_return"]),
    "n_oos_days": len(ledger),
}
# alpha/beta needs a benchmark series reconstructed from prices_live (SPY), separately.
# wfe: not computable without adding IS-Sharpe-per-fold tracking to walk_forward_pipeline().
```

Also note `walk_forward_pipeline()` already prints its own summary via
`ascent.research.evaluation.format_metrics(result.summary())`, and `BacktestResult.summary()`
calls `ascent.research.evaluation.compute_all_metrics(...)` — a *different*, simpler metrics
function already exercised automatically on every run (not `PerformanceAnalyzer`). Its console
output is a second, free source for cross-checking Sharpe/CAGR/max-drawdown once Task 2 runs,
independent of manually computing from the CSV.

## 5. Dead-folds bug — shared or independent?

**Independent — `walk_forward_runner.py` does not use `AscentPortfolioStrategy` and does not
share the bug.**

Evidence:
- `grep -n "AscentPortfolioStrategy\|ascent_strategy" ascent/research/walk_forward_runner.py`
  → **no matches**. The file never imports the class or the module that defines it.
- `walk_forward_runner.py`'s fold loop (lines ~199–452) is a self-contained `for i, test_date in
  enumerate(all_dates):` loop that calls `FeatureBuilder(...)`, `build_alpha_stack(...)`, and
  `sector_constrained_weighted(...)` directly, per fold, with no class-level cache of any kind —
  every fold recomputes features and alpha from scratch on its own point-in-time slice.
- The actual bug mechanism (confirmed by reading `ascent/research/wf_framework/ascent_strategy.py`
  lines 1–20, 55–75, 150–170) is `AscentPortfolioStrategy`'s **class-level** `_feature_cache` /
  `_alpha_cache` dicts, keyed by `data_key = (all_dates[0], as_of_date, data["symbol"].nunique())`
  — i.e. keyed by symbol *count*, not identity, and without `pit_boundary` distinguishing folds
  that share a symbol count. That collision mechanism lives entirely inside
  `AscentPortfolioStrategy.__init__`/its cache lookup, a class that `walk_forward_runner.py`
  never instantiates.
- `walk_forward_runner.py` instead uses `ascent/backtest/engine.py::BacktestEngine`, a plain
  stateless per-call engine with no fold-to-fold memoization at all (each `daily_rows.append`
  is built fresh from `target_weights`/`close_prices` passed into `engine.run()`).

**Conclusion**: the "3 dead folds" (folds 17, 18, 20 returning `OOS Sharpe = 0.0`) is a
`run_ascent_wf.py` / `AscentPortfolioStrategy`-specific bug. `walk_forward_runner.py` has a
structurally different, independent fold-execution path and does not need this caveat
disclosed for its own results — Task 2 does not need to report a dead-fold adjustment for this
framework's numbers (though it should still watch its own `folds_skipped_thin` /
`failed_folds` diagnostics, which are a different, legitimate mechanism in this file — thin
universe or optimizer failure per fold, printed in the `[WF] FOLD SUMMARY:` block at the end of
the run).

## Summary for Task 2

- Call `walk_forward_pipeline()` with **no arguments** — defaults correctly resolve to
  `train_days=252, purge_days=5→21 (floored), top_n=15, max_weight=0.1` via `get_config()`.
- No smoke-test flag exists; either accept the risk (data availability is already confirmed
  clean) or build one manually by slicing `price_df`/`macro_df` to a short recent window before
  calling.
- Compute Sharpe/Sortino/CAGR/max-drawdown/win-rate directly from
  `ascent_daily_ledger.csv`'s `net_return` column via `PerformanceAnalyzer`, no new code needed.
  Alpha/beta needs a benchmark series reconstructed from `prices_live` SPY closes. WFE is **not**
  computable from this framework's native `fold_results` without adding IS-Sharpe-per-fold
  tracking — flag this gap explicitly in Task 2 rather than fabricating a WFE number.
- The 3-dead-folds bug is `AscentPortfolioStrategy`-specific and does **not** apply to
  `walk_forward_runner.py`'s results — no adjustment needed, no caveat to carry forward from
  that specific bug for this framework.
