"""
ascent/monitoring/live_vs_backtest.py
Compares live paper trading performance against walk-forward backtest expectations.

Reads:
    - ascent_daily_ledger.csv  (walk-forward OOS backtest results)
    - logs/eod_log.jsonl       (live EOD execution logs with portfolio values)

Exports:
    - dashboard/live_vs_backtest.json

Run standalone:
    python3 -m ascent.monitoring.live_vs_backtest
"""

import json
import math
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from ascent.research.deflated_sharpe import probabilistic_sharpe_ratio

# ── Cost-model baseline (see ascent/backtest/costs.py module docstring) ───────
# The backtest cost model and the live Almgren-Chriss estimator
# (ascent/execution/cost_model.py) are measured to diverge by design, not by
# bug: ~5.5bps (backtest) vs ~6.6bps (live) round-trip at low (0.2%)
# participation, widening to ~6.9bps vs ~19.6bps round-trip at high (15%)
# participation. A live-vs-backtest tracking-error / Sharpe divergence of
# roughly this magnitude is an EXPECTED baseline, not evidence of a live
# execution or alpha problem, and should not trip a "something is wrong"
# flag on its own. Expressed as an annualized-return drag range this baseline
# spans roughly (6.6-5.5)=1.1bps to (19.6-6.9)=12.7bps round-trip per rebalance
# — used below only to inform the significance threshold and to caveat the
# output, not to change how tracking_error_ann itself is computed.
COST_MODEL_BASELINE_BPS_LOW = 6.6 - 5.5    # ~1.1bps round-trip @ 0.2% participation
COST_MODEL_BASELINE_BPS_HIGH = 19.6 - 6.9  # ~12.7bps round-trip @ 15% participation

# PSR significance threshold: probability that the live-vs-backtest
# DIFFERENCE series' Sharpe is distinguishable from zero. 0.95 is the
# conventional one-sided 95% confidence cut used elsewhere in this codebase
# for DSR/PSR gating (see ascent/research/deflated_sharpe.py callers).
DIVERGENCE_PSR_THRESHOLD = 0.95


LEDGER_PATH = Path("ascent_daily_ledger.csv")
EOD_LOG_PATH = Path("logs/eod_log.jsonl")
OUTPUT_PATH = Path("dashboard/live_vs_backtest.json")

# ── Live-returns incremental cache ──────────────────────────────────────────
# `get_portfolio_history()` was being called with period="1A" on EVERY daily
# run -- 251 of 252 fetched days were already known, for 1 newly-settled day.
# This cache persists previously-fetched days locally (CSV, keyed on date) so
# a call only needs to re-fetch a small trailing overlap window (to catch any
# late-settling bars), not a full year, every time.
LIVE_RETURNS_CACHE_PATH = Path("data_cache/live_portfolio_returns.csv")
LIVE_RETURNS_FETCH_OVERLAP_DAYS = 8
# If the cache was refreshed more recently than this, skip the Alpaca round
# trip entirely -- repeated calls in quick succession (e.g. within one daily
# run, or interactive re-runs) shouldn't re-hit the network at all.
LIVE_RETURNS_CACHE_TTL_SECONDS = 3600


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_backtest_returns() -> pd.Series:
    """
    Load daily returns from the walk-forward backtest ledger.
    Returns a Series indexed by date.

    The ledger may have a 'daily_return' column directly, or an 'equity'
    column from which returns are computed. Checks both.
    """
    if not LEDGER_PATH.exists():
        print(f"[Monitor] {LEDGER_PATH} not found")
        return pd.Series(dtype=float)

    try:
        df = pd.read_csv(LEDGER_PATH, parse_dates=["date"])
        df = df.set_index("date").sort_index()

        if "daily_return" in df.columns:
            returns = df["daily_return"].dropna()
        elif "net_return" in df.columns:
            returns = df["net_return"].dropna()
        elif "return" in df.columns:
            returns = df["return"].dropna()
        elif "equity" in df.columns:
            returns = df["equity"].pct_change().dropna()
        elif "portfolio_value" in df.columns:
            returns = df["portfolio_value"].pct_change().dropna()
        else:
            print(f"[Monitor] Cannot find return/equity column in ledger. Columns: {df.columns.tolist()}")
            return pd.Series(dtype=float)

        print(f"[Monitor] Loaded {len(returns)} backtest return rows ({returns.index[0].date()} to {returns.index[-1].date()})")
        return returns

    except Exception as e:
        print(f"[Monitor] Failed to load backtest ledger ({e})")
        return pd.Series(dtype=float)


def _load_cached_live_returns() -> pd.Series:
    """Read the locally-persisted settled-returns cache, if any."""
    if not LIVE_RETURNS_CACHE_PATH.exists():
        return pd.Series(dtype=float)
    try:
        df = pd.read_csv(LIVE_RETURNS_CACHE_PATH, parse_dates=["date"])
        s = pd.Series(df["return"].values, index=df["date"]).sort_index()
        return s[~s.index.duplicated(keep="last")]
    except Exception as e:
        print(f"[Monitor] Failed to read live-returns cache ({e}); refetching from scratch")
        return pd.Series(dtype=float)


def _save_live_returns_cache(returns: pd.Series) -> None:
    """Atomically persist the merged returns series (temp file + os.replace)."""
    import tempfile
    LIVE_RETURNS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = returns.sort_index().rename("return").rename_axis("date").reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    tmp = Path(tempfile.mktemp(dir=LIVE_RETURNS_CACHE_PATH.parent, suffix=".tmp"))
    out.to_csv(tmp, index=False)
    os.replace(tmp, LIVE_RETURNS_CACHE_PATH)


def _cache_is_fresh() -> bool:
    try:
        mtime = LIVE_RETURNS_CACHE_PATH.stat().st_mtime
    except OSError:
        return False
    return (datetime.now().timestamp() - mtime) < LIVE_RETURNS_CACHE_TTL_SECONDS


def load_live_portfolio_values() -> pd.Series:
    """
    Load SETTLED daily live returns via Alpaca's portfolio-history endpoint,
    backed by a local incremental cache.

    Historically this hand-rolled `pct_change()` on same-day NAV pulled from
    `logs/eod_log.jsonl`. That is wrong: Alpaca's 1D portfolio-history bars
    don't settle until ~17:00 PT, well after the daily run, so an early run
    would read `equity == last_equity` and silently record a fake 0.0 return
    (see `ascent/execution/alpaca_broker.py:get_portfolio_history()`'s
    docstring). `get_portfolio_history()` is the authoritative source — it
    already returns `{date_iso: settled_day_return}`, computed in market time.

    Caching: `get_portfolio_history()` used to be called with period="1A" on
    every invocation -- 251 of 252 fetched days were already known, for 1
    newly-settled day. Now, previously-fetched days are persisted to
    `LIVE_RETURNS_CACHE_PATH` and only a small trailing overlap window
    (`LIVE_RETURNS_FETCH_OVERLAP_DAYS`) is re-fetched and merged in (the
    overlap catches any late-settling bars). A cache refreshed within
    `LIVE_RETURNS_CACHE_TTL_SECONDS` is returned as-is, with no Alpaca call
    at all. The very first call (empty cache) still does a full period="1A"
    fetch to backfill history.

    Returns a Series of DAILY RETURNS indexed by date (NOT raw NAV — the name
    is kept for backward compatibility with callers, but the contract this
    module actually relies on is "a return series", which is exactly what
    `compute_live_returns()` used to produce via `pct_change()`). Making this
    function return returns directly and making `compute_live_returns()` a
    passthrough keeps `export_live_vs_backtest()`'s call sequence unchanged.
    """
    from ascent.execution.alpaca_broker import get_portfolio_history

    cached = _load_cached_live_returns()

    if not cached.empty and _cache_is_fresh():
        print(f"[Monitor] Using cached settled live returns ({len(cached)} days, "
              f"refreshed <{LIVE_RETURNS_CACHE_TTL_SECONDS}s ago) — skipping Alpaca fetch")
        return cached

    period = "1A" if cached.empty else f"{LIVE_RETURNS_FETCH_OVERLAP_DAYS}D"
    hist = get_portfolio_history(period=period)

    fetched = pd.Series(
        {pd.Timestamp(d): float(r) for d, r in hist.items()}
    ).sort_index() if hist else pd.Series(dtype=float)

    combined = pd.concat([cached, fetched])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()

    if combined.empty:
        print("[Monitor] No settled live returns from Alpaca portfolio history")
        return combined

    if not fetched.empty:
        try:
            _save_live_returns_cache(combined)
        except Exception as e:
            print(f"[Monitor] Failed to persist live-returns cache ({e})")

    print(f"[Monitor] Loaded {len(combined)} settled live daily returns "
          f"({combined.index[0].date()} to {combined.index[-1].date()})")
    return combined


# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_live_returns(live_series: pd.Series) -> pd.Series:
    """
    Return the live daily-return series as-is.

    `load_live_portfolio_values()` now returns settled daily returns
    directly (via `alpaca_broker.get_portfolio_history()`), not raw NAV, so
    there is nothing left to `pct_change()`. Kept as a distinct function
    (rather than removed) so `export_live_vs_backtest()`'s call sequence —
    and any other caller relying on this name — doesn't need to change.
    """
    return live_series.dropna()


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """63-day rolling annualized Sharpe ratio."""
    roll_mean = returns.rolling(window).mean() * 252
    roll_std  = returns.rolling(window).std() * np.sqrt(252)
    sharpe    = (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan)
    return sharpe


def build_comparison(
    backtest_returns: pd.Series,
    live_returns: pd.Series,
) -> dict:
    """
    Build the full comparison payload for dashboard consumption.

    Returns a dict with:
        backtest_cumulative:      list of {date, value}
        live_cumulative:          list of {date, value}
        backtest_rolling_sharpe:  list of {date, value}
        live_rolling_sharpe:      list of {date, value}
        overlap_start:            first date both series have data
        overlap_end:              last date both series have data
        overlap_days:             int
        summary:                  aggregate stats for the overlap period
    """
    def series_to_records(s: pd.Series):
        return [
            {"date": d.isoformat(), "value": round(float(v), 6)}
            for d, v in s.dropna().items()
        ]

    bt_cum   = (1 + backtest_returns).cumprod()
    live_cum = (1 + live_returns).cumprod()

    bt_sharpe   = rolling_sharpe(backtest_returns)
    live_sharpe = rolling_sharpe(live_returns)

    overlap = backtest_returns.index.intersection(live_returns.index)

    payload = {
        "generated_at":            datetime.now().isoformat(),
        "backtest_cumulative":      series_to_records(bt_cum),
        "live_cumulative":          series_to_records(live_cum),
        "backtest_rolling_sharpe":  series_to_records(bt_sharpe),
        "live_rolling_sharpe":      series_to_records(live_sharpe),
        "overlap_start":            overlap.min().isoformat() if len(overlap) > 0 else None,
        "overlap_end":              overlap.max().isoformat() if len(overlap) > 0 else None,
        "overlap_days":             len(overlap),
    }

    # Summary stats computed only on the overlapping period
    if len(overlap) >= 5:
        bt_ov   = backtest_returns.loc[overlap]
        live_ov = live_returns.loc[overlap]

        bt_cum_ov   = bt_cum.reindex(overlap)
        live_cum_ov = live_cum.reindex(overlap)

        def ann_sharpe(r):
            if r.std() == 0:
                return 0.0
            return float(r.mean() / r.std() * np.sqrt(252))

        payload["summary"] = {
            "backtest_total_return":  round(float(bt_cum_ov.iloc[-1] / bt_cum_ov.iloc[0] - 1), 4),
            "live_total_return":      round(float(live_cum_ov.iloc[-1] / live_cum_ov.iloc[0] - 1), 4),
            "backtest_ann_sharpe":    round(ann_sharpe(bt_ov), 3),
            "live_ann_sharpe":        round(ann_sharpe(live_ov), 3),
            "tracking_error_ann":     round(float((bt_ov - live_ov).std() * np.sqrt(252)), 4),
            "overlap_days":           len(overlap),
        }
        payload["significance"] = _divergence_significance(live_ov - bt_ov)

    return payload


def _divergence_significance(diff: pd.Series) -> dict:
    """
    Is the live-vs-backtest DIFFERENCE series' Sharpe statistically
    distinguishable from zero? Uses `probabilistic_sharpe_ratio()`
    (Bailey & Lopez de Prado / Mertens PSR) with a zero benchmark: PSR here
    is P(true Sharpe of the difference series > 0).

    A high PSR means the divergence looks like a real, persistent gap
    rather than noise — worth investigating. It does NOT by itself mean the
    gap is a bug: see COST_MODEL_BASELINE_BPS_LOW/HIGH above for the
    expected backtest-vs-live cost-model gap that alone can produce a
    "real" (high-PSR) divergence with a mundane, already-documented cause.

    `probabilistic_sharpe_ratio()` can return three DISTINCT things and
    they are handled distinctly here, not coerced together:
      - a float in [0, 1]: a real PSR estimate.
      - 0.5: the function's own "genuinely uninformative" sentinel
        (n_obs <= 1) — reported as such, not treated as "50% chance real".
      - None: the PSR formula itself degenerated (denominator <= 0 or NaN)
        for these skew/kurtosis/Sharpe inputs — reported as "indeterminate",
        never coerced to a number.
    """
    diff = diff.dropna()
    n_obs = len(diff)

    result = {
        "n_obs": int(n_obs),
        "diff_ann_sharpe": None,
        "psr_vs_zero": None,
        "psr_status": None,   # "computed" | "uninformative_n" | "formula_degenerate"
        "diverged_significantly": False,
        "cost_model_baseline_bps": {
            "low_participation": round(COST_MODEL_BASELINE_BPS_LOW, 2),
            "high_participation": round(COST_MODEL_BASELINE_BPS_HIGH, 2),
            "note": ("Backtest-vs-live cost-model gap alone (see "
                     "ascent/backtest/costs.py docstring) can produce a "
                     "divergence in this rough range without any real "
                     "execution/alpha problem."),
        },
        "threshold": DIVERGENCE_PSR_THRESHOLD,
    }

    if n_obs < 5 or diff.std(ddof=1) == 0 or math.isnan(diff.std(ddof=1)):
        result["psr_status"] = "uninformative_n"
        return result

    sharpe_obs = float(diff.mean() / diff.std(ddof=1) * np.sqrt(252))
    skew = float(diff.skew())
    # pandas .kurtosis() is EXCESS kurtosis (0 = normal); the PSR formula's
    # (kurtosis - 1)/4 term expects RAW kurtosis (3 = normal) per
    # deflated_sharpe.py's docstring, so convert here.
    kurtosis_raw = float(diff.kurtosis()) + 3.0

    result["diff_ann_sharpe"] = round(sharpe_obs, 3)

    psr = probabilistic_sharpe_ratio(
        sharpe_observed=sharpe_obs,
        sharpe_benchmark=0.0,
        skew=skew,
        kurtosis=kurtosis_raw,
        n_obs=n_obs,
    )

    if psr is None:
        result["psr_status"] = "formula_degenerate"
        result["psr_vs_zero"] = None
        result["diverged_significantly"] = False
    elif n_obs <= 1:
        # probabilistic_sharpe_ratio's own 0.5 sentinel path (defensive;
        # n_obs < 5 already returns above, so this is unreachable today but
        # kept so a future threshold change can't silently misreport it).
        result["psr_status"] = "uninformative_n"
        result["psr_vs_zero"] = 0.5
        result["diverged_significantly"] = False
    else:
        result["psr_status"] = "computed"
        result["psr_vs_zero"] = round(float(psr), 4)
        result["diverged_significantly"] = psr >= DIVERGENCE_PSR_THRESHOLD

    return result


# ── Main entry point ──────────────────────────────────────────────────────────

def export_live_vs_backtest() -> dict:
    """
    Load both data sources, compute comparison, write JSON to dashboard/.
    Returns the payload dict (also useful for testing / import).
    """
    # Cheap, local check first: no point paying the Alpaca network round-trip
    # in load_live_portfolio_values() if there's no backtest ledger to
    # compare against anyway (ascent_daily_ledger.csv missing is a condition
    # this module's own error path already anticipates).
    bt_returns = load_backtest_returns()
    if bt_returns.empty:
        print("[Monitor] No backtest data — cannot build comparison")
        return {}

    live_nav = load_live_portfolio_values()
    if live_nav.empty:
        print("[Monitor] No live data — cannot build comparison")
        return {}

    live_returns = compute_live_returns(live_nav)
    if live_returns.empty:
        print("[Monitor] No settled live returns available — need at least 1 settled day")
        return {}

    payload = build_comparison(bt_returns, live_returns)

    # Atomic write: temp file in the same directory + os.replace, matching
    # the pattern used elsewhere in this package (weekly_debrief.py,
    # scenario_planner.py, weekend_runner.py) -- a write-time failure leaves
    # the previous (or no) file intact rather than a truncated/corrupt one,
    # which matters here because dashboard/ is auto-published to GitHub Pages
    # after every daily run.
    import tempfile
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    tmp = Path(tempfile.mktemp(dir=OUTPUT_PATH.parent, suffix=".tmp"))
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, OUTPUT_PATH)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    print(
        f"[Monitor] Exported live vs backtest to {OUTPUT_PATH} "
        f"({payload.get('overlap_days', 0)} overlap days)"
    )
    return payload


if __name__ == "__main__":
    export_live_vs_backtest()
