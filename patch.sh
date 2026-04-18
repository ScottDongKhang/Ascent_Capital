#!/usr/bin/env bash
set -euo pipefail

# ── 0. Timestamps & backups ────────────────────────────────────────────────
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/${TS}"
mkdir -p "${BACKUP_DIR}"

cp ascent/data/store/parquet.py  "${BACKUP_DIR}/parquet.py.bak"
cp ascent/main.py                 "${BACKUP_DIR}/main.py.bak"
echo "[backup] saved to ${BACKUP_DIR}/"

# ── 1. Patch ascent/data/store/parquet.py ─────────────────────────────────
# Add validate_cache() and a FORCE_REFRESH env-var check on top of the
# existing save/load helpers.  Architecture preserved: only additive.

cat <<'PARQUET_EOF' > ascent/data/store/parquet.py
"""
Ascent Capital — Parquet Store
Handles saving and loading DataFrames to/from the data_cache directory.

Additions (cache-validation patch):
  validate_cache(name, required_start, required_end, required_symbols,
                 stale_days) -> (bool, str)
      Returns (True, "") if the cache is acceptable, else (False, reason).

  ASCENT_FORCE_REFRESH env-var:
      Set to "1" or "true" to bypass the cache entirely for the current run.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd

log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[3]          # repo root
DATA_DIR = _ROOT / "data_cache"


def _cache_path(name: str) -> Path:
    return DATA_DIR / f"{name}.parquet"


def _force_refresh() -> bool:
    """Return True when the caller has set ASCENT_FORCE_REFRESH=1/true."""
    val = os.environ.get("ASCENT_FORCE_REFRESH", "0").strip().lower()
    return val in ("1", "true", "yes")


# ── core helpers ───────────────────────────────────────────────────────────

def has_data(name: str) -> bool:
    """Return True if the parquet cache file exists AND force-refresh is off."""
    if _force_refresh():
        log.info("[cache] ASCENT_FORCE_REFRESH active — treating %s as missing", name)
        return False
    return _cache_path(name).exists()


def save_parquet(df: pd.DataFrame, name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name)
    if path.exists():
        old = pd.read_parquet(path)
        key_cols = [c for c in old.columns if c not in ("known_time", "source")]
        df = pd.concat([old, df], ignore_index=True).drop_duplicates(
            subset=key_cols, keep="last"
        )
    df.to_parquet(path, index=False)
    log.info("[cache] saved %s  rows=%d", name, len(df))


def load_parquet(name: str) -> pd.DataFrame:
    path = _cache_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Cache file not found: {path}")
    return pd.read_parquet(path)


# ── NEW: cache validation ──────────────────────────────────────────────────

def validate_cache(
    name: str,
    required_start: Optional[str] = None,
    required_end: Optional[str] = None,
    required_symbols: Optional[List[str]] = None,
    stale_days: int = 3,
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> tuple[bool, str]:
    """
    Check whether the cached parquet is fit for purpose.

    Args:
        name:             Cache name (no extension).
        required_start:   ISO date — cache must cover from at least this date.
        required_end:     ISO date — cache must cover up to at least this date.
        required_symbols: List of symbols that must be present.
        stale_days:       Reject cache if latest date is older than this many
                          calendar days ago.  Set 0 to skip freshness check.
        date_col:         Column name holding the date.
        symbol_col:       Column name holding the symbol (skipped for macro).

    Returns:
        (True, "")              — cache is acceptable
        (False, reason_string)  — cache should be refreshed
    """
    if _force_refresh():
        return False, "ASCENT_FORCE_REFRESH is active"

    path = _cache_path(name)
    if not path.exists():
        return False, f"cache file missing: {path}"

    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return False, f"cache unreadable: {exc}"

    if date_col not in df.columns:
        # No date column (e.g. profiles) — skip date checks
        return True, ""

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return False, "cache has no valid dates"

    cache_start = dates.min().normalize()
    cache_end   = dates.max().normalize()

    # ── date coverage ──────────────────────────────────────────────────────
    if required_start is not None:
        req_s = pd.Timestamp(required_start).normalize()
        if cache_start > req_s:
            return False, (
                f"cache starts {cache_start.date()} but "
                f"required_start={req_s.date()}"
            )

    if required_end is not None:
        req_e = pd.Timestamp(required_end).normalize()
        if cache_end < req_e:
            return False, (
                f"cache ends {cache_end.date()} but "
                f"required_end={req_e.date()}"
            )

    # ── freshness ─────────────────────────────────────────────────────────
    if stale_days > 0:
        today = pd.Timestamp.today().normalize()
        lag = (today - cache_end).days
        if lag > stale_days:
            return False, (
                f"cache latest date {cache_end.date()} is "
                f"{lag} calendar days old (limit={stale_days})"
            )

    # ── symbol coverage ───────────────────────────────────────────────────
    if required_symbols and symbol_col in df.columns:
        cached_syms = set(df[symbol_col].unique())
        missing = set(required_symbols) - cached_syms
        if missing:
            sample = sorted(missing)[:5]
            return False, (
                f"{len(missing)} required symbols missing from cache "
                f"(sample: {sample})"
            )

    return True, ""
PARQUET_EOF

echo "[patch] ascent/data/store/parquet.py written"

# ── 2. Patch main.py — swap load_or_fetch_prices / load_or_fetch_macro ────
# Replace just those two functions; everything else in main.py is untouched.

cat <<'MAIN_EOF' > ascent/main.py
#!/usr/bin/env python3
"""
Ascent Capital — Main Pipeline Runner
End-to-end: data → features → alpha → portfolio → backtest → report
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from ascent.config.settings import get_config, Config
from ascent.data.ingest.simulated import generate_price_data, generate_macro_data
from ascent.data.normalize.prices import normalize_prices, normalize_macro, pivot_prices
from ascent.data.store.parquet import save_parquet, load_parquet, has_data, validate_cache
from ascent.features.build_features import FeatureBuilder
from ascent.alpha.stack import build_alpha_stack, alpha_to_ranks
from ascent.portfolio.optimizer import top_n_equal_weight, rank_weighted, sector_constrained_weighted
from ascent.backtest.engine import BacktestEngine
from ascent.backtest.reports import print_report
from ascent.research.evaluation import format_metrics


# ── staleness tolerance for live vs simulated caches (calendar days) ───────
_LIVE_STALE_DAYS = 3      # live data: reject if > 3 days old
_SIM_STALE_DAYS  = 0      # simulated: never stale (deterministic)


def load_or_fetch_prices(cfg: Config, live: bool) -> pd.DataFrame:
    cache_name = "prices_live" if live else "prices_simulated"
    stale_days = _LIVE_STALE_DAYS if live else _SIM_STALE_DAYS
    required_symbols = cfg.universe.symbols + [cfg.universe.benchmark]

    ok, reason = validate_cache(
        cache_name,
        required_start=cfg.backtest.start_date,
        required_end=cfg.backtest.end_date if live else None,
        required_symbols=required_symbols,
        stale_days=stale_days,
    )

    if ok:
        print("[Data] Loading cached price data...")
        return load_parquet(cache_name)

    print(f"[Data] Cache invalid ({reason}) — refreshing...")

    if live:
        print("[Data] Fetching live price data...")
        try:
            from ascent.data.ingest.yahoo import fetch_universe_daily
            df = fetch_universe_daily(
                required_symbols,
                cfg.backtest.start_date,
                cfg.backtest.end_date,
            )
        except Exception as e:
            print(f"[Data] Yahoo failed: {e}")
            df = pd.DataFrame()
        if df.empty:
            print("[Data] No live data, falling back to simulated")
            df = generate_price_data(
                required_symbols,
                cfg.backtest.start_date,
                cfg.backtest.end_date,
            )
    else:
        print("[Data] Generating simulated price data...")
        df = generate_price_data(
            required_symbols,
            cfg.backtest.start_date,
            cfg.backtest.end_date,
        )

    df = normalize_prices(df)
    save_parquet(df, cache_name)
    print(f"[Data] {len(df)} rows, {df['symbol'].nunique()} symbols cached")
    return df


def load_or_fetch_macro(cfg: Config, live: bool) -> pd.DataFrame:
    cache_name = "macro_live" if live else "macro_simulated"
    stale_days = _LIVE_STALE_DAYS if live else _SIM_STALE_DAYS

    ok, reason = validate_cache(
        cache_name,
        required_start=cfg.backtest.start_date,
        required_end=cfg.backtest.end_date if live else None,
        stale_days=stale_days,
        symbol_col="series",          # macro uses 'series', not 'symbol'
    )

    if ok:
        print("[Data] Loading cached macro data...")
        return load_parquet(cache_name)

    print(f"[Data] Macro cache invalid ({reason}) — refreshing...")

    if live:
        try:
            from ascent.data.ingest.fred import fetch_all_macro
            df = fetch_all_macro(
                start_date=cfg.backtest.start_date,
                end_date=cfg.backtest.end_date,
            )
        except Exception as e:
            print(f"[Data] FRED failed ({e}), using simulated macro")
            df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)
        if df.empty:
            df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)
    else:
        print("[Data] Generating simulated macro data...")
        df = generate_macro_data(cfg.backtest.start_date, cfg.backtest.end_date)

    df = normalize_macro(df)
    save_parquet(df, cache_name)
    return df


def run_pipeline(
    live: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance_days: int | None = None,
) -> None:
    cfg = get_config()
    if not cfg.backtest.end_date:
        from datetime import date
        cfg.backtest.end_date = pd.bdate_range(end="today", periods=1)[0].strftime("%Y-%m-%d")
        print(f"[Config] End date set to today: {cfg.backtest.end_date}")

    if start_date:
        cfg.backtest.start_date = start_date
    if end_date:
        cfg.backtest.end_date = end_date
    if top_n:
        cfg.backtest.top_n = top_n
    if rebalance_days:
        cfg.backtest.rebalance_freq_days = rebalance_days

    t0 = time.time()

    # ══════════════════════════════════════════════════════════════
    # 1. DATA
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 1: DATA INGESTION")
    print("=" * 70)

    price_df = load_or_fetch_prices(cfg, live)
    macro_df = load_or_fetch_macro(cfg, live)

    benchmark_sym = cfg.universe.benchmark
    benchmark_mask = price_df["symbol"] == benchmark_sym
    benchmark_df = price_df[benchmark_mask].copy()
    universe_df = price_df[~benchmark_mask].copy()

    symbols = sorted(universe_df["symbol"].unique())
    print(f"[Data] Universe: {len(symbols)} symbols")
    print(f"[Data] Date range: {universe_df['date'].min().date()} → {universe_df['date'].max().date()}")

    # ══════════════════════════════════════════════════════════════
    # 1.5 REGIME ENGINE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 1.5: REGIME ENGINE")
    print("=" * 70)

    regime_engine = None
    regime_signal = None

    try:
        from ascent.regime import RegimeEngine, apply_regime_to_portfolio

        spy_wide = (
            benchmark_df.set_index("date")["close"]
            .sort_index()
            .pipe(lambda s: s[~s.index.duplicated(keep="last")])
        )

        univ_wide = (
            universe_df.pivot_table(index="date", columns="symbol", values="close")
            .sort_index()
        )

        vix_series = None
        if live:
            try:
                import yfinance as yf
                vix_raw = yf.download(
                    "^VIX",
                    start=cfg.backtest.start_date,
                    end=cfg.backtest.end_date,
                    progress=False,
                )["Close"].squeeze()
                vix_raw.index = pd.to_datetime(vix_raw.index).tz_localize(None)
                vix_series = vix_raw
                print("[Regime] VIX data fetched")
            except Exception as e:
                print(f"[Regime] VIX fetch skipped: {e}")

        if hasattr(spy_wide.index, 'tz') and spy_wide.index.tz is not None:
            spy_wide.index = spy_wide.index.tz_localize(None)
        if hasattr(univ_wide.index, 'tz') and univ_wide.index.tz is not None:
            univ_wide.index = univ_wide.index.tz_localize(None)
        regime_engine = RegimeEngine(config=cfg.regime.to_engine_dict())
        regime_engine.fit(
            spy_prices=spy_wide,
            universe_prices=univ_wide,
            vix_prices=vix_series,
            macro_df=None,
            run_model_selection=True,
        )

        last_date = spy_wide.index[-1]
        regime_signal = regime_engine.get_signal(last_date)

        if regime_signal:
            print(
                f"[Regime] K={regime_engine.best_k}  "
                f"label={regime_signal.label.value}  "
                f"risk_mult={regime_signal.risk_multiplier:.2f}  "
                f"entropy={regime_signal.entropy:.3f}"
            )
        else:
            print("[Regime] Engine fitted but no signal returned — continuing without regime")

    except Exception as exc:
        print(f"[Regime] Engine failed ({exc}) — pipeline continues without regime adjustment")
        regime_engine = None
        regime_signal = None

    # ══════════════════════════════════════════════════════════════
    # 2. FEATURES
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    builder = FeatureBuilder(universe_df, macro_df)
    features = builder.compute_features()
    targets = builder.compute_targets(horizons=[1, 5, 21])

    print(f"[Features] Computed {len(features)} features across {len(builder.symbols)} symbols")
    print(f"[Features] Date range: {builder.dates.min().date()} → {builder.dates.max().date()}")

    # ══════════════════════════════════════════════════════════════
    # 3. ALPHA  (regime signal adjusts sleeve weights here)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 3: ALPHA GENERATION")
    print("=" * 70)

    alpha = -build_alpha_stack(features, regime_signal=regime_signal)
    ranks = alpha_to_ranks(alpha)
    print(f"[Alpha] Composite alpha computed: {alpha.shape}")

    if "fwd_ret_21d" in targets:
        fwd = targets["fwd_ret_21d"]
        common = alpha.index.intersection(fwd.index)
        if len(common) > 100:
            ic_series = alpha.loc[common].corrwith(fwd.loc[common], axis=1)
            mean_ic = ic_series.mean()
            print(f"[Alpha] Mean IC (21d fwd return): {mean_ic:.4f}")
            print(f"[Alpha] IC t-stat: {mean_ic / (ic_series.std() / np.sqrt(len(ic_series))):.2f}")

    # ══════════════════════════════════════════════════════════════
    # 4. PORTFOLIO CONSTRUCTION  (regime tightens max_weight here)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 4: PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    from ascent.data.store.parquet import has_data as _hd
    sector_map = {}
    if _hd("profiles"):
        from ascent.data.store.parquet import load_parquet as _lp
        _prof = _lp("profiles")
        sector_map = dict(zip(_prof["symbol"], _prof["sector"]))
        print("[Portfolio] Sector constraint: max 1 per sector")

    target_weights = sector_constrained_weighted(
        alpha,
        n=cfg.backtest.top_n,
        max_weight=cfg.backtest.max_weight,
        max_per_sector=1,
        sector_map=sector_map,
        regime_signal=regime_signal,
    )

    warmup = 252 + 21
    if len(target_weights) > warmup:
        target_weights = target_weights.iloc[warmup:]

    if regime_engine is not None:
        try:
            from ascent.regime import apply_regime_to_portfolio
            scaled_rows = []
            for dt in target_weights.index:
                raw_w = target_weights.loc[dt]
                sig = regime_engine.get_signal(pd.Timestamp(dt))
                if sig is not None and raw_w.sum() > 0.01:
                    adj_w, _ = apply_regime_to_portfolio(
                        raw_w,
                        sig,
                        base_max_weight=cfg.backtest.max_weight,
                    )
                    scaled_rows.append(adj_w)
                else:
                    scaled_rows.append(raw_w)
            target_weights = pd.DataFrame(scaled_rows, index=target_weights.index)
            print("[Portfolio] Regime gross exposure scaling applied")
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"[Portfolio] Regime scaling skipped: {exc}")

    active_dates = (target_weights.sum(axis=1) > 0.01).sum()
    print(f"[Portfolio] Target weights: {target_weights.shape}")
    print(f"[Portfolio] Active rebalance dates: {active_dates}")
    print(f"[Portfolio] Top-{cfg.backtest.top_n} rank-weighted, max weight {cfg.backtest.max_weight:.0%}")

    # ══════════════════════════════════════════════════════════════
    # 5. BACKTEST
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  STEP 5: BACKTEST")
    print("=" * 70)

    engine = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        spread_bps=cfg.backtest.spread_bps,
        impact_bps=cfg.backtest.impact_bps,
        rebalance_freq_days=cfg.backtest.rebalance_freq_days,
        execution_delay=cfg.backtest.execution_delay_days,
    )

    close = builder.close
    open_ = builder.open
    bm_close = benchmark_df.set_index("date")["close"].sort_index()
    bm_close = bm_close[~bm_close.index.duplicated(keep="last")]

    result = engine.run(
        target_weights=target_weights,
        close_prices=close,
        open_prices=open_,
        benchmark_prices=bm_close,
    )

    # ══════════════════════════════════════════════════════════════
    # 6. REPORT
    # ══════════════════════════════════════════════════════════════
    print_report(result, title="ASCENT CAPITAL — FULL BACKTEST")

    if regime_engine is not None:
        try:
            from ascent.regime.diagnostics import regime_occupancy_table, regime_return_stats
            sigs = regime_engine.get_signal_series()
            if not sigs.empty:
                print("\n── Regime Occupancy ──────────────────────────────────")
                print(regime_occupancy_table(sigs).to_string())
                spy_ret = bm_close.pct_change().dropna()
                print("\n── Per-Regime Return Stats ───────────────────────────")
                print(regime_return_stats(spy_ret, sigs).to_string())
        except Exception:
            pass

    elapsed = time.time() - t0
    print(f"\n  Pipeline completed in {elapsed:.1f}s\n")

    # Export VIX-based regime signal to JSON for dashboard
    try:
        import json, yfinance as yf
        vix = yf.download("^VIX", start="2020-01-01", progress=False)["Close"].squeeze()
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        regime_json = []
        for dt, v in vix.items():
            if v < 15:
                lbl, rs, rm = "CALM", 0.1, 1.0
            elif v < 20:
                lbl, rs, rm = "ELEVATED", 0.4, 0.9
            elif v < 30:
                lbl, rs, rm = "STRESS", 0.7, 0.75
            else:
                lbl, rs, rm = "CRISIS", 0.95, 0.5
            regime_json.append({"d": dt.strftime("%Y-%m-%d"), "rs": rs, "label": lbl, "risk_mult": rm})
        with open("dashboard/regime_signal.json", "w") as f:
            json.dump(regime_json, f)
        print(f"[Regime] Exported {len(regime_json)} VIX-based regime signals to dashboard/regime_signal.json")
    except Exception as e:
        print(f"[Regime] VIX export failed: {e}")

    return result, regime_engine


def main():
    parser = argparse.ArgumentParser(description="Ascent Capital — Quant Trading Platform")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--rebalance", type=int, default=None)
    parser.add_argument("--mode", type=str, default="walkforward", choices=["walkforward", "full"])
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass all parquet caches and re-fetch/re-generate data",
    )

    args = parser.parse_args()

    # ── NEW: honour --force-refresh via env-var ────────────────────────────
    if args.force_refresh:
        import os
        os.environ["ASCENT_FORCE_REFRESH"] = "1"
        print("[Cache] Force-refresh enabled — all caches will be ignored this run")

    if args.clear_cache:
        import shutil
        cfg = get_config()
        if cfg.data_dir.exists():
            shutil.rmtree(cfg.data_dir)
            print("Cache cleared.")

    _, _regime_engine = run_pipeline(
        live=args.live,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        rebalance_days=args.rebalance,
    )

    if args.mode == "walkforward":
        print("\n" + "#" * 70)
        print("  RUNNING WALK-FORWARD OUT-OF-SAMPLE EVALUATION")
        print("#" * 70)
        from ascent.research.walk_forward_runner import walk_forward_pipeline
        walk_forward_pipeline(regime_engine=_regime_engine)


if __name__ == "__main__":
    main()
MAIN_EOF

echo "[patch] main.py written"

# ── 3. Smoke-test: import the patched parquet module ──────────────────────
python - <<'PYEOF'
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(".").resolve()))
from ascent.data.store.parquet import validate_cache, has_data, _force_refresh
print("imports OK")

# validate_cache on a non-existent cache
ok, reason = validate_cache("__nonexistent__", required_start="2020-01-01")
assert not ok, "expected False for missing cache"
print(f"missing-cache test OK: reason='{reason}'")

# ASCENT_FORCE_REFRESH
import os
os.environ["ASCENT_FORCE_REFRESH"] = "1"
assert _force_refresh(), "expected True"
assert not has_data("prices_live"), "has_data should return False when force-refresh"
del os.environ["ASCENT_FORCE_REFRESH"]
assert not _force_refresh(), "expected False after clearing env-var"
print("force-refresh env-var test OK")

# If a real cache exists, run a coverage check
import pandas as pd
cache = pathlib.Path("data_cache/prices_simulated.parquet")
if cache.exists():
    ok, reason = validate_cache(
        "prices_simulated",
        required_start="2020-01-01",
        stale_days=0,           # skip freshness for simulated
    )
    print(f"prices_simulated coverage check: ok={ok}  reason='{reason}'")

    # Stale-date test: demand data through tomorrow (should fail for any real cache)
    import datetime
    future = (datetime.date.today() + datetime.timedelta(days=2)).isoformat()
    ok2, reason2 = validate_cache(
        "prices_simulated",
        required_end=future,
        stale_days=0,
    )
    print(f"future-date test: ok={ok2}  reason='{reason2}'")

print("all smoke tests passed")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  PATCH COMPLETE — run these verification commands next"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "# 1. Re-run the smoke tests manually:"
echo "python -c \""
echo "from ascent.data.store.parquet import validate_cache"
echo "print(validate_cache('prices_simulated', required_start='2020-01-01', stale_days=0))"
echo "\""
echo ""
echo "# 2. Confirm --force-refresh flag exists in main.py:"
echo "python main.py --help | grep force-refresh"
echo ""
echo "# 3. Test force-refresh path (simulated, no network):"
echo "ASCENT_FORCE_REFRESH=1 python main.py --mode full 2>&1 | head -30"
echo ""
echo "# 4. Normal run — should use (valid) cache if present:"
echo "python main.py --mode full 2>&1 | grep -E '\[Data\]|\[Cache\]'"
echo ""
echo "# 5. Hard clear then full re-fetch:"
echo "python main.py --clear-cache --force-refresh --mode full 2>&1 | head -40"
