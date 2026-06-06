#!/usr/bin/env python3
"""
Ascent Capital — Main Pipeline Runner
End-to-end: data → features → alpha → portfolio → backtest → report
"""
from __future__ import annotations
import argparse
import json
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
from ascent.alpha.stack import build_alpha_stack
from ascent.portfolio.optimizer import (
    top_n_equal_weight, rank_weighted, sector_constrained_weighted,
    sector_constrained_weighted_mvo, apply_bl_to_latest,
)
from ascent.backtest.engine import BacktestEngine
from ascent.research.evaluation import format_metrics


_LIVE_STALE_DAYS = 3
_SIM_STALE_DAYS  = 0


def _log_sleeve_ic(features: dict, targets: dict) -> None:
    """
    Compute IC for each individual alpha sleeve and log to logs/sleeve_ic_log.jsonl.
    Provides the daily fast-feedback signal needed to detect IC decay early.
    """
    import json
    from datetime import date

    try:
        fwd = targets.get("fwd_ret_21d")
        if fwd is None or fwd.empty:
            return

        sleeve_builders: dict = {}
        try:
            from ascent.alpha.trend import trend_alpha
            sleeve_builders["trend"] = lambda f: trend_alpha(f)
        except Exception:
            pass
        try:
            from ascent.alpha.meanrev import meanrev_alpha
            sleeve_builders["meanrev"] = lambda f: meanrev_alpha(f)
        except Exception:
            pass
        try:
            from ascent.alpha.statarb import statarb_alpha
            sleeve_builders["statarb"] = lambda f: statarb_alpha(f, sector_map={})
        except Exception:
            pass
        try:
            from ascent.alpha.fundamental import fundamental_alpha
            sleeve_builders["fundamental"] = lambda f: fundamental_alpha(f)
        except Exception:
            pass
        try:
            from ascent.alpha.earnings import earnings_alpha
            sleeve_builders["earnings"] = lambda f: earnings_alpha(f)
        except Exception:
            pass
        try:
            from ascent.alpha.analyst import analyst_alpha
            sleeve_builders["analyst"] = lambda f: analyst_alpha(f)
        except Exception:
            pass
        try:
            from ascent.alpha.options_flow import options_flow_alpha
            sleeve_builders["options_flow"] = lambda f: options_flow_alpha(f)
        except Exception:
            pass
        try:
            from ascent.alpha.insider import insider_alpha
            sleeve_builders["insider"] = lambda f: insider_alpha(f)
        except Exception:
            pass
        try:
            from ascent.alpha.short_interest import short_interest_alpha
            sleeve_builders["short_interest"] = lambda f: short_interest_alpha(f)
        except Exception:
            pass

        sleeve_ics: dict = {}
        for name, builder in sleeve_builders.items():
            try:
                alpha_df = builder(features)
                if alpha_df is None or alpha_df.empty:
                    continue
                common = alpha_df.index.intersection(fwd.index)
                if len(common) < 50:
                    continue
                ic_series = alpha_df.loc[common].corrwith(fwd.loc[common], axis=1).dropna()
                if len(ic_series) < 10:
                    continue
                mean_ic = float(ic_series.mean())
                std_ic  = float(ic_series.std())
                t_stat  = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic > 0 else 0.0
                sleeve_ics[name] = {
                    "mean_ic": round(mean_ic, 5),
                    "ic_t":    round(t_stat, 3),
                    "n":       len(ic_series),
                }
                print(f"[Alpha] {name:12s}  IC={mean_ic:+.4f}  t={t_stat:+.2f}  n={len(ic_series)}")
            except Exception:
                pass

        if not sleeve_ics:
            return

        log_path = Path("logs/sleeve_ic_log.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"date": date.today().isoformat(), "sleeves": sleeve_ics}) + "\n")
        print(f"[Alpha] Per-sleeve IC logged → {log_path}")

    except Exception as exc:
        print(f"[Alpha] Per-sleeve IC logging failed: {exc}")


def load_or_fetch_prices(cfg: Config, live: bool) -> tuple[pd.DataFrame, str]:
    """Returns (price_df, cache_name_used)."""
    cache_name       = "prices_live" if live else "prices_simulated"
    stale_days       = _LIVE_STALE_DAYS if live else _SIM_STALE_DAYS
    required_symbols = cfg.universe.symbols + [cfg.universe.benchmark]

    # Hub bypass: if the centralized hub ran recently, trust its data and skip
    # validate_cache. validate_cache would fail because required_end=today but
    # today's close isn't available until after market close. The hub guarantees
    # data is as fresh as it can be — re-fetching wastes time and hits rate limits.
    if live:
        try:
            from ascent.data.hub import hub_is_fresh
            if hub_is_fresh():
                df = load_parquet(cache_name)
                if not df.empty:
                    print("[Data] Hub data is fresh — loading prices_live.parquet (skipping re-fetch)")
                    return df, cache_name
        except Exception:
            pass  # fall through to validate_cache path

    ok, reason = validate_cache(
        cache_name,
        required_start=cfg.backtest.start_date,
        required_end=cfg.backtest.end_date if live else None,
        required_symbols=required_symbols,
        stale_days=stale_days,
    )

    if ok:
        print("[Data] Loading cached price data...")
        return load_parquet(cache_name), cache_name

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

        if not df.empty:
            fetched_symbols = set(df["symbol"].unique())
            required_set    = set(required_symbols)
            missing_symbols = required_set - fetched_symbols
            if missing_symbols:
                print(
                    f"[Data] WARNING: live fetch returned only {len(fetched_symbols)} of "
                    f"{len(required_set)} required symbols. "
                    f"Missing: {sorted(missing_symbols)} — likely delisted, using available symbols."
                )
                # do not wipe df — use whatever Yahoo returned

        if df.empty:
            print("[Data] No complete live data — falling back to simulated")
            df = generate_price_data(
                required_symbols,
                cfg.backtest.start_date,
                cfg.backtest.end_date,
            )
            df = normalize_prices(df)
            # Bug 11 fix: write to a distinct cache name so future runs never
            # confuse this simulated fallback with real fetched market data.
            # "prices_live" is reserved for genuine Yahoo/Polygon data only.
            fallback_cache = "prices_live_fallback_simulated"
            save_parquet(df, fallback_cache)
            print(
                f"[Data] WARNING: live fetch failed — saved SIMULATED fallback to "
                f"'{fallback_cache}' (NOT prices_live). "
                "Walk-forward will use simulated data."
            )
            return df, fallback_cache

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
    return df, cache_name


def load_or_fetch_macro(cfg: Config, live: bool) -> tuple[pd.DataFrame, str]:
    """Returns (macro_df, cache_name_used)."""
    cache_name = "macro_live" if live else "macro_simulated"
    stale_days = _LIVE_STALE_DAYS if live else _SIM_STALE_DAYS

    # FIX #3: use series_id consistently — simulated.py and fred.py both write
    # series_id, not series. validate_cache was checking symbol_col="series"
    # which never matched, so the cache was always treated as missing.
    ok, reason = validate_cache(
        cache_name,
        required_start=cfg.backtest.start_date,
        required_end=cfg.backtest.end_date if live else None,
        stale_days=stale_days,
        symbol_col="series_id",   # FIX #3: was "series"
    )

    if ok:
        print("[Data] Loading cached macro data...")
        return load_parquet(cache_name), cache_name

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
    return df, cache_name


def run_pipeline(
    live: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance_days: int | None = None,
) -> tuple:
    cfg = get_config()
    if not cfg.backtest.end_date:
        from datetime import date
        _today = pd.Timestamp.today().normalize()
        _days_back = max(0, _today.weekday() - 4)
        cfg.backtest.end_date = (_today - pd.Timedelta(days=_days_back)).strftime("%Y-%m-%d")
        print(f"[Config] End date set to today: {cfg.backtest.end_date}")

    if start_date:     cfg.backtest.start_date         = start_date
    if end_date:       cfg.backtest.end_date            = end_date
    if top_n:          cfg.backtest.top_n               = top_n
    if rebalance_days: cfg.backtest.rebalance_freq_days = rebalance_days

    t0 = time.time()

    print("\n" + "=" * 70)
    print("  STEP 1: DATA INGESTION")
    print("=" * 70)

    # FIX #2: capture the cache name actually used so we can pass it to
    # walk_forward_pipeline — avoiding hardcoded "prices_live" lookups later
    price_df, price_cache_name = load_or_fetch_prices(cfg, live)
    macro_df, macro_cache_name = load_or_fetch_macro(cfg, live)

    # Strip timezone from price dates — yfinance returns tz-aware (America/New_York),
    # but fundamental/earnings panels strip tz, causing DatetimeIndex → plain Index
    # degradation in build_alpha_stack when unioning tz-aware and tz-naive indices.
    if price_df["date"].dtype.tz is not None:
        price_df = price_df.copy()
        price_df["date"] = price_df["date"].dt.tz_localize(None)

    benchmark_sym  = cfg.universe.benchmark
    benchmark_mask = price_df["symbol"] == benchmark_sym
    benchmark_df   = price_df[benchmark_mask].copy()
    universe_df    = price_df[~benchmark_mask].copy()

    symbols = sorted(universe_df["symbol"].unique())
    print(f"[Data] Universe: {len(symbols)} symbols")
    print(f"[Data] Date range: {universe_df['date'].min().date()} → {universe_df['date'].max().date()}")

    print("\n" + "=" * 70)
    print("  STEP 1.5: REGIME ENGINE")
    print("=" * 70)

    regime_engine = None
    regime_signal = None
    spy_wide      = None
    univ_wide     = None
    vix_series    = None

    # Fast-path: use the regime JSON written by run_all_agents._refresh_regime()
    # rather than re-running an expensive HMM fit on every agent call.
    # The JSON is refreshed at daily startup if stale (>5 days old).
    _REGIME_JSON = Path("dashboard/regime_signal.json")
    _REGIME_STALE_DAYS = 5
    _regime_from_json = False
    try:
        if _REGIME_JSON.exists():
            import json as _json
            from datetime import date as _date
            _rsig = _json.loads(_REGIME_JSON.read_text())
            if isinstance(_rsig, dict):
                _label_str = _rsig.get("label") or _rsig.get("regime") or "uncertain"
                _date_str  = _rsig.get("last_refit_date") or _rsig.get("as_of") or ""
                _days_old  = (_date.today() - _date.fromisoformat(_date_str[:10])).days if _date_str else 99
                if _days_old <= _REGIME_STALE_DAYS:
                    from ascent.regime.types import RegimeLabel, RegimeSignal, REGIME_CONFIG_DEFAULTS
                    try:
                        _label = RegimeLabel(_label_str)
                    except Exception:
                        _label = RegimeLabel.UNCERTAIN
                    _risk_mults = REGIME_CONFIG_DEFAULTS.get("regime_risk_multiplier", {})
                    _risk_mult  = _risk_mults.get(_label.value, 1.0)
                    _sleeve_adj = REGIME_CONFIG_DEFAULTS.get("regime_sleeve_adjustments", {}).get(_label.value, {})
                    regime_signal = RegimeSignal(
                        date=pd.Timestamp.today(),
                        probs=np.ones(3) / 3,
                        label=_label,
                        entropy=float(_rsig.get("entropy", 0.0)),
                        transition_flag=False,
                        risk_multiplier=_risk_mult,
                        sleeve_adjustments=_sleeve_adj,
                    )

                    class _JsonRegimeEngine:
                        best_k = 3
                        def get_signal(self, _dt=None):
                            return regime_signal
                        def get_signal_series(self):
                            return pd.DataFrame()
                        def save_for_intel(self, dashboard_dir="dashboard"):
                            pass  # already saved by _refresh_regime()

                    regime_engine    = _JsonRegimeEngine()
                    _regime_from_json = True
                    print(f"[Regime] Loaded from JSON (regime={_label.value}, {_days_old}d old) — skipping HMM fit")
    except Exception as _rje:
        print(f"[Regime] JSON load failed ({_rje}) — falling back to full fit")

    if not _regime_from_json:
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
            market_prices_df = None
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

                try:
                    import yfinance as _yf
                    _mkt_raw = _yf.download(
                        ["HYG", "LQD", "TLT", "IEF"],
                        start=str(cfg.backtest.start_date),
                        end=str(cfg.backtest.end_date),
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                    if isinstance(_mkt_raw.columns, pd.MultiIndex):
                        _close_key = [c for c in _mkt_raw.columns.get_level_values(0) if str(c).lower() == "close"]
                        if _close_key:
                            _mkt_raw = _mkt_raw[_close_key[0]]
                    market_prices_df = _mkt_raw.reindex(columns=["HYG", "LQD", "TLT", "IEF"])
                    market_prices_df.index = pd.to_datetime(market_prices_df.index).tz_localize(None)
                    print(f"[Regime] Credit/yield instruments fetched: {market_prices_df.shape}")
                except Exception as _e:
                    print(f"[Regime] Credit/yield fetch skipped: {_e}")

            if hasattr(spy_wide.index, "tz") and spy_wide.index.tz is not None:
                spy_wide.index = spy_wide.index.tz_localize(None)
            if hasattr(univ_wide.index, "tz") and univ_wide.index.tz is not None:
                univ_wide.index = univ_wide.index.tz_localize(None)

            macro_wide = None
            if macro_df is not None and not macro_df.empty:
                try:
                    pivot_col = "series_id" if "series_id" in macro_df.columns else "series"
                    macro_wide = macro_df.pivot_table(
                        index="date", columns=pivot_col, values="value"
                    ).sort_index()
                    if hasattr(macro_wide.index, "tz") and macro_wide.index.tz is not None:
                        macro_wide.index = macro_wide.index.tz_localize(None)
                    print(f"[Regime] Macro panel: {macro_wide.shape[1]} series, "
                          f"{macro_wide.shape[0]} dates")
                except Exception as e:
                    print(f"[Regime] Macro pivot failed ({e}) — fitting without macro")
                    macro_wide = None

            _univ_for_regime = univ_wide
            if _univ_for_regime is not None and _univ_for_regime.shape[1] > 200:
                _completeness = _univ_for_regime.notna().mean()
                _best_cols = _completeness.nlargest(200).index
                _univ_for_regime = _univ_for_regime[_best_cols]

            regime_engine = RegimeEngine(config=cfg.regime.to_engine_dict())
            regime_engine.fit(
                spy_prices=spy_wide,
                universe_prices=_univ_for_regime,
                vix_prices=vix_series,
                macro_df=macro_wide,
                market_prices=market_prices_df,
                run_model_selection=False,
            )

            last_date     = spy_wide.index[-1]
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

    # Blend AI regime assessment into regime engine if pre-thesis wrote one today
    _ai_sleeve_prior: dict = {}
    _ai_assess_path = Path("data_cache/ai_regime_assessment.json")
    if _ai_assess_path.exists():
        try:
            _ai_assess = json.loads(_ai_assess_path.read_text())
            _assess_date = _ai_assess.get("as_of_date", "")
            if _assess_date and regime_engine is not None:
                regime_engine.blend_with_ai(_ai_assess, as_of_date=_assess_date)
                print(f"[Regime] AI blend applied: {_ai_assess.get('label', 'n/a')} "
                      f"conf={_ai_assess.get('confidence', 0):.2f}")
            _ai_sleeve_prior = dict(_ai_assess.get("sleeve_weight_prior") or {})
        except Exception as _blend_e:
            print(f"[Regime] AI blend skipped: {_blend_e}")

    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    from ascent.data.store.parquet import has_data as _hd, load_parquet as _lp
    fundamentals_df = None
    if _hd("fundamentals"):
        try:
            fundamentals_df = _lp("fundamentals")
            print(f"[Alpha] Fundamentals loaded: {len(fundamentals_df)} rows")
        except Exception as _fe:
            print(f"[Alpha] Fundamentals load failed: {_fe}")

    earnings_df = None
    if _hd("earnings"):
        try:
            earnings_df = _lp("earnings")
            print(f"[Alpha] Earnings loaded: {len(earnings_df)} rows")
        except Exception as _ee:
            print(f"[Alpha] Earnings load failed: {_ee}")

    analyst_df = None
    if _hd("analyst_revisions"):
        try:
            analyst_df = _lp("analyst_revisions")
            print(f"[Alpha] Analyst revisions loaded: {len(analyst_df)} rows")
        except Exception as _ae:
            print(f"[Alpha] Analyst load failed: {_ae}")

    options_df = None
    if _hd("options_flow"):
        try:
            options_df = _lp("options_flow")
            print(f"[Alpha] Options flow loaded: {len(options_df)} rows")
        except Exception as _oe:
            print(f"[Alpha] Options load failed: {_oe}")

    insider_df = None
    if _hd("insider_transactions"):
        try:
            insider_df = _lp("insider_transactions")
            print(f"[Alpha] Insider transactions loaded: {len(insider_df)} rows")
        except Exception as _ie:
            print(f"[Alpha] Insider load failed: {_ie}")

    short_df = None
    if _hd("short_interest"):
        try:
            short_df = _lp("short_interest")
            print(f"[Alpha] Short interest loaded: {len(short_df)} rows")
        except Exception as _se:
            print(f"[Alpha] Short interest load failed: {_se}")

    builder  = FeatureBuilder(universe_df, macro_df, fundamentals_df=fundamentals_df,
                               earnings_df=earnings_df, analyst_df=analyst_df,
                               options_df=options_df, insider_df=insider_df,
                               short_df=short_df)
    features = builder.compute_features()
    targets  = builder.compute_targets(horizons=[1, 5, 21])

    print(f"[Features] Computed {len(features)} features across {len(builder.symbols)} symbols")
    print(f"[Features] Date range: {builder.dates.min().date()} → {builder.dates.max().date()}")

    print("\n" + "=" * 70)
    print("  STEP 3: ALPHA GENERATION")
    print("=" * 70)

    # Inject 21-day forward returns into features dict so ML sleeve can train
    if "fwd_ret_21d" in targets:
        features["targets"] = targets["fwd_ret_21d"]

    alpha, _alpha_breakdown = build_alpha_stack(
        features,
        regime_signal=regime_signal,
        ai_prior=_ai_sleeve_prior or None,
        return_breakdown=True,
    )
    print(f"[Alpha] Composite alpha computed: {alpha.shape}")

    if "fwd_ret_21d" in targets:
        fwd    = targets["fwd_ret_21d"]
        common = alpha.index.intersection(fwd.index)
        if len(common) > 100:
            ic_series = alpha.loc[common].corrwith(fwd.loc[common], axis=1)
            mean_ic   = ic_series.mean()
            print(f"[Alpha] Mean IC (21d fwd return): {mean_ic:.4f}")
            print(f"[Alpha] IC t-stat: {mean_ic / (ic_series.std() / np.sqrt(len(ic_series))):.2f}")
        _log_sleeve_ic(features, targets)

    print("\n" + "=" * 70)
    print("  STEP 4: PORTFOLIO CONSTRUCTION")
    print("=" * 70)

    from ascent.data.store.parquet import has_data as _hd, load_parquet as _lp
    sector_map = {}
    if _hd("profiles"):
        _prof      = _lp("profiles")
        sector_map = dict(zip(_prof["symbol"], _prof["sector"]))
        print("[Portfolio] Sector constraint: max 1 per sector")

    # ── AI PM two-way: apply pre-thesis alpha floor + names-to-avoid ────────────
    # These come from Phase 1 (run before quant), stored in the prethesis object
    # passed via run_all_agents.py. Injected here so portfolio construction sees them.
    _ai_conviction_syms: list = []
    _ai_avoid_syms: set = set()
    try:
        _pt_path = Path("data_cache/ai_prethesis_latest.json")
        if _pt_path.exists():
            import json as _json
            _pt = _json.loads(_pt_path.read_text())
            _ai_conviction_syms = [n["symbol"] for n in _pt.get("high_conviction_names", []) if "symbol" in n]
            _ai_avoid_syms      = {n["symbol"] for n in _pt.get("names_to_avoid", []) if "symbol" in n}
    except Exception:
        pass

    if not alpha.empty:
        # Floor: conviction stocks below the 20th percentile get bumped up to it.
        # Ensures AI PM thesis names can't be silently excluded by sector cap.
        if _ai_conviction_syms:
            last_row = alpha.iloc[-1]
            floor_val = float(last_row.quantile(0.20))
            for sym in _ai_conviction_syms:
                if sym in last_row.index and last_row[sym] < floor_val:
                    alpha.iloc[-1, alpha.columns.get_loc(sym)] = floor_val
            n_floored = sum(1 for s in _ai_conviction_syms if s in last_row.index and last_row[s] < floor_val)
            if n_floored:
                print(f"[Portfolio] AI PM alpha floor: bumped {n_floored} conviction stocks to 20th pct")

        # Avoid: zero out alpha for names AI PM flagged — they can't enter the portfolio
        if _ai_avoid_syms:
            avoid_present = [s for s in _ai_avoid_syms if s in alpha.columns]
            if avoid_present:
                alpha[avoid_present] = 0.0
                print(f"[Portfolio] AI PM avoid signal: zeroed alpha for {avoid_present}")

    # ── MVO on the latest rebalance date; rank-weight for historical dates ──────
    opt_method = "rank_weight_fallback"
    last_alpha_row = alpha.iloc[-1].dropna() if not alpha.empty else pd.Series(dtype=float)

    # Build factor covariance for the last date (Plan 1 integration)
    mvo_covariance = None
    try:
        from ascent.risk.covariance_model import build_factor_covariance_matrix
        last_date_str = str(alpha.index[-1].date()) if not alpha.empty else None
        if last_date_str:
            weights_for_cov = pd.Series(
                np.ones(len(last_alpha_row)) / max(len(last_alpha_row), 1),
                index=last_alpha_row.index,
            )
            cov_result = build_factor_covariance_matrix(weights_for_cov, last_date_str)
            if cov_result and cov_result.get("full") is not None:
                mvo_covariance = cov_result["full"]
    except Exception as _cov_e:
        print(f"[Portfolio] Factor covariance unavailable: {_cov_e}")

    # Get LLM alpha for BL blending (sparse — zero-fill missing symbols)
    llm_alpha_latest = None
    try:
        from ascent.alpha.llm_fundamental import llm_fundamental_alpha
        _llm_df = llm_fundamental_alpha(features)
        if not _llm_df.empty:
            llm_alpha_latest = _llm_df.iloc[-1].dropna()
    except Exception:
        pass

    # MVO on latest date
    mvo_weights_latest, opt_method = sector_constrained_weighted_mvo(
        alpha_scores=last_alpha_row,
        regime_label=regime_signal.label.value if regime_signal else "calm_bull",
        covariance=mvo_covariance,
        factor_constraints=None,   # Plan 2 wires in; constraints per-symbol not per-date
        current_weights=None,
        sector_map=sector_map,
        n=cfg.backtest.top_n,
        max_weight=cfg.backtest.max_weight,
        llm_alpha=llm_alpha_latest,
    )
    print(f"[Portfolio] Optimization method: {opt_method}")

    # Historical dates: rank-weighting (MVO is single-period, not a panel optimizer)
    target_weights = sector_constrained_weighted(
        alpha,
        n=cfg.backtest.top_n,
        max_weight=cfg.backtest.max_weight,
        max_per_sector=1,
        sector_map=sector_map,
        regime_signal=None,
    )

    # Overwrite the last date with MVO weights if successful
    if not mvo_weights_latest.empty and not target_weights.empty:
        last_dt = target_weights.index[-1]
        target_weights.loc[last_dt] = 0.0
        for sym, w in mvo_weights_latest.items():
            if sym in target_weights.columns:
                target_weights.loc[last_dt, sym] = float(w)

    warmup = 252 + 21
    if len(target_weights) > warmup:
        target_weights = target_weights.iloc[warmup:]

    try:
        spy_close         = benchmark_df.set_index("date")["close"].sort_index()
        spy_close         = spy_close[~spy_close.index.duplicated(keep="last")]
        spy_ma200         = spy_close.rolling(200, min_periods=150).mean()
        spy_below_ma      = spy_close < spy_ma200

        # VIX confirmation: only cut exposure when both SPY < 200MA AND VIX > 20.
        # SPY-alone fires during relief rallies (e.g. April 2026) where markets
        # recover faster than the MA catches up. VIX < 20 = options market disagrees.
        from ascent.regime.engine import VIX_STRESSED_CONFIRMATION as _VIX_MA_THRESHOLD
        if vix_series is not None and not (hasattr(vix_series, "empty") and vix_series.empty):
            if hasattr(vix_series, "columns"):
                _vix_close = vix_series["Close"] if "Close" in vix_series.columns else vix_series.iloc[:, 0]
            else:
                _vix_close = vix_series
            _vix_aligned      = _vix_close.reindex(spy_below_ma.index, method="ffill").fillna(25.0)
            spy_below_confirmed = spy_below_ma & _vix_aligned.gt(_VIX_MA_THRESHOLD)
        else:
            spy_below_confirmed = spy_below_ma  # no VIX data — fall back to MA-only

        spy_below_aligned = spy_below_confirmed.reindex(target_weights.index, method="ffill").fillna(False)
        below_dates       = spy_below_aligned[spy_below_aligned].index
        if len(below_dates) > 0:
            target_weights.loc[below_dates] = target_weights.loc[below_dates] * 0.70
            pct = len(below_dates) / max(len(target_weights), 1) * 100
            print(f"[Portfolio] SPY 200d MA filter: {len(below_dates)} dates below MA+VIX>{_VIX_MA_THRESHOLD:.0f} ({pct:.1f}%) → 30% exposure cut")
        else:
            print("[Portfolio] SPY 200d MA filter: no confirmed stress dates — no cuts applied")
    except Exception as _e:
        print(f"[Portfolio] SPY 200d MA filter skipped: {_e}")

    active_dates = (target_weights.sum(axis=1) > 0.01).sum()
    print(f"[Portfolio] Target weights: {target_weights.shape}")
    print(f"[Portfolio] Active rebalance dates: {active_dates}")
    print(f"[Portfolio] Top-{cfg.backtest.top_n} rank-weighted, max weight {cfg.backtest.max_weight:.0%}")

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

    close    = builder.close
    open_    = builder.open
    bm_close = benchmark_df.set_index("date")["close"].sort_index()
    bm_close = bm_close[~bm_close.index.duplicated(keep="last")]

    result = engine.run(
        target_weights=target_weights,
        close_prices=close,
        open_prices=open_,
        benchmark_prices=bm_close,
    )


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

    if regime_engine is not None:
        try:
            import os
            os.makedirs("dashboard", exist_ok=True)
            regime_engine.save_for_intel(dashboard_dir="dashboard")
        except Exception as _e:
            print(f"[Regime] Intel export failed: {_e}")
    else:
        print("[Regime] No regime engine - Intel export skipped")

    return result, regime_engine, spy_wide, univ_wide, vix_series, target_weights, price_df, macro_df, price_cache_name, _alpha_breakdown


def main():
    parser = argparse.ArgumentParser(description="Ascent Capital — Quant Trading Platform")
    parser.add_argument("--live",          action="store_true")
    parser.add_argument("--start",         type=str, default=None)
    parser.add_argument("--end",           type=str, default=None)
    parser.add_argument("--top-n",         type=int, default=None)
    parser.add_argument("--rebalance",     type=int, default=None)
    parser.add_argument("--clear-cache",   action="store_true")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Bypass all parquet caches and re-fetch/re-generate data")

    args = parser.parse_args()

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

    run_pipeline(
        live=args.live,
        start_date=args.start,
        end_date=args.end,
        top_n=args.top_n,
        rebalance_days=args.rebalance,
    )


if __name__ == "__main__":
    main()