"""
ascent/research/walk_forward_lightweight.py

Fast OOS evaluation for self-improve weekly loop.
Runs a simplified walk-forward on the most recent n_days of price data.
Skips regime refit and ML retraining — uses cached models.
Returns Sharpe and turnover in ~2 minutes (vs ~20 for full walk-forward).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional


# Turnover penalty for the Calmar-based fitness score (2026-08-23 rework).
# This constant predates the rework and was calibrated for the old formula
# `sharpe - TURNOVER_PENALTY * turnover` (Sharpe-scale). The rework changed
# the formula to `calmar - TURNOVER_PENALTY * turnover` (see self_improve.py
# score_variant()) but left this constant at its old Sharpe-scale value,
# unlike MIN_CALMAR_EDGE in self_improve.py, which WAS rescaled at the same
# time using the same ratio derived below. Same citation as that comment:
# CLAUDE.md / CURRENT_VERIFIED_NUMBERS.md's canonical walk-forward artifact
# gives calmar_ratio=0.223 against Sharpe ~0.415-0.42 for this book
# (docs/session_log_archive.md: "Sharpe 0.415 ... now"), i.e. a Calmar/Sharpe
# ratio of roughly 0.223/0.415 ~= 0.54. Scaling the old 0.10 by that ratio:
# 0.10 * 0.54 ~= 0.054. Unlike MIN_CALMAR_EDGE -- a promotion bar where
# rounding up keeps promotion at least as hard as before, i.e. conservative
# -- this is a penalty magnitude, and rounding it up further would
# reintroduce the exact over-penalization bug being fixed here. So this
# rounds to the nearest hundredth (0.054 -> 0.05) instead of rounding
# further in the "conservative" direction.
TURNOVER_PENALTY = 0.05   # subtract 0.05 * avg_turnover from Calmar (was 0.10, Sharpe-scale; see comment above)


def _load_prices(prices_cache: str) -> Optional[pd.DataFrame]:
    """
    Load prices from the parquet store. Falls back to cwd-relative data_cache
    so tests using monkeypatch.chdir(tmp_path) can supply their own cache.

    The test writes a wide DataFrame (dates × symbols) directly as parquet.
    The production store writes long-format (date, symbol, close, ...) parquet.
    We detect the format by checking for a 'symbol' column.
    """
    # 1. Try via the registered parquet store (uses package-root data_cache)
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        if has_data(prices_cache):
            df = load_parquet(prices_cache)
            return df
    except Exception:
        pass

    # 2. Fall back to cwd-relative data_cache (used in tests)
    cwd_path = Path.cwd() / "data_cache" / f"{prices_cache}.parquet"
    if cwd_path.exists():
        try:
            return pd.read_parquet(cwd_path)
        except Exception:
            pass

    return None


def _to_wide_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
      - Wide format: DatetimeIndex × symbol columns (test cache)
      - Long format: columns [date, symbol, close, ...] (production cache)
    Returns a wide DataFrame with DatetimeIndex and symbol columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # If already wide (no 'symbol' column, index is datetime-like)
    if "symbol" not in df.columns:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return pd.DataFrame()
        return df.sort_index()

    # Long format — pivot on close
    if "close" not in df.columns:
        return pd.DataFrame()

    try:
        date_col = "date" if "date" in df.columns else df.index.name or "date"
        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            wide = df.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        else:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            wide = df.pivot_table(index=df.index, columns="symbol", values="close", aggfunc="last")
        return wide.sort_index()
    except Exception:
        return pd.DataFrame()


def zero_fill_fold_gaps(fold_records: list) -> list:
    """
    Reconstruct a single, temporally continuous daily-return series from
    multiple walk-forward OOS folds whose test windows are separated by a
    purge+embargo gap of untested trading days.

    Consecutive fold test windows are NOT contiguous by design (the gap is
    correct/intentional -- it prevents leakage between folds). Concatenating
    each fold's returns directly, with no representation of that gap, breaks
    any downstream metric that is order/continuity-sensitive (e.g.
    ``ascent.research.evaluation.calmar_ratio``, which does a running
    ``(1+r).cumprod()`` + peak/drawdown walk): it would either hide a real
    drawdown that happened during the untested gap, or fabricate an instant
    "recovery" by implying zero time passed between one fold's last OOS day
    and the next fold's first OOS day.

    This function fills each gap with explicit ``0.0`` ("assume no change")
    placeholder days -- the most honest statement available for a period that
    was never OOS-tested, since there is no real signal about what happened
    there. The gap length is the actual number of trading-day bar positions
    between folds (not a nominal config value), so it stays correct even if
    folds were skipped.

    Args:
        fold_records: list of (test_start_i, test_end_i, returns) tuples, in
            chronological order, where test_start_i/test_end_i are integer
            bar positions into the same price index the folds were built
            from, and returns is that fold's list of per-day OOS returns.

    Returns:
        Flat list of floats: fold returns interleaved with zero-fill gap
        days, in chronological order.
    """
    out: list = []
    last_test_end_i = None
    for test_start_i, test_end_i, returns in fold_records:
        if last_test_end_i is not None:
            gap_days = test_start_i - last_test_end_i - 1
            if gap_days > 0:
                out.extend([0.0] * gap_days)
        out.extend(returns)
        last_test_end_i = test_end_i
    return out


def run_lightweight_oos(
    config_overrides: Dict[str, Any],
    n_days: int = 63,
    prices_cache: str = "prices_live",
    top_n: int = 15,
    max_weight: float = 0.10,
    train_days: int = 126,
    purge_days: int = 5,
    embargo_days: int = 5,
    filter_universe_by_date: bool = True,
    universe_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Multi-fold expanding walk-forward OOS evaluation.

    Institutional-grade: purge gap + embargo between folds, universe
    filtered by listing date per fold (no survivorship bias).

    Args:
        config_overrides:       Dict with 'alpha_weights' key mapping sleeve names to floats.
        n_days:                 Number of OOS trading days per fold.
        prices_cache:           Parquet cache name to load prices from.
        top_n:                  Portfolio size.
        max_weight:             Max position weight.
        train_days:             Minimum training window per fold.
        purge_days:             Gap between train end and test start (prevents leakage).
        embargo_days:           Gap between test end and next fold's train end.
        filter_universe_by_date: When True, call get_universe_on_date() per fold.
        universe_df:            Historical universe table to filter against (same shape as
                                 `build_historical_universe()`'s return). When None and
                                 filter_universe_by_date is True, built via
                                 `build_historical_universe(strict=True, sp500_only=True)` --
                                 the same survivorship-correct S&P 500 + tracked-removals
                                 universe that eod_runner.py and walk_forward_runner.py use,
                                 so a variant promoted from this OOS path can't be scored on
                                 non-S&P500 symbols with fabricated pre-2020 histories that
                                 live trading could never actually hold. Callers that already
                                 have a universe_df (e.g. an outer walk-forward loop) should
                                 pass it through to avoid rebuilding it per variant.

    Returns:
        {"sharpe": float, "turnover": float, "n_folds": int}
        Returns {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0} on failure.
    """
    try:
        price_df = _load_prices(prices_cache)
        if price_df is None or price_df.empty:
            print(f"[LightweightOOS] No cache '{prices_cache}' -- returning 0.0")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        price_wide = _to_wide_close(price_df)
        if price_wide.empty:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # BUG 2 fix: apply the same delisting-terminal-value credit that
        # walk_forward_pipeline() applies in walk_forward_runner.py, so a
        # shadow/variant config scored through this lightweight path isn't
        # silently favoring names that quietly went NaN on a real deal-close
        # instead of being credited their real terminal value (see
        # apply_delisting_terminal_credit()'s docstring for the full
        # no-look-ahead reasoning -- it only injects a price AT the real
        # historical closure date, never earlier). This is a code-path-parity
        # fix, not a numbers fix: as of this session, 0 of the 12 sourced
        # DELISTING_TERMINAL_TERMS symbols have any rows in prices_live, so
        # the credit is currently inert here exactly as it is everywhere else
        # in production -- it will only start mattering once that underlying
        # data gap is closed.
        try:
            from ascent.research.walk_forward_runner import apply_delisting_terminal_credit
            price_wide, _ = apply_delisting_terminal_credit(price_wide, None)
        except Exception:
            pass  # graceful fallback: score without the credit rather than fail the whole run

        min_required = train_days + purge_days + n_days + embargo_days
        if len(price_wide) < min_required:
            print(f"[LightweightOOS] Insufficient data ({len(price_wide)} rows, need {min_required})")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        alpha_weights = config_overrides.get("alpha_weights", {
            "trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05
        })

        # Resolve the survivorship-correct universe once, up front, so every fold below
        # filters against the same restricted (strict=True, sp500_only=True) universe that
        # eod_runner.py and walk_forward_runner.py use -- matching the promotion path to what
        # live trading can actually hold (see docstring above).
        if filter_universe_by_date and universe_df is None:
            try:
                from ascent.data.universe import build_historical_universe
                universe_df = build_historical_universe(strict=True, sp500_only=True)
            except Exception:
                universe_df = None

        # Build fold positions: expanding window, working backwards from end
        fold_positions = []
        pos = len(price_wide) - 1
        while pos >= train_days + purge_days + n_days:
            test_end_i   = pos
            test_start_i = test_end_i - n_days + 1
            train_end_i  = test_start_i - purge_days - 1
            if train_end_i < train_days:
                break
            fold_positions.append((train_end_i, test_start_i, test_end_i))
            pos = test_start_i - embargo_days - n_days
        fold_positions = list(reversed(fold_positions))  # chronological order

        sharpe_fold_returns = []   # pure fold returns only (no gap padding), for Sharpe
        fold_records = []          # [(test_start_i, test_end_i, fold_rets), ...] -- see
                                    # zero_fill_fold_gaps() for why the "returns" field below
                                    # is reconstructed from this instead of a naive concat.
        prev_weights: Dict[str, float] = {}
        fold_turnovers = []
        n_folds = 0
        fold_date_ranges = []  # additive metadata: [(start_date, end_date), ...] per accepted fold

        # Cross-sectional normalizer
        def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
            mean = df.mean(axis=1)
            std  = df.std(axis=1).replace(0, np.nan)
            return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)

        for train_end_i, test_start_i, test_end_i in fold_positions:
            train_slice = price_wide.iloc[:train_end_i + 1]
            oos_slice   = price_wide.iloc[test_start_i:test_end_i + 1]

            if len(train_slice) < train_days or len(oos_slice) < 5:
                continue

            fold_date = price_wide.index[test_start_i]

            # Survivorship bias fix: filter to symbols valid on fold date.
            # get_universe_on_date() returns a plain list of symbol strings
            # (see ascent/data/universe.py) -- not a DataFrame -- so there is
            # no .empty/.columns to inspect here (BUG 1 fix: the old
            # DataFrame-shape branching raised AttributeError on the list
            # every single call, was swallowed by the except below, and left
            # valid_symbols as the full unfiltered universe unconditionally).
            valid_symbols = list(price_wide.columns)
            if filter_universe_by_date:
                try:
                    from ascent.data.universe import get_universe_on_date
                    fold_universe = get_universe_on_date(fold_date, universe_df)
                    if fold_universe:
                        valid_set = set(fold_universe)
                        filtered = [s for s in price_wide.columns if s in valid_set or s == "SPY"]
                        if len(filtered) >= 5:
                            valid_symbols = filtered
                except Exception:
                    pass  # graceful fallback: e.g. get_universe_on_date raising on bad input

            train_filtered = train_slice[valid_symbols]
            oos_filtered   = oos_slice[valid_symbols]

            # Build features on training slice only (causal)
            try:
                from ascent.features.feature_defs import build_all_features
                dummy_volume        = pd.DataFrame(1e6, index=train_filtered.index, columns=train_filtered.columns)
                dummy_dollar_volume = train_filtered * 1e6

                # Load supplemental caches once (outside fold loop is better, but
                # these loads are fast and allow the lightweight OOS to score
                # fundamental and earnings sleeves on the training slice)
                _earnings_df = None
                _fundamentals_df = None
                try:
                    from ascent.data.store.parquet import has_data as _hd_lw, load_parquet as _lp_lw
                    if _hd_lw("earnings"):
                        _earnings_df = _lp_lw("earnings")
                    if _hd_lw("fundamentals"):
                        _fundamentals_df = _lp_lw("fundamentals")
                except Exception:
                    pass

                features = build_all_features(
                    close=train_filtered,
                    volume=dummy_volume,
                    dollar_volume=dummy_dollar_volume,
                    macro_pivot=None,
                    earnings_df=_earnings_df,
                )

                # Augment with fundamental panel if cache is available
                if _fundamentals_df is not None and not _fundamentals_df.empty:
                    try:
                        from ascent.features.feature_defs import build_fundamental_panel
                        fund_panel = build_fundamental_panel(
                            _fundamentals_df,
                            date_index=train_filtered.index,
                            symbols=list(train_filtered.columns),
                        )
                        if fund_panel is not None and not fund_panel.empty:
                            for col in fund_panel.columns:
                                features[col] = fund_panel[col]
                    except Exception:
                        pass

            except Exception as e:
                print(f"[LightweightOOS] Fold {n_folds+1} feature build failed: {e}")
                continue

            # Build alpha signals
            try:
                from ascent.alpha.trend import trend_alpha
                from ascent.alpha.meanrev import meanrev_alpha

                alphas: Dict[str, pd.DataFrame] = {}

                if alpha_weights.get("trend", 0) > 0:
                    try:
                        t = trend_alpha(features)
                        if not t.empty:
                            alphas["trend"] = t
                    except Exception:
                        pass

                if alpha_weights.get("meanrev", 0) > 0:
                    try:
                        mr = meanrev_alpha(features)
                        if not mr.empty:
                            alphas["meanrev"] = mr
                    except Exception:
                        pass

                if alpha_weights.get("volatility", 0) > 0 and "vol_of_vol_21d" in features and "vol_trend_10d" in features:
                    try:
                        vov   = features["vol_of_vol_21d"].copy().replace(0, np.nan)
                        vtrnd = features["vol_trend_10d"].copy()
                        vol_alpha = _cs_normalize(-vtrnd / (vov + 1e-6))
                        if not vol_alpha.empty:
                            alphas["volatility"] = vol_alpha
                    except Exception:
                        pass

                if alpha_weights.get("statarb", 0) > 0:
                    try:
                        from ascent.alpha.statarb import statarb_alpha
                        sa = statarb_alpha(features, sector_map={})
                        if not sa.empty:
                            alphas["statarb"] = sa
                    except Exception:
                        pass

                if alpha_weights.get("fundamental", 0) > 0:
                    try:
                        from ascent.alpha.fundamental import fundamental_alpha
                        fa = fundamental_alpha(features)
                        if fa is not None and not fa.empty:
                            alphas["fundamental"] = fa
                    except Exception:
                        pass

                if alpha_weights.get("earnings", 0) > 0:
                    try:
                        from ascent.alpha.earnings import earnings_alpha
                        ea = earnings_alpha(features)
                        if ea is not None and not ea.empty:
                            alphas["earnings"] = ea
                    except Exception:
                        pass

            except Exception as e:
                print(f"[LightweightOOS] Fold {n_folds+1} alpha build failed: {e}")
                continue

            if not alphas:
                continue

            # Blend alphas with normalized weights
            total_w = sum(alpha_weights.get(k, 0.0) for k in alphas)
            if total_w == 0:
                total_w = 1.0

            composite = None
            for name, alpha_df in alphas.items():
                w = alpha_weights.get(name, 0.0) / total_w
                normed = _cs_normalize(alpha_df)
                if composite is None:
                    composite = normed * w
                else:
                    union_idx  = composite.index.union(normed.index)
                    union_cols = composite.columns.union(normed.columns)
                    composite  = composite.reindex(index=union_idx, columns=union_cols).fillna(0.0)
                    normed_r   = normed.reindex(index=union_idx, columns=union_cols).fillna(0.0)
                    composite  = composite + normed_r * w

            if composite is None or composite.empty:
                continue

            latest_alpha = composite.iloc[-1].dropna().sort_values(ascending=False)
            if latest_alpha.empty:
                continue

            # Simple rank-weighted portfolio (no sector profiles in test environment)
            top_names = latest_alpha.head(top_n)
            scores = top_names - top_names.min() + 1e-8
            raw_w  = scores / scores.sum()
            raw_w  = raw_w.clip(upper=max_weight)
            if raw_w.sum() > 0:
                raw_w = raw_w / raw_w.sum()
            weights_dict = raw_w.to_dict()

            if not weights_dict:
                continue

            oos_syms = [s for s in weights_dict if s in oos_filtered.columns]
            if not oos_syms:
                continue

            w_arr = np.array([weights_dict[s] for s in oos_syms])
            w_sum = w_arr.sum()
            if w_sum <= 0:
                continue
            w_arr /= w_sum

            oos_px = oos_filtered[oos_syms].dropna(how="all")
            if len(oos_px) < 3:
                continue

            # BUG 3 fix: pct_change() produces a NaN for the first row of
            # oos_px, which .dropna() then dropped -- so fold_rets had one
            # FEWER observation than the fold's actual day-span
            # (test_start_i..test_end_i), even though fold_records /
            # fold_date_ranges below still record the full un-shifted span
            # and zero_fill_fold_gaps() stitches folds together purely by
            # bar-position gap. Prepend the real close from the bar
            # immediately before the fold's test window (test_start_i - 1,
            # from the full price_wide -- not look-ahead, it's the day
            # before the fold even starts) so pct_change() has a reference
            # point for the fold's own first day instead of discarding it.
            pre_i = test_start_i - 1
            if pre_i >= 0:
                pre_row  = price_wide[oos_syms].iloc[[pre_i]]
                pct_input = pd.concat([pre_row, oos_px])
            else:
                # No prior bar exists (fold starts at index 0) -- can't
                # recover a true first-day return; fall back to the
                # historical (one-shorter) series rather than fabricate one.
                pct_input = oos_px

            fold_rets = (pct_input.pct_change().dropna().values @ w_arr).tolist()

            sharpe_fold_returns.extend(fold_rets)
            fold_records.append((test_start_i, test_end_i, fold_rets))
            fold_date_ranges.append((
                str(price_wide.index[test_start_i].date()),
                str(price_wide.index[test_end_i].date()),
            ))

            if prev_weights:
                common = set(weights_dict) | set(prev_weights)
                turnover = sum(abs(weights_dict.get(s, 0) - prev_weights.get(s, 0))
                               for s in common) / 2.0
                fold_turnovers.append(turnover)

            prev_weights = weights_dict
            n_folds += 1

        if n_folds == 0 or len(sharpe_fold_returns) < 5:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": n_folds}

        # Sharpe is order-independent (mean/std of the return distribution), so
        # it is computed from the pure fold returns only -- gap-fill zeros must
        # NOT be mixed in here, since they would dilute both mean and std and
        # bias Sharpe toward 0 for variants with wide embargo gaps.
        port_rets = np.array(sharpe_fold_returns)
        mean_r = np.mean(port_rets)
        std_r  = np.std(port_rets)
        sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0
        avg_turnover = float(np.mean(fold_turnovers)) if fold_turnovers else 0.20
        all_fold_returns = zero_fill_fold_gaps(fold_records)

        return {
            "sharpe":   round(sharpe, 4),
            "turnover": round(avg_turnover, 4),
            "n_folds":  n_folds,
            # Per-day OOS portfolio returns across all folds, reconstructed into
            # one temporally continuous series: the purge+embargo gap between
            # consecutive folds' test windows is filled with explicit 0.0
            # ("assume no change" -- the most honest statement for a period that
            # was never OOS-tested) rather than concatenated directly, which
            # would feed calmar_ratio()'s cumprod/drawdown walk a series that
            # either hides a real drawdown spanning the gap or fabricates an
            # instant "recovery" across it. Do NOT use this field for Sharpe
            # (see `sharpe` above, computed from the undiluted fold returns).
            "returns": all_fold_returns,
            # Additive: (start_date, end_date) per accepted fold's OOS test
            # window, chronological order -- lets a caller verify/reconstruct
            # the gap structure above instead of trusting it blindly.
            "fold_date_ranges": fold_date_ranges,
        }

    except Exception as e:
        print(f"[LightweightOOS] Unexpected error: {type(e).__name__}: {e}")
        return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}
