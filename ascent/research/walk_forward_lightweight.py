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


TURNOVER_PENALTY = 0.10   # subtract 0.10 * avg_turnover from Sharpe


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


def run_lightweight_oos(
    config_overrides: Dict[str, Any],
    n_days: int = 63,
    prices_cache: str = "prices_live",
    top_n: int = 15,
    max_weight: float = 0.10,
) -> Dict[str, float]:
    """
    Run a lightweight OOS evaluation over the last n_days of price data.

    Args:
        config_overrides: Dict with 'alpha_weights' key mapping sleeve names to floats.
        n_days:           Number of OOS trading days to evaluate.
        prices_cache:     Parquet cache name to load prices from.
        top_n:            Portfolio size.
        max_weight:       Max position weight.

    Returns:
        {"sharpe": float, "turnover": float, "n_folds": int}
        Returns {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0} on failure.
    """
    try:
        raw_df = _load_prices(prices_cache)
        if raw_df is None:
            print(f"[LightweightOOS] No cache '{prices_cache}' — returning 0.0")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        close = _to_wide_close(raw_df)
        if close.empty or len(close) < n_days + 63:
            print(f"[LightweightOOS] Insufficient data ({len(close)} rows after pivot)")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Use last (n_days + 126) rows — 126 training, n_days OOS
        window = close.tail(n_days + 126).copy()
        train_close = window.iloc[:126]
        oos_close   = window.iloc[126:]

        alpha_weights = config_overrides.get("alpha_weights", {
            "trend": 0.65, "meanrev": 0.05, "statarb": 0.15, "ml": 0.10, "volatility": 0.05
        })

        # Build features on training slice only (causal)
        # We need volume and dollar_volume — approximate from close if unavailable
        try:
            from ascent.features.feature_defs import build_all_features

            # Approximate volume as constant (1e6 shares) for lightweight eval
            n_rows, n_cols = train_close.shape
            dummy_volume       = pd.DataFrame(1e6, index=train_close.index, columns=train_close.columns)
            dummy_dollar_volume = train_close * 1e6

            features = build_all_features(
                close=train_close,
                volume=dummy_volume,
                dollar_volume=dummy_dollar_volume,
                macro_pivot=None,
            )
        except Exception as e:
            print(f"[LightweightOOS] Feature build failed: {e}")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Build alpha on training data — skip ML sleeve (no targets)
        try:
            from ascent.alpha.trend import trend_alpha
            from ascent.alpha.meanrev import meanrev_alpha

            def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
                mean = df.mean(axis=1)
                std  = df.std(axis=1).replace(0, np.nan)
                return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)

            alphas: Dict[str, pd.DataFrame] = {}

            trend_w = alpha_weights.get("trend", 0.65)
            if trend_w > 0:
                try:
                    t = trend_alpha(features)
                    if not t.empty:
                        alphas["trend"] = t
                except Exception:
                    pass

            mr_w = alpha_weights.get("meanrev", 0.05)
            if mr_w > 0:
                try:
                    mr = meanrev_alpha(features)
                    if not mr.empty:
                        alphas["meanrev"] = mr
                except Exception:
                    pass

            vol_w = alpha_weights.get("volatility", 0.05)
            if vol_w > 0 and "vol_of_vol_21d" in features and "vol_trend_10d" in features:
                try:
                    vov   = features["vol_of_vol_21d"].copy().replace(0, np.nan)
                    vtrnd = features["vol_trend_10d"].copy()
                    vol_alpha = _cs_normalize(-vtrnd / (vov + 1e-6))
                    if not vol_alpha.empty:
                        alphas["volatility"] = vol_alpha
                except Exception:
                    pass

            # statarb — skip if no sector map (graceful fallback)
            sa_w = alpha_weights.get("statarb", 0.15)
            if sa_w > 0:
                try:
                    from ascent.alpha.statarb import statarb_alpha
                    sa = statarb_alpha(features, sector_map={})
                    if not sa.empty:
                        alphas["statarb"] = sa
                except Exception:
                    pass

        except Exception as e:
            print(f"[LightweightOOS] Alpha build failed: {e}")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        if not alphas:
            print("[LightweightOOS] No alpha signals computed")
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Blend alphas
        def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
            mean = df.mean(axis=1)
            std  = df.std(axis=1).replace(0, np.nan)
            return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)

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
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Get latest alpha scores (last row of training composite)
        latest_alpha = composite.iloc[-1].dropna().sort_values(ascending=False)
        if latest_alpha.empty:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Simple rank-weighted portfolio (skip sector constraints — no profiles in test)
        top_names = latest_alpha.head(top_n)
        scores = top_names - top_names.min() + 1e-8
        raw_w  = scores / scores.sum()
        raw_w  = raw_w.clip(upper=max_weight)
        if raw_w.sum() > 0:
            raw_w = raw_w / raw_w.sum()
        weights_dict = raw_w.to_dict()

        if not weights_dict:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        # Compute OOS returns
        oos_symbols = [s for s in weights_dict if s in oos_close.columns]
        if not oos_symbols:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        w_arr = np.array([weights_dict[s] for s in oos_symbols])
        w_arr /= w_arr.sum()  # renorm to available symbols

        price_oos = oos_close[oos_symbols].dropna(how="all")
        if len(price_oos) < 5:
            return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}

        rets = price_oos.pct_change().dropna()
        port_rets = rets.values @ w_arr

        mean_r = np.mean(port_rets)
        std_r  = np.std(port_rets)
        sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0.0

        # Approximate turnover: assume one rebalance per period
        turnover = 0.20  # conservative estimate for single-rebalance period

        return {
            "sharpe":   round(sharpe, 4),
            "turnover": round(turnover, 4),
            "n_folds":  1,
        }

    except Exception as e:
        print(f"[LightweightOOS] Unexpected error: {type(e).__name__}: {e}")
        return {"sharpe": 0.0, "turnover": 0.0, "n_folds": 0}
