"""
Ascent Portfolio Strategy
==========================
Wraps the Ascent quant alpha pipeline as a walk-forward portfolio strategy.

Full alpha stack
----------------
All 13 live sleeves are activated by loading the auxiliary data caches
(fundamentals, earnings, analyst revisions, options flow, insider transactions,
short interest) at class level and point-in-time slicing them per fold.

ML sleeve is intentionally excluded: enabling it safely requires knowing the
IS/OOS boundary inside generate_signals (to avoid using OOS forward returns
as training targets). The other sleeves cover 15+ percent of alpha weight with
no such constraint.

Smart caching
-------------
Feature computation (FeatureBuilder) and alpha computation (build_alpha_stack)
are cached at the CLASS level so all 243 grid-search instances within a single
fold share results. The engine calls `AscentPortfolioStrategy.clear_cache()`
before each fold's IS optimization to prevent stale entries bleeding across folds.

Aux data (_aux_data) is loaded ONCE per process and never cleared — it is
static market data that only grows over time.

Per fold, 243 combos collapse to:
  - 3 FeatureBuilder calls  (one per mom_window value)
  - 27 build_alpha_stack calls  (one per mom_window × trend_weight × statarb_weight)
  - 243 sector_constrained_weighted calls  (fast, ~0.05s each)

Boundary defense
----------------
generate_signals is strictly causal:
  - Price features use only rolling windows (no future prices).
  - Auxiliary data (fundamentals, earnings, etc.) is point-in-time sliced to
    the last date in the data window before being passed to FeatureBuilder.
  - ML sleeve targets are excluded to avoid inadvertent OOS target leakage.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from .portfolio_strategy import PortfolioBaseStrategy

from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS, build_alpha_stack
from ascent.features.build_features import FeatureBuilder
from ascent.portfolio.optimizer import sector_constrained_weighted


# Mapping: (FeatureBuilder kwarg, parquet cache name, date column for PIT slice)
_AUX_CACHES = [
    ("fundamentals_df", "fundamentals",         "date"),
    ("earnings_df",     "earnings",             "signal_date"),
    ("analyst_df",      "analyst_revisions",    "signal_date"),
    ("options_df",      "options_flow",         "date"),
    ("insider_df",      "insider_transactions", "signal_date"),
    ("short_df",        "short_interest",       "date"),
]


class AscentPortfolioStrategy(PortfolioBaseStrategy):
    """Ascent quant alpha stack as a PortfolioBaseStrategy."""

    # Feature / alpha caches — cleared per fold by the engine
    _feature_cache: dict = {}
    _alpha_cache:   dict = {}

    # Auxiliary data — loaded once, never cleared
    # Format: {kwarg_name: (DataFrame | None, date_col)}
    _aux_data:   dict = {}
    _aux_loaded: bool = False

    def __init__(
        self,
        top_n:          int   = 15,
        max_weight:     float = 0.10,
        trend_weight:   float = 0.38,
        statarb_weight: float = 0.15,
        mom_window:     int   = 252,
        sector_map:     Optional[dict] = None,
        rebalance_freq: int   = 10,
    ):
        super().__init__(
            top_n=top_n,
            max_weight=max_weight,
            trend_weight=trend_weight,
            statarb_weight=statarb_weight,
            mom_window=mom_window,
        )
        self.sector_map     = sector_map if sector_map is not None else self._load_sector_map()
        self.rebalance_freq = rebalance_freq

    # ------------------------------------------------------------------
    # PortfolioBaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> dict[str, list]:
        return {
            "top_n":          [10, 15, 20],
            "max_weight":     [0.08, 0.10, 0.12],
            "trend_weight":   [0.30, 0.38, 0.50],
            "statarb_weight": [0.10, 0.15, 0.20],
            "mom_window":     [63, 126, 252],
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Clear per-fold feature/alpha caches. Aux data is NOT cleared."""
        cls._feature_cache.clear()
        cls._alpha_cache.clear()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data : long-format OHLCV with 'date' and 'symbol' columns.
        Returns pd.DataFrame (dates × symbols) of portfolio weights, ffilled.
        """
        # Ensure auxiliary data is loaded (runs once per process)
        self._ensure_aux_loaded()

        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])
        # Strip tz so all downstream date comparisons (fundamental, analyst, etc.)
        # use tz-naive timestamps consistently
        if data["date"].dt.tz is not None:
            data["date"] = data["date"].dt.tz_localize(None)

        all_dates = sorted(data["date"].unique())
        if not all_dates:
            return pd.DataFrame()

        as_of_date = all_dates[-1]

        # Cache key: uniquely identifies this data window
        data_key = (all_dates[0], as_of_date, data["symbol"].nunique())

        # --- Step 1: Features (cached per mom_window) ---
        feat_key = (*data_key, self.mom_window)
        if feat_key not in AscentPortfolioStrategy._feature_cache:
            if self.mom_window < len(all_dates):
                cutoff     = all_dates[-self.mom_window]
                data_slice = data[data["date"] >= cutoff]
            else:
                data_slice = data

            # Point-in-time slice all auxiliary data to as_of_date
            fb_kwargs = self._pit_aux_kwargs(as_of_date)

            builder  = FeatureBuilder(data_slice, macro_df=None, **fb_kwargs)
            features = builder.compute_features()

            AscentPortfolioStrategy._feature_cache[feat_key] = features
        features = AscentPortfolioStrategy._feature_cache[feat_key]

        # --- Step 2: Alpha stack (cached per mom_window + sleeve blend) ---
        alpha_key = (*data_key, self.mom_window,
                     round(self.trend_weight, 4), round(self.statarb_weight, 4))
        if alpha_key not in AscentPortfolioStrategy._alpha_cache:
            alpha_weights = self._make_alpha_weights()
            AscentPortfolioStrategy._alpha_cache[alpha_key] = build_alpha_stack(
                features, alpha_weights=alpha_weights, agent_id="us_equities"
            )
        alpha_scores = AscentPortfolioStrategy._alpha_cache[alpha_key]

        # --- Step 3: Portfolio construction at rebalance dates ---
        alpha_dates = alpha_scores.index
        if len(alpha_dates) == 0:
            return pd.DataFrame()

        rebal_dates    = [alpha_dates[i] for i in range(0, len(alpha_dates), self.rebalance_freq)]
        alpha_at_rebal = alpha_scores.loc[rebal_dates]

        weights_at_rebal = sector_constrained_weighted(
            alpha_at_rebal,
            n=self.top_n,
            max_weight=self.max_weight,
            sector_map=self.sector_map,
            regime_signal=None,
        )

        weights_ffilled = (
            weights_at_rebal
            .reindex(alpha_dates)
            .ffill()
            .fillna(0.0)
        )
        return weights_ffilled

    # ------------------------------------------------------------------
    # Auxiliary data loading
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_aux_loaded(cls) -> None:
        """Load all auxiliary data caches once. Subsequent calls are no-ops."""
        if cls._aux_loaded:
            return
        try:
            from ascent.data.store.parquet import load_parquet, has_data
        except ImportError:
            cls._aux_loaded = True
            return

        loaded, missing = [], []
        for kwarg_name, cache_name, date_col in _AUX_CACHES:
            if has_data(cache_name):
                try:
                    df = load_parquet(cache_name)
                    df[date_col] = pd.to_datetime(df[date_col])
                    if df[date_col].dt.tz is not None:
                        df[date_col] = df[date_col].dt.tz_localize(None)
                    cls._aux_data[kwarg_name] = (df, date_col)
                    loaded.append(cache_name)
                except Exception as e:
                    cls._aux_data[kwarg_name] = (None, date_col)
                    missing.append(f"{cache_name}(err:{e})")
            else:
                cls._aux_data[kwarg_name] = (None, date_col)
                missing.append(cache_name)

        print(f"[AscentStrategy] Aux data loaded: {loaded}")
        if missing:
            print(f"[AscentStrategy] Aux data missing (sleeves zeroed): {missing}")
        cls._aux_loaded = True

    @classmethod
    def _pit_aux_kwargs(cls, as_of_date: pd.Timestamp) -> dict:
        """Return PIT-sliced auxiliary DataFrames for FeatureBuilder kwargs."""
        # Normalize to tz-naive so comparisons work regardless of prices_live tz
        ts = pd.Timestamp(as_of_date)
        as_of_naive = ts.tz_localize(None) if ts.tzinfo is not None else ts

        kwargs = {}
        for kwarg_name, (df, date_col) in cls._aux_data.items():
            if df is None or df.empty:
                kwargs[kwarg_name] = None
            else:
                col = pd.to_datetime(df[date_col])
                if col.dt.tz is not None:
                    col = col.dt.tz_localize(None)
                mask = col <= as_of_naive
                if not mask.any():
                    kwargs[kwarg_name] = None
                    continue
                sliced = df.loc[mask].copy()
                # Strip tz from the date column so downstream alpha functions
                # can compare against tz-naive price dates without errors
                if isinstance(sliced[date_col].dtype, pd.DatetimeTZDtype):
                    sliced[date_col] = sliced[date_col].dt.tz_localize(None)
                kwargs[kwarg_name] = sliced
        return kwargs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_alpha_weights(self) -> dict:
        """Return sleeve weight dict with trend/statarb overrides, others scaled."""
        base = dict(DEFAULT_ALPHA_WEIGHTS)
        base["trend"]   = self.trend_weight
        base["statarb"] = self.statarb_weight

        other_keys        = [k for k in base if k not in ("trend", "statarb")]
        other_default_sum = sum(DEFAULT_ALPHA_WEIGHTS[k] for k in other_keys)
        remaining         = 1.0 - self.trend_weight - self.statarb_weight

        if other_default_sum > 1e-9 and remaining > 0:
            scale = remaining / other_default_sum
            for k in other_keys:
                base[k] = DEFAULT_ALPHA_WEIGHTS[k] * scale

        return base

    @staticmethod
    def _load_sector_map() -> dict:
        """Load sector map from profiles.parquet if available; empty dict otherwise."""
        try:
            from ascent.data.store.parquet import load_parquet, has_data
            if has_data("profiles"):
                profiles = load_parquet("profiles")
                return dict(zip(profiles["symbol"], profiles["sector"]))
        except Exception:
            pass
        return {}
