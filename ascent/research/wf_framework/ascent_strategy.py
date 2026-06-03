"""
Ascent Portfolio Strategy
==========================
Wraps the Ascent quant alpha pipeline as a walk-forward portfolio strategy.

Smart caching
-------------
Feature computation (FeatureBuilder) and alpha computation (build_alpha_stack)
are cached at the CLASS level so all instances in a single optimizer grid search
share results. The engine calls `AscentPortfolioStrategy.clear_cache()` before
each fold's IS optimization to prevent stale entries from bleeding across folds.

Per fold, 243 combos collapse to:
  - 3 FeatureBuilder calls  (one per mom_window value)
  - 27 build_alpha_stack calls  (one per mom_window × trend_weight × statarb_weight)
  - 243 sector_constrained_weighted calls  (fast, ~0.05s each)

Boundary defense
----------------
generate_signals is strictly causal: it uses only data[date <= call_date] via
FeatureBuilder's rolling operations. No forward returns or future prices are used.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from .portfolio_strategy import PortfolioBaseStrategy

from ascent.alpha.stack import DEFAULT_ALPHA_WEIGHTS, build_alpha_stack
from ascent.features.build_features import FeatureBuilder
from ascent.portfolio.optimizer import sector_constrained_weighted


class AscentPortfolioStrategy(PortfolioBaseStrategy):
    """Ascent quant alpha stack as a PortfolioBaseStrategy."""

    # Class-level caches: shared across all instances in the same optimize() loop
    _feature_cache: dict = {}
    _alpha_cache:   dict = {}

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
        """Clear class-level caches. Engine calls this before each fold."""
        cls._feature_cache.clear()
        cls._alpha_cache.clear()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        data : long-format OHLCV with 'date' and 'symbol' columns.
        Returns pd.DataFrame (dates × symbols) of portfolio weights, ffilled.
        """
        data = data.copy()
        data["date"] = pd.to_datetime(data["date"])

        all_dates = sorted(data["date"].unique())
        if not all_dates:
            return pd.DataFrame()

        # Cache key: uniquely identifies this data window
        data_key = (all_dates[0], all_dates[-1], data["symbol"].nunique())

        # --- Step 1: Features (cached per mom_window) ---
        feat_key = (*data_key, self.mom_window)
        if feat_key not in AscentPortfolioStrategy._feature_cache:
            if self.mom_window < len(all_dates):
                cutoff     = all_dates[-self.mom_window]
                data_slice = data[data["date"] >= cutoff]
            else:
                data_slice = data
            builder = FeatureBuilder(data_slice, macro_df=None)
            AscentPortfolioStrategy._feature_cache[feat_key] = builder.compute_features()
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

        rebal_dates = [alpha_dates[i] for i in range(0, len(alpha_dates), self.rebalance_freq)]
        alpha_at_rebal = alpha_scores.loc[rebal_dates]

        weights_at_rebal = sector_constrained_weighted(
            alpha_at_rebal,
            n=self.top_n,
            max_weight=self.max_weight,
            sector_map=self.sector_map,
            regime_signal=None,
        )

        # Forward-fill to every trading day in the alpha index
        weights_ffilled = (
            weights_at_rebal
            .reindex(alpha_dates)
            .ffill()
            .fillna(0.0)
        )
        return weights_ffilled

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
