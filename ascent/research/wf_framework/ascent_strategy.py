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
from ascent.portfolio.optimizer import sector_constrained_weighted, enforce_risk_budget_cap


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
        top_n:                  int   = 15,
        max_weight:             float = 0.10,
        trend_weight:           float = 0.38,
        statarb_weight:         float = 0.15,
        mom_window:             int   = 252,
        sector_map:             Optional[dict] = None,
        rebalance_freq:         int   = 10,
        alpha_weights_override: Optional[dict] = None,
        fixed_params:           Optional[dict] = None,
    ):
        super().__init__(
            top_n=top_n,
            max_weight=max_weight,
            trend_weight=trend_weight,
            statarb_weight=statarb_weight,
            mom_window=mom_window,
        )
        self.sector_map             = sector_map if sector_map is not None else self._load_sector_map()
        self.rebalance_freq         = rebalance_freq
        self.alpha_weights_override = alpha_weights_override
        self.fixed_params           = fixed_params  # bypass IS optimizer when set
        self.short_n                = 0  # set via subclass or direct assignment

    # ------------------------------------------------------------------
    # PortfolioBaseStrategy interface
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> dict[str, list]:
        # When fixed_params is set the optimizer runs a single-element grid
        # (WFE is meaningless when WFE<0; fixed params remove IS overfit).
        if self.fixed_params:
            return {k: [v] for k, v in self.fixed_params.items()}
        return {
            "trend_weight":   [0.30, 0.38, 0.50],
            "statarb_weight": [0.10, 0.15, 0.20],
            "mom_window":     [63, 126, 252],
        }

    @classmethod
    def clear_cache(cls) -> None:
        """Clear per-fold feature/alpha caches. Aux data is NOT cleared."""
        cls._feature_cache.clear()
        cls._alpha_cache.clear()

    def generate_signals(
        self,
        data: pd.DataFrame,
        pit_boundary: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        data         : long-format OHLCV with 'date' and 'symbol' columns.
        pit_boundary : ceiling for PIT auxiliary data slicing. The engine passes
                       the last IS date so earnings/analyst data does not leak
                       OOS announcements into OOS signals. Defaults to the last
                       date in data (safe for standalone / non-WF use).

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

        # PIT ceiling: use provided boundary (IS end) when available so auxiliary
        # data (earnings, analyst revisions) cannot leak OOS announcements.
        if pit_boundary is not None:
            pb = pd.Timestamp(pit_boundary)
            as_of_date = pb.tz_localize(None) if pb.tzinfo is not None else pb
        else:
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

        # Risk-aware construction — same config flags as production (parity)
        try:
            from ascent.config.settings import get_config as _get_cfg
            _bt = _get_cfg().backtest
            _tilt_on, _cluster_on = _bt.inverse_vol_tilt, _bt.cluster_cap_enabled
            _max_cluster, _cluster_corr = _bt.max_cluster_weight, _bt.cluster_corr_threshold
            _rb_on, _rb_budget = _bt.risk_budget_cap_enabled, _bt.risk_budget_per_name
        except Exception:
            _tilt_on, _cluster_on = True, True
            _max_cluster, _cluster_corr = 0.20, 0.70
            _rb_on, _rb_budget = True, 0.012

        _close_panel = None
        _vol_panel = None
        try:
            _close_panel = data.pivot_table(
                index="date", columns="symbol", values="close", aggfunc="last"
            )
            _close_panel.index = pd.to_datetime(_close_panel.index)
            if getattr(_close_panel.index, "tz", None) is not None:
                _close_panel.index = _close_panel.index.tz_localize(None)
            _close_panel = _close_panel.sort_index()
            if _tilt_on:
                _vol_panel = (_close_panel.pct_change()
                              .rolling(63, min_periods=21).std()
                              .mul(np.sqrt(252)).shift(1))
        except Exception:
            pass

        weights_at_rebal = sector_constrained_weighted(
            alpha_at_rebal,
            n=self.top_n,
            max_weight=self.max_weight,
            sector_map=self.sector_map,
            regime_signal=None,
            vol_panel=_vol_panel,
        )

        # Correlation-cluster cap per rebalance row (causal trailing returns)
        if _cluster_on and _close_panel is not None and not weights_at_rebal.empty:
            try:
                from ascent.portfolio.optimizer import enforce_cluster_cap
                _rets_all = _close_panel.pct_change()
                for _dt in weights_at_rebal.index:
                    _trail = _rets_all.loc[:_dt].iloc[-63:]
                    weights_at_rebal.loc[_dt] = enforce_cluster_cap(
                        weights_at_rebal.loc[_dt], _trail,
                        max_cluster_weight=_max_cluster,
                        corr_threshold=_cluster_corr,
                        max_weight=self.max_weight,
                    )
            except Exception:
                pass

        # Per-name risk-budget cap — same guard as production main.py, kept in
        # parity so research and live sizing don't silently diverge (see
        # ascent/portfolio/exposure.py header + W1 in
        # docs/superpowers/plans/2026-07-27-post-outage-remediation.md).
        if _rb_on and _close_panel is not None and not weights_at_rebal.empty:
            try:
                _rb_vol_panel = _vol_panel
                if _rb_vol_panel is None:
                    _rb_vol_panel = (_close_panel.pct_change()
                                      .rolling(63, min_periods=21).std()
                                      .mul(np.sqrt(252)).shift(1))
                for _dt in weights_at_rebal.index:
                    if _dt in _rb_vol_panel.index:
                        _vol_row = _rb_vol_panel.loc[_dt].reindex(weights_at_rebal.columns)
                    else:
                        _vol_row = pd.Series(dtype=float, index=weights_at_rebal.columns)
                    weights_at_rebal.loc[_dt] = enforce_risk_budget_cap(
                        weights_at_rebal.loc[_dt], _vol_row, budget=_rb_budget,
                    )
            except Exception:
                pass

        # --- Step 3b: Long/short extension ---
        # Short bottom-N momentum stocks with half the gross of the long side.
        # Borrow cost handled in execution model via negative weights → turnover.
        if self.short_n > 0:
            short_alpha = alpha_at_rebal.multiply(-1)
            short_raw   = sector_constrained_weighted(
                short_alpha, n=self.short_n, max_weight=self.max_weight,
                sector_map=self.sector_map, regime_signal=None,
            )
            # Short at 50% gross (100L/50S = net 50% long)
            short_scaled = short_raw * 0.50
            # Subtract short weights from portfolio (negative = short)
            longs  = weights_at_rebal.reindex(columns=weights_at_rebal.columns.union(short_scaled.columns), fill_value=0.0)
            shorts = short_scaled.reindex(index=longs.index, columns=longs.columns, fill_value=0.0)
            weights_at_rebal = longs - shorts

        # --- Step 4: SPY 200MA overlay (matches production pipeline) ---
        # When SPY is below its 200-day MA, cut all weights by 30%.
        # Causal: 200MA computed from prices available at each rebalance date.
        weights_at_rebal = self._apply_200ma_overlay(data, weights_at_rebal)

        # --- Step 5: Volatility targeting ---
        weights_at_rebal = self._apply_vol_target(data, weights_at_rebal)

        # --- Step 6: Momentum-crash overlay (parity with production) ---
        weights_at_rebal = self._apply_momentum_crash_overlay(
            data, weights_at_rebal
        )

        # Position-level stop-loss — parity with production (see
        # docs/superpowers/plans/2026-07-27-position-stop-loss.md). Deliberately
        # LAST, after Step 3b and the Step 4/5 overlays, not right after the
        # risk-budget cap: it reassigns weights_at_rebal from a rebalance-
        # frequency frame to a DAILY frame (a stop has to be able to fire
        # between rebalances), and anything downstream would receive a daily
        # frame it wasn't written for. Placed early, it broke two things:
        # (1) Step 3b's `shorts = short_scaled.reindex(index=longs.index, ...,
        # fill_value=0.0)` line has short_scaled at rebalance frequency vs a
        # daily longs.index, so every non-rebalance day filled the short leg
        # with 0.0, silently zeroing the short book; (2) the Step 4/5 200MA
        # and vol-target overlays are written to compute one scale per
        # rebalance row, held constant by the final ffill — on a daily frame
        # they recompute per calendar day instead, diverging from production.
        try:
            from ascent.config.settings import get_config as _get_cfg2
            _sl = _get_cfg2().backtest
            _sl_on = getattr(_sl, "stop_loss_enabled", False)
        except Exception:
            _sl_on = False

        if _sl_on and _close_panel is not None and not weights_at_rebal.empty:
            try:
                from ascent.portfolio.stop_loss import apply_stop_loss_panel
                _daily = weights_at_rebal.reindex(
                    _close_panel.index, method="ffill"
                ).dropna(how="all")
                _daily, _events = apply_stop_loss_panel(
                    _daily, _close_panel,
                    threshold=_sl.stop_loss_threshold,
                    cooldown_days=_sl.stop_loss_cooldown_days,
                    redistribute=_sl.stop_loss_redistribute,
                )
                weights_at_rebal = _daily
                if _events:
                    # ascent_strategy.py has NO module-level `log` (verified
                    # 2026-07-27) — use a self-contained local logger.
                    import logging as _lg
                    _lg.getLogger(__name__).info(
                        "[StopLoss/WF] %d stop events", len(_events))
            except Exception as _sl_e:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "[StopLoss/WF] skipped: %s", _sl_e)

        weights_ffilled = (
            weights_at_rebal
            .reindex(alpha_dates)
            .ffill()
            .fillna(0.0)
        )
        return weights_ffilled

    def _apply_200ma_overlay(
        self,
        data: pd.DataFrame,
        weights: pd.DataFrame,
        ma_window: int = 200,
        multiplier: float = 0.70,
    ) -> pd.DataFrame:
        """
        When SPY closes below its ma_window-day MA at a rebalance date,
        multiply all weights by `multiplier` (default 0.70 = 30% cut).
        Delegates to ascent/portfolio/exposure.py — the single source of truth
        shared with production. Includes the production VIX>20 confirmation
        (loaded point-in-time from the macro cache; MA-only when unavailable).
        Fully causal — uses only prices available before each rebalance date.
        """
        try:
            from ascent.portfolio.exposure import ma_filter_scale, load_vix_from_macro_cache

            spy = (
                data[data["symbol"] == "SPY"]
                .copy()
                .assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
                .sort_values("date")
                .set_index("date")["close"]
            )
            vix = load_vix_from_macro_cache()
            scale = ma_filter_scale(
                spy, weights.index, vix_close=vix,
                ma_window=ma_window, multiplier=multiplier,
            )
            return weights.mul(scale, axis=0)
        except Exception:
            return weights

    def _apply_momentum_crash_overlay(self, data, weights):
        """
        Daniel & Moskowitz (2016) crash-state cut. Delegates to
        ascent/portfolio/exposure.py — single source of truth shared with
        production. See docs/superpowers/plans/2026-07-27-momentum-crash-indicator.md.

        Reads `momentum_crash_overlay_enabled` / `momentum_crash_multiplier`,
        the same two config fields production reads, so research and production
        cannot silently diverge. Inert while the flag is False.
        """
        if weights is None or weights.empty:
            return weights
        try:
            from ascent.config.settings import get_config as _gc
            _bt = _gc().backtest
            if not getattr(_bt, "momentum_crash_overlay_enabled", False):
                return weights
            mult = float(getattr(_bt, "momentum_crash_multiplier", 0.50))
        except Exception:
            return weights

        try:
            from ascent.portfolio.exposure import momentum_crash_scale

            # Same SPY-loading mechanism as _apply_200ma_overlay — do not add
            # a second way of obtaining the benchmark series.
            spy = (
                data[data["symbol"] == "SPY"]
                .copy()
                .assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
                .sort_values("date")
                .set_index("date")["close"]
            )
            if spy is None or spy.empty:
                return weights
            scale = momentum_crash_scale(spy, weights.index, multiplier=mult)
            return weights.mul(scale, axis=0)
        except Exception as exc:
            # ascent_strategy.py has NO module-level `log` (verified
            # 2026-07-27) — use a self-contained local logger.
            import logging as _lg
            _lg.getLogger(__name__).warning(
                "[Exposure/WF] momentum-crash overlay skipped: %s", exc)
            return weights

    def _apply_vol_target(
        self,
        data: pd.DataFrame,
        weights: pd.DataFrame,
        target_vol: Optional[float] = None,
        lookback: int = 21,
        floor: float = 0.25,
        cap: float = 1.00,
        port_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Scale portfolio weights so expected portfolio volatility ≈ target_vol.
        Uses SPY trailing 21-day vol as proxy. Fully causal.

        scale = clip(target_vol / realized_spy_vol, floor, cap)

        Delegates to ascent/portfolio/exposure.py — single source of truth
        shared with production. `target_vol` defaults to the shared
        `VOL_TARGET` constant so this tracks any future retuning instead of
        drifting from production (see CLAUDE.md history on this divergence).
        """
        try:
            from ascent.portfolio.exposure import (
                vol_target_scale, realized_vol_scale, strategy_return_proxy,
                VOL_TARGET,
            )
            if target_vol is None:
                target_vol = VOL_TARGET

            spy = (
                data[data["symbol"] == "SPY"]
                .copy()
                .assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
                .sort_values("date")
                .set_index("date")["close"]
            )

            try:
                from ascent.config.settings import get_config as _gc
                _ref = str(getattr(_gc().backtest, "vol_target_reference", "spy")).lower()
            except Exception:
                _ref = "spy"

            # `data` is long-format; strategy-own vol targeting needs a
            # (dates x symbols) close panel, so build one on demand.
            _close_panel = None
            if _ref == "strategy":
                try:
                    _close_panel = (
                        data.assign(
                            date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
                        .pivot_table(index="date", columns="symbol",
                                     values="close", aggfunc="last")
                        .sort_index()
                    )
                except Exception:
                    _close_panel = None

            if _ref == "strategy" and _close_panel is not None and not _close_panel.empty:
                # `weights` here is indexed at REBALANCE dates only. The proxy
                # reindexes prices onto the weights index, so feeding it the
                # sparse frame would produce ~10-day returns annualized by
                # sqrt(252) (a ~3x vol overstatement that pins the scale to the
                # floor). Forward-fill the held book onto the daily grid first.
                _daily_w = (
                    weights.reindex(_close_panel.index.union(weights.index))
                    .ffill()
                    .reindex(_close_panel.index)
                    .fillna(0.0)
                )
                scale = realized_vol_scale(
                    strategy_return_proxy(_daily_w, _close_panel),
                    weights.index, target_vol=target_vol,
                    lookback=lookback, floor=floor, cap=cap,
                )
            else:
                scale = vol_target_scale(
                    spy, weights.index, target_vol=target_vol,
                    lookback=lookback, floor=floor, cap=cap,
                )
            return weights.mul(scale, axis=0)
        except Exception:
            return weights

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

    @staticmethod
    def _estimate_regime(data: pd.DataFrame, as_of_date: pd.Timestamp) -> str:
        """
        Price-only regime estimate at `as_of_date` using SPY.
        Used by the WF engine so each fold selects regime-appropriate sleeve weights.

        Labels:  calm_bull | stressed | crisis | uncertain
        Rules (in priority order):
          crisis   : SPY 5d return < -7%  AND  realized_vol_21d > 25% ann.
          stressed : SPY below 200MA      AND  realized_vol_21d > 18% ann.
          calm_bull: SPY above 200MA      AND  realized_vol_21d < 20% ann.
          uncertain: everything else
        """
        try:
            spy = (
                data[data["symbol"] == "SPY"]
                .copy()
                .assign(date=lambda d: pd.to_datetime(d["date"]).dt.tz_localize(None))
                .sort_values("date")
            )
            spy = spy[spy["date"] <= as_of_date]
            if len(spy) < 30:
                return "uncertain"

            closes = spy["close"].values.astype(float)

            # Realized vol: 21-day annualized
            rets = np.diff(closes) / closes[:-1]
            vol21 = float(np.std(rets[-21:]) * np.sqrt(252)) if len(rets) >= 21 else 0.15

            # 200-day MA (use available history, min 50 days)
            ma_window = min(200, len(closes))
            ma200 = float(np.mean(closes[-ma_window:]))
            current = float(closes[-1])

            # 5-day return
            ret5 = float((closes[-1] / closes[-6] - 1)) if len(closes) >= 6 else 0.0

            if ret5 < -0.07 and vol21 > 0.25:
                return "crisis"
            if current < ma200 and vol21 > 0.18:
                return "stressed"
            if current >= ma200 and vol21 < 0.20:
                return "calm_bull"
            return "uncertain"
        except Exception:
            return "uncertain"

    def _make_alpha_weights(self, regime: str = None) -> dict:
        """Return sleeve weight dict with trend/statarb overrides, others scaled.

        Uses regime-specific base weights when regime is provided, falling back
        to DEFAULT_ALPHA_WEIGHTS. alpha_weights_override values are applied last.

        In stressed/crisis the trend cap is also enforced here:
          stressed : trend capped at 0.40 even if optimizer chose higher
          crisis   : trend capped at 0.30
        This prevents the optimizer from running full momentum into a crash.
        """
        # DEFAULT_ALPHA_WEIGHTS_BY_REGIME was removed (only meanrev/statarb survive
        # the reduction, so there is nothing left to regime-tilt between).
        base_defaults = dict(DEFAULT_ALPHA_WEIGHTS)

        if self.alpha_weights_override:
            base_defaults.update(self.alpha_weights_override)

        # Enforce regime-aware trend cap so optimizer can't select 50% trend in crisis
        trend_caps = {"crisis": 0.30, "stressed": 0.40}
        effective_trend = self.trend_weight
        if regime in trend_caps:
            effective_trend = min(self.trend_weight, trend_caps[regime])

        base = dict(base_defaults)
        base["trend"]   = effective_trend
        base["statarb"] = self.statarb_weight

        other_keys        = [k for k in base if k not in ("trend", "statarb")]
        other_default_sum = sum(base_defaults[k] for k in other_keys)
        remaining         = 1.0 - effective_trend - self.statarb_weight

        if other_default_sum > 1e-9 and remaining > 0:
            scale = remaining / other_default_sum
            for k in other_keys:
                base[k] = base_defaults[k] * scale

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
