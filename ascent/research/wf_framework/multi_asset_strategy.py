"""
Multi-asset WF strategy: US equity momentum + macro trend-following.

Inherits AscentPortfolioStrategy's efficient per-fold caching, then blends
the equity weights with a macro ETF trend portfolio at each rebalance date.

Blend ratio driven by FRED macro regime:
  crisis   (yield_curve < -0.5 AND fed_funds +2% / 6m): equity 25% / macro 75%
  stressed (yield_curve < 0  OR  fed_funds +1% / 6m):   equity 45% / macro 55%
  calm_bull (default):                                    equity 70% / macro 30%

Macro sub-portfolio: cross-sectional momentum on 10 ETFs — picks top-3.
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from .ascent_strategy import AscentPortfolioStrategy
from ascent.portfolio.optimizer import sector_constrained_weighted


MACRO_UNIVERSE = ['TLT', 'IEF', 'GLD', 'PDBC', 'BIL', 'KMLM', 'HYG', 'LQD', 'DBB', 'UUP']


class MultiAssetPortfolioStrategy(AscentPortfolioStrategy):
    """
    Extends AscentPortfolioStrategy with macro blending.
    Equity signals are computed identically (same caching, same alpha stack).
    Macro signals are added at the portfolio construction step.
    """

    # FRED data loaded once per process
    _macro_fred: Optional[pd.DataFrame] = None
    _fred_loaded: bool = False

    def __init__(
        self,
        top_n:           int   = 15,
        max_weight:      float = 0.10,
        trend_weight:    float = 0.38,
        statarb_weight:  float = 0.15,
        mom_window:      int   = 252,
        sector_map:      Optional[dict] = None,
        rebalance_freq:  int   = 10,
        alpha_weights_override: Optional[dict] = None,
        fixed_params:    Optional[dict] = None,
        macro_top_n:     int   = 3,
        base_equity_wt:  float = 0.70,
    ):
        super().__init__(
            top_n=top_n, max_weight=max_weight,
            trend_weight=trend_weight, statarb_weight=statarb_weight,
            mom_window=mom_window, sector_map=sector_map,
            rebalance_freq=rebalance_freq,
            alpha_weights_override=alpha_weights_override,
            fixed_params=fixed_params,
        )
        self.macro_top_n   = macro_top_n
        self.base_equity_wt = base_equity_wt

    # ------------------------------------------------------------------ #
    # Override: blend equity + macro after equity signals are built       #
    # ------------------------------------------------------------------ #

    def generate_signals(
        self,
        data: pd.DataFrame,
        pit_boundary: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        # Step 1: equity weights from parent (already vol-targeted + 200MA scaled)
        equity_weighted = super().generate_signals(data, pit_boundary)
        if equity_weighted.empty:
            return equity_weighted

        # Step 2: RENORMALIZE equity weights to sum=1 per row.
        # The parent applied vol targeting + 200MA overlay — those reduce the sum
        # below 1.0. We undo that here so the blend is at full portfolio scale,
        # then apply a SINGLE combined overlay at the end.
        row_sums = equity_weighted.abs().sum(axis=1).replace(0.0, np.nan)
        equity_normed = equity_weighted.div(row_sums, axis=0).fillna(0.0)

        data_clean = data.copy()
        data_clean["date"] = pd.to_datetime(data_clean["date"])
        if data_clean["date"].dt.tz is not None:
            data_clean["date"] = data_clean["date"].dt.tz_localize(None)

        all_dates   = sorted(data_clean["date"].unique())
        rebal_dates = [all_dates[i] for i in range(0, len(all_dates), self.rebalance_freq)]

        self._load_fred()

        # Step 3: at each rebalance date, blend equity + macro
        all_syms: set = set(equity_normed.columns)
        blend_at: dict[pd.Timestamp, dict[str, float]] = {}

        for rebal_date in rebal_dates:
            if rebal_date not in equity_normed.index:
                continue
            eq_alloc, macro_alloc = self._regime_allocation(rebal_date)
            macro_w  = self._macro_weights_at(data_clean, rebal_date)
            all_syms.update(macro_w.keys())

            combined: dict[str, float] = {}
            # Equity side
            for sym, w in equity_normed.loc[rebal_date].items():
                combined[sym] = w * eq_alloc
            # Macro side (additive — macro ETFs may also appear in equity universe)
            for sym, w in macro_w.items():
                combined[sym] = combined.get(sym, 0.0) + w * macro_alloc

            blend_at[rebal_date] = combined

        if not blend_at:
            return equity_weighted

        # Step 4: build blended DataFrame and ffill
        all_syms = sorted(all_syms)
        blended  = pd.DataFrame(0.0, index=pd.DatetimeIndex(all_dates), columns=all_syms)

        for rebal_date, combined in blend_at.items():
            for sym, w in combined.items():
                if sym in blended.columns:
                    blended.loc[rebal_date, sym] = w

        blended = (
            blended
            .loc[blended.index.isin(rebal_dates)]
            .reindex(pd.DatetimeIndex(all_dates))
            .ffill()
            .fillna(0.0)
        )

        # Step 5: apply 200MA overlay + vol targeting ONCE on the combined portfolio
        blended = self._apply_200ma_overlay(data_clean, blended)
        blended = self._apply_vol_target(data_clean, blended)

        return blended

    # ------------------------------------------------------------------ #
    # Macro sub-portfolio                                                  #
    # ------------------------------------------------------------------ #

    def _macro_weights_at(
        self,
        data: pd.DataFrame,
        rebal_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Cross-sectional momentum on macro ETFs — top-N equal weight."""
        try:
            past = data[
                (data["symbol"].isin(MACRO_UNIVERSE)) &
                (data["date"] <= rebal_date)
            ]
            if past.empty:
                return {"BIL": 1.0}

            window = min(self.mom_window, 63)
            momentum: dict[str, float] = {}
            for sym, grp in past.groupby("symbol"):
                closes = grp.sort_values("date")["close"].values.astype(float)
                if len(closes) >= window:
                    momentum[sym] = float(closes[-1] / closes[-window] - 1)

            if not momentum:
                return {"BIL": 1.0}

            top = sorted(momentum, key=momentum.get, reverse=True)[:self.macro_top_n]
            w   = 1.0 / len(top)
            return {sym: w for sym in top}
        except Exception:
            return {"BIL": 1.0}

    # ------------------------------------------------------------------ #
    # FRED macro regime detection                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def _load_fred(cls) -> None:
        if cls._fred_loaded:
            return
        try:
            from ascent.data.store.parquet import load_parquet, has_data
            if has_data("macro_live"):
                df = load_parquet("macro_live")
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                cls._macro_fred = df
        except Exception:
            pass
        cls._fred_loaded = True

    def _regime_allocation(self, as_of: pd.Timestamp) -> tuple[float, float]:
        """
        Returns (equity_weight, macro_weight) from FRED signals at as_of.
        Falls back to base_equity_wt if FRED data unavailable.
        """
        if self._macro_fred is None:
            return self.base_equity_wt, 1.0 - self.base_equity_wt
        try:
            fred = self._macro_fred
            past = fred[fred["date"] <= as_of]

            def _last(sid: str) -> Optional[float]:
                r = past[past["series_id"] == sid].sort_values("date")
                return float(r["value"].iloc[-1]) if not r.empty else None

            def _6m_ago(sid: str) -> Optional[float]:
                cutoff = as_of - pd.DateOffset(months=6)
                r = past[(past["series_id"] == sid) & (past["date"] >= cutoff)]
                r = r.sort_values("date")
                return float(r["value"].iloc[0]) if not r.empty else None

            ys   = _last("T10Y2Y") or _last("yield_spread_10y2y") or 1.0
            ff   = _last("DFF")    or _last("fed_funds_rate")
            ff6  = _6m_ago("DFF")  or _6m_ago("fed_funds_rate")
            rise = (ff - ff6) if (ff is not None and ff6 is not None) else 0.0

            if ys < -0.5 and rise > 2.0:   # crisis: inverted + aggressive hikes
                return 0.25, 0.75
            if ys < 0.0 or rise > 1.0:     # stressed: inverted or hiking
                return 0.45, 0.55
            return self.base_equity_wt, 1.0 - self.base_equity_wt
        except Exception:
            return self.base_equity_wt, 1.0 - self.base_equity_wt
