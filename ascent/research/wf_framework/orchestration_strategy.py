"""
Full orchestration WF strategy — simulates the live 4-agent system.

Agents and base allocations (mirrors orchestrator/central_intelligence.py):
  calm_bull : US 60% / macro 15% / intl 15% / alt 10%
  stressed  : US 45% / macro 25% / intl 10% / alt 20%
  crisis    : US 30% / macro 30% / intl  5% / alt 35%

Regime detected from FRED (same logic as MultiAssetPortfolioStrategy).

Sub-portfolio construction:
  US equity   : AscentPortfolioStrategy (full alpha stack, sector-constrained)
  Macro       : cross-sectional momentum on 12 ETFs, top-4 equal weight
  International: cross-sectional momentum on 12 ETFs, top-4, max 2 per region
  Alternatives : trend momentum on 7 ETFs, top-3 equal weight
"""
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from .multi_asset_strategy import MultiAssetPortfolioStrategy


MACRO_UNIVERSE = ['TLT','IEF','UUP','GLD','PDBC','HYG','LQD','TIP','SGOV','BIL','DBB','KMLM']
INTL_UNIVERSE  = ['EEM','VWO','EWT','AAXJ','EWJ','EWZ','EWC','EWY','INDA','EWG','EWU','EFA']
ALT_UNIVERSE   = ['VNQ','GLD','PDBC','DBA','IFRA','VIXY','BIL']

# Region map for international max-2-per-region cap
REGION_MAP = {
    'EEM': 'em_broad', 'VWO': 'em_broad',
    'EWT': 'asia', 'AAXJ': 'asia', 'EWJ': 'asia', 'INDA': 'asia',
    'EWZ': 'latam',
    'EWC': 'na_developed',
    'EWY': 'korea',
    'EWG': 'europe', 'EWU': 'europe', 'EFA': 'europe',
}

# Regime allocations matching live orchestrator (calm_bull baseline)
REGIME_ALLOCS = {
    'crisis':    {'us': 0.30, 'macro': 0.30, 'intl': 0.05, 'alt': 0.35},
    'stressed':  {'us': 0.45, 'macro': 0.25, 'intl': 0.10, 'alt': 0.20},
    'calm_bull': {'us': 0.60, 'macro': 0.15, 'intl': 0.15, 'alt': 0.10},
}


class FullOrchestrationStrategy(MultiAssetPortfolioStrategy):
    """
    WF simulation of the live 4-agent orchestrated portfolio.
    Inherits US equity alpha stack + FRED regime detection from parent.
    Adds international and alternatives sub-portfolios.
    """

    def generate_signals(
        self,
        data: pd.DataFrame,
        pit_boundary: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        # Step 1: equity weights (from grandparent AscentPortfolioStrategy)
        # Call grandparent directly to get equity-only weights
        from .ascent_strategy import AscentPortfolioStrategy
        equity_weighted = AscentPortfolioStrategy.generate_signals(self, data, pit_boundary)
        if equity_weighted.empty:
            return equity_weighted

        # Renormalize equity to sum=1 (undo vol targeting + 200MA from parent)
        row_sums = equity_weighted.abs().sum(axis=1).replace(0.0, np.nan)
        equity_normed = equity_weighted.div(row_sums, axis=0).fillna(0.0)

        data_clean = data.copy()
        data_clean["date"] = pd.to_datetime(data_clean["date"])
        if data_clean["date"].dt.tz is not None:
            data_clean["date"] = data_clean["date"].dt.tz_localize(None)

        all_dates   = sorted(data_clean["date"].unique())
        rebal_dates = [all_dates[i] for i in range(0, len(all_dates), self.rebalance_freq)]

        self._load_fred()

        # Step 2: build combined portfolio at each rebalance date
        all_syms: set = set(equity_normed.columns)
        blend_at: dict = {}

        for rebal_date in rebal_dates:
            if rebal_date not in equity_normed.index:
                continue

            allocs = self._four_way_allocation(rebal_date)

            macro_w = self._momentum_weights(data_clean, rebal_date, MACRO_UNIVERSE, top_n=4)
            intl_w  = self._intl_weights(data_clean, rebal_date, top_n=4)
            alt_w   = self._momentum_weights(data_clean, rebal_date, ALT_UNIVERSE,  top_n=3)

            combined: dict[str, float] = {}
            for sym, w in equity_normed.loc[rebal_date].items():
                combined[sym] = combined.get(sym, 0.0) + w * allocs['us']
            for sym, w in macro_w.items():
                combined[sym] = combined.get(sym, 0.0) + w * allocs['macro']
            for sym, w in intl_w.items():
                combined[sym] = combined.get(sym, 0.0) + w * allocs['intl']
            for sym, w in alt_w.items():
                combined[sym] = combined.get(sym, 0.0) + w * allocs['alt']

            all_syms.update(combined.keys())
            blend_at[rebal_date] = combined

        if not blend_at:
            return equity_weighted

        # Step 3: DataFrame + ffill
        all_syms = sorted(all_syms)
        blended  = pd.DataFrame(0.0, index=pd.DatetimeIndex(all_dates), columns=all_syms)
        for rebal_date, combined in blend_at.items():
            for sym, w in combined.items():
                if sym in blended.columns:
                    blended.loc[rebal_date, sym] = w

        blended = (
            blended.loc[blended.index.isin(rebal_dates)]
            .reindex(pd.DatetimeIndex(all_dates))
            .ffill()
            .fillna(0.0)
        )

        # Step 4: single combined overlay
        blended = self._apply_200ma_overlay(data_clean, blended)
        blended = self._apply_vol_target(data_clean, blended)
        return blended

    # ------------------------------------------------------------------ #
    # Four-way allocation from FRED regime                                #
    # ------------------------------------------------------------------ #

    def _four_way_allocation(self, as_of: pd.Timestamp) -> dict[str, float]:
        if self._macro_fred is None:
            return REGIME_ALLOCS['calm_bull']
        try:
            fred = self._macro_fred
            past = fred[fred["date"] <= as_of]

            def _last(sid):
                r = past[past["series_id"] == sid].sort_values("date")
                return float(r["value"].iloc[-1]) if not r.empty else None

            def _6m_ago(sid):
                cutoff = as_of - pd.DateOffset(months=6)
                r = past[(past["series_id"] == sid) & (past["date"] >= cutoff)].sort_values("date")
                return float(r["value"].iloc[0]) if not r.empty else None

            ys   = _last("T10Y2Y") or _last("yield_spread_10y2y") or 1.0
            ff   = _last("DFF")    or _last("fed_funds_rate")
            ff6  = _6m_ago("DFF")  or _6m_ago("fed_funds_rate")
            rise = (ff - ff6) if (ff is not None and ff6 is not None) else 0.0

            if ys < -0.5 and rise > 2.0:
                return REGIME_ALLOCS['crisis']
            if ys < 0.0 or rise > 1.0:
                return REGIME_ALLOCS['stressed']
            return REGIME_ALLOCS['calm_bull']
        except Exception:
            return REGIME_ALLOCS['calm_bull']

    # ------------------------------------------------------------------ #
    # Sub-portfolio builders                                               #
    # ------------------------------------------------------------------ #

    def _momentum_weights(
        self, data: pd.DataFrame, rebal_date: pd.Timestamp,
        universe: list, top_n: int = 4,
    ) -> dict[str, float]:
        """Cross-sectional momentum: top-N by trailing return, equal weight."""
        try:
            past = data[(data["symbol"].isin(universe)) & (data["date"] <= rebal_date)]
            if past.empty:
                return {universe[-1]: 1.0}
            window = min(self.mom_window, 63)
            mom = {}
            for sym, grp in past.groupby("symbol"):
                closes = grp.sort_values("date")["close"].values.astype(float)
                if len(closes) >= window:
                    mom[sym] = float(closes[-1] / closes[-window] - 1)
            if not mom:
                return {universe[-1]: 1.0}
            top = sorted(mom, key=mom.get, reverse=True)[:top_n]
            return {s: 1.0 / len(top) for s in top}
        except Exception:
            return {universe[-1]: 1.0}

    def _intl_weights(
        self, data: pd.DataFrame, rebal_date: pd.Timestamp, top_n: int = 4,
    ) -> dict[str, float]:
        """International: momentum-ranked, max 2 per region."""
        try:
            past = data[(data["symbol"].isin(INTL_UNIVERSE)) & (data["date"] <= rebal_date)]
            if past.empty:
                return {"EFA": 1.0}
            window = min(self.mom_window, 63)
            mom = {}
            for sym, grp in past.groupby("symbol"):
                closes = grp.sort_values("date")["close"].values.astype(float)
                if len(closes) >= window:
                    mom[sym] = float(closes[-1] / closes[-window] - 1)
            if not mom:
                return {"EFA": 1.0}
            ranked   = sorted(mom, key=mom.get, reverse=True)
            selected = []
            region_count: dict[str, int] = {}
            for sym in ranked:
                region = REGION_MAP.get(sym, sym)
                if region_count.get(region, 0) < 2:
                    selected.append(sym)
                    region_count[region] = region_count.get(region, 0) + 1
                if len(selected) >= top_n:
                    break
            if not selected:
                return {"EFA": 1.0}
            return {s: 1.0 / len(selected) for s in selected}
        except Exception:
            return {"EFA": 1.0}
