"""
ascent/regime/features.py
--------------------------
Regime feature panel builder.

Design principles:
  • Every feature is computed with TRAILING windows only — no future leakage.
  • Missing optional data (VIX, volume) is handled gracefully with NaN fill.
  • All outputs are aligned on the SAME DatetimeIndex as the input prices.
  • Unit-tested transformations keep the math auditable.

Feature groups
  A. Market-state features  (SPY price, vol, trend, skew, gaps)
  B. Cross-sectional features (dispersion, breadth, sector)
  C. Risk / stress features   (VIX, term spread — optional)
  D. Internal strategy state  (rolling IC, turnover — optional)
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────

def _safe_rolling(s: pd.Series, window: int, func: str = "mean", min_periods: Optional[int] = None) -> pd.Series:
    """Apply a rolling function with sane min_periods fallback."""
    mp = min_periods if min_periods is not None else max(1, window // 2)
    return getattr(s.rolling(window, min_periods=mp), func)()


def _log_return(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1))


def _trailing_vol(returns: pd.Series, window: int) -> pd.Series:
    """Annualized trailing realized vol."""
    return _safe_rolling(returns, window, "std") * np.sqrt(252)


def _downside_vol(returns: pd.Series, window: int, threshold: float = 0.0) -> pd.Series:
    """Annualized downside semi-volatility."""
    downside = returns.clip(upper=threshold)
    return _safe_rolling(downside, window, "std") * np.sqrt(252)


def _rolling_drawdown(prices: pd.Series, window: int) -> pd.Series:
    """Rolling drawdown from peak within last `window` days."""
    peak = prices.rolling(window, min_periods=1).max()
    return (prices - peak) / peak


def _skew(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window // 2).skew()


def _kurt(returns: pd.Series, window: int) -> pd.Series:
    return returns.rolling(window, min_periods=window // 2).kurt()


def _pct_above_ma(prices: pd.DataFrame, ma_window: int) -> pd.Series:
    """Cross-sectional breadth: fraction of names above their MA."""
    ma = prices.rolling(ma_window, min_periods=ma_window // 2).mean()
    above = (prices > ma).astype(float)
    return above.mean(axis=1)


def _pct_positive_return(returns: pd.DataFrame, window: int) -> pd.Series:
    """Fraction of names with positive cumulative return over last window days."""
    cum = returns.rolling(window, min_periods=window // 2).sum()
    return (cum > 0).astype(float).mean(axis=1)


def _cross_sectional_dispersion(returns: pd.DataFrame, window: int) -> pd.Series:
    """Rolling cross-sectional return dispersion (std across names)."""
    return returns.rolling(window, min_periods=window // 2).apply(
        lambda x: np.nanstd(x[-1] if x.ndim > 1 else x), raw=True
    )


def _ic_series(alpha_scores: pd.DataFrame, fwd_returns: pd.DataFrame, window: int = 21) -> pd.Series:
    """
    Rolling rank IC between alpha scores and forward returns.
    Both DataFrames must share the same DatetimeIndex and columns.
    """
    from scipy.stats import spearmanr

    def _daily_ic(row_idx: int) -> float:
        a = alpha_scores.iloc[row_idx]
        f = fwd_returns.iloc[row_idx]
        mask = a.notna() & f.notna()
        if mask.sum() < 5:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(a[mask].values, f[mask].values)
        return rho

    ic_raw = pd.Series(
        [_daily_ic(i) for i in range(len(alpha_scores))],
        index=alpha_scores.index,
    )
    return ic_raw.rolling(window, min_periods=window // 2).mean()


# ── main builder ──────────────────────────────────────────────────────────

class RegimeFeatureBuilder:
    """
    Assembles a regime feature DataFrame.

    Parameters
    ----------
    spy_prices : pd.Series
        Daily close prices for SPY (or another index proxy).
    universe_prices : pd.DataFrame, optional
        Daily close prices for the cross-sectional universe (symbols as columns).
    vix_prices : pd.Series, optional
        Daily VIX index levels; used for stress features if provided.
    macro_df : pd.DataFrame, optional
        Macro time series (e.g. FRED data); columns like 'DGS10', 'DFF', etc.
    alpha_scores : pd.DataFrame, optional
        Daily cross-sectional alpha scores (dates × symbols).
    fwd_returns : pd.DataFrame, optional
        Forward return targets aligned to alpha_scores.
    lookbacks : list of int
        Rolling window sizes (days) to use for most features.
    """

    def __init__(
        self,
        spy_prices: pd.Series,
        universe_prices: Optional[pd.DataFrame] = None,
        vix_prices: Optional[pd.Series] = None,
        macro_df: Optional[pd.DataFrame] = None,
        alpha_scores: Optional[pd.DataFrame] = None,
        fwd_returns: Optional[pd.DataFrame] = None,
        lookbacks: Tuple[int, ...] = (5, 21, 63),
    ):
        self.spy          = spy_prices.copy()
        self.universe     = universe_prices.copy() if universe_prices is not None else None
        self.vix          = vix_prices.copy() if vix_prices is not None else None
        self.macro        = macro_df.copy() if macro_df is not None else None
        self.alpha_scores = alpha_scores
        self.fwd_returns  = fwd_returns
        self.lookbacks    = sorted(lookbacks)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not isinstance(self.spy.index, pd.DatetimeIndex):
            raise TypeError("spy_prices must have a DatetimeIndex")
        if not self.spy.index.is_monotonic_increasing:
            self.spy = self.spy.sort_index()
            log.warning("regime.features: spy_prices sorted ascending")

    # ── Group A: market-state features ──────────────────────────────────

    def _build_market_features(self) -> pd.DataFrame:
        spy_ret  = _log_return(self.spy.to_frame("spy"))["spy"]
        features: Dict[str, pd.Series] = {}

        for lb in self.lookbacks:
            features[f"spy_ret_{lb}d"] = self.spy.pct_change(lb)

        for lb in self.lookbacks:
            features[f"spy_rvol_{lb}d"] = _trailing_vol(spy_ret, lb)

        features["spy_downvol_21d"] = _downside_vol(spy_ret, 21)

        for lb in [63, 126]:
            features[f"spy_dd_{lb}d"] = _rolling_drawdown(self.spy, lb)

        for ma in [50, 200]:
            ma_val = self.spy.rolling(ma, min_periods=ma // 2).mean()
            features[f"spy_dist_ma{ma}"] = (self.spy - ma_val) / ma_val

        features["spy_skew_63d"] = _skew(spy_ret, 63)
        features["spy_kurt_63d"] = _kurt(spy_ret, 63)

        features["spy_jump_freq_21d"] = (
            (spy_ret.abs() > 0.015).astype(float)
            .rolling(21, min_periods=10).mean()
        )

        return pd.DataFrame(features, index=self.spy.index)

    # ── Group B: cross-sectional features ───────────────────────────────

    def _build_cs_features(self) -> pd.DataFrame:
        if self.universe is None:
            log.info("regime.features: no universe_prices — skipping cross-sectional group")
            return pd.DataFrame(index=self.spy.index)

        features: Dict[str, pd.Series] = {}

        # Use log returns for vol/dispersion features (additive, symmetric)
        univ_log_ret = _log_return(self.universe)

        # Breadth
        for ma in [20, 50, 200]:
            features[f"breadth_ma{ma}"] = _pct_above_ma(self.universe, ma)
        for lb in [5, 21]:
            features[f"breadth_pos{lb}d"] = _pct_positive_return(univ_log_ret, lb)

        # Cross-sectional return dispersion
        for lb in self.lookbacks:
            daily_std = univ_log_ret.std(axis=1)
            features[f"cs_disp_{lb}d"] = daily_std.rolling(lb, min_periods=lb // 2).mean()

        # ── FIX #27: EW-vs-SPY spread ────────────────────────────────────────
        #
        # BEFORE (broken):
        #   univ_ret  = _log_return(self.universe)          # log returns
        #   ew_index  = (1 + univ_ret).prod(...) ** (1/n)  # wrong: (1 + log_ret)
        #                                                   # is not a gross return
        #   ew_ret    = np.log(ew_index.clip(lower=1e-9))  # logs it again
        #   spy_daily = _log_return(spy)["spy"]             # log return
        #   spread    = (ew_ret - spy_daily).rolling(21)    # mixed math throughout
        #
        # FIX: use simple returns consistently for the EW index construction.
        #   1. Simple returns from prices (pct_change).
        #   2. EW daily return = arithmetic mean across symbols (simple ret basis).
        #   3. SPY simple return on the same basis.
        #   4. Roll the daily spread over 21 days.
        #
        # This matches what a directly computed equal-weight index produces.
        univ_simple_ret = self.universe.pct_change()   # simple returns
        spy_simple_ret  = self.spy.pct_change()        # SPY simple return, same basis

        # Equal-weight daily return: arithmetic mean of simple returns across symbols
        ew_daily_ret = univ_simple_ret.mean(axis=1)    # NaNs excluded automatically

        # Spread: EW minus SPY, rolled over 21 days
        daily_spread = ew_daily_ret - spy_simple_ret
        features["ew_spy_spread_21d"] = daily_spread.rolling(21, min_periods=10).mean()
        # ─────────────────────────────────────────────────────────────────────

        return pd.DataFrame(features, index=self.spy.index)

    # ── Group C: risk / stress features ─────────────────────────────────

    def _build_stress_features(self) -> pd.DataFrame:
        features: Dict[str, pd.Series] = {}

        if self.vix is not None:
            vix_series = self.vix.squeeze() if hasattr(self.vix, "squeeze") else self.vix
            if isinstance(vix_series, pd.DataFrame):
                vix_series = vix_series.iloc[:, 0]
            vix_aligned = vix_series.reindex(self.spy.index, method="ffill")
            features["vix_level"]   = vix_aligned
            features["vix_chg_5d"]  = vix_aligned.pct_change(5)
            features["vix_chg_21d"] = vix_aligned.pct_change(21)
            log.info("regime.features: VIX features included")
        else:
            log.info("regime.features: VIX not available — stress group uses SPY vol only")

        if self.macro is not None:
            macro_aligned = self.macro.reindex(self.spy.index, method="ffill")
            if "DGS10" in macro_aligned.columns and "DFF" in macro_aligned.columns:
                features["term_spread"] = macro_aligned["DGS10"] - macro_aligned["DFF"]
                log.info("regime.features: term spread included")
            for col in ["UNRATE", "CPIAUCSL", "T10Y2Y"]:
                if col in macro_aligned.columns:
                    features[f"macro_{col}"] = macro_aligned[col]

        return pd.DataFrame(features, index=self.spy.index)

    # ── Group D: internal strategy-state features ────────────────────────

    def _build_strategy_features(self) -> pd.DataFrame:
        features: Dict[str, pd.Series] = {}

        if self.alpha_scores is not None and self.fwd_returns is not None:
            try:
                ic          = _ic_series(self.alpha_scores, self.fwd_returns, window=21)
                ic_aligned  = ic.reindex(self.spy.index)
                features["strategy_ic_21d"]      = ic_aligned
                features["strategy_ic_positive"] = (ic_aligned > 0).astype(float)
                log.info("regime.features: strategy IC features included")
            except Exception as exc:
                log.warning(f"regime.features: strategy IC computation failed: {exc}")

        return pd.DataFrame(features, index=self.spy.index)

    # ── public API ───────────────────────────────────────────────────────

    def build(self) -> pd.DataFrame:
        """
        Build and return the full regime feature panel.

        Returns a DataFrame with DatetimeIndex (same as spy_prices)
        and one column per feature. NaNs are allowed — the model layer
        handles missing data through min_periods and imputation.
        """
        parts = [
            self._build_market_features(),
            self._build_cs_features(),
            self._build_stress_features(),
            self._build_strategy_features(),
        ]
        panel = pd.concat([p for p in parts if not p.empty], axis=1)
        panel = panel.reindex(self.spy.index)

        n_features = panel.shape[1]
        n_complete = panel.dropna().shape[0]
        log.info(
            f"regime.features: built {n_features} features, "
            f"{n_complete}/{len(panel)} rows fully complete"
        )
        return panel

    def get_core_features(self) -> pd.DataFrame:
        """
        Return only the minimal 'always-available' feature subset
        used when optional data is missing. Used as fallback in model fitting.
        """
        panel     = self.build()
        core_cols = [c for c in panel.columns if c.startswith("spy_")]
        return panel[core_cols]