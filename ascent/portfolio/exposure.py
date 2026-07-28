# ascent/portfolio/exposure.py
"""
Portfolio-level exposure overlays — single source of truth.

Both the production pipeline (ascent/main.py) and the walk-forward framework
(ascent/research/wf_framework/ascent_strategy.py) MUST apply exposure overlays
through this module. Research and production previously carried separate
implementations and silently diverged (research had vol targeting, production
didn't; production had a VIX gate on the 200MA cut, research didn't).

Three overlays, applied in order:
  1. ma_filter_scale()      — SPY < 200d MA (VIX-confirmed when VIX available) → ×0.70
  2. vol_target_scale()     — scale toward target portfolio vol using trailing SPY vol
     (or the strategy's own realized vol — see `vol_reference`)
  3. momentum_crash_scale() — 2y decline + rebound (Daniel & Moskowitz) → ×0.50
     (disabled by default)

All computations are causal: only data strictly before (vol) or up to (MA)
each decision date is used.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MA_WINDOW          = 200
MA_MIN_PERIODS     = 150
MA_CUT_MULTIPLIER  = 0.70
VOL_TARGET         = 0.15
VOL_LOOKBACK       = 21
VOL_FLOOR          = 0.25
VOL_CAP            = 1.00


def _vix_threshold() -> float:
    try:
        from ascent.regime.engine import VIX_STRESSED_CONFIRMATION
        return float(VIX_STRESSED_CONFIRMATION)
    except Exception:
        return 20.0


def ma_filter_scale(
    spy_close: pd.Series,
    dates: pd.Index,
    vix_close: Optional[pd.Series] = None,
    ma_window: int = MA_WINDOW,
    multiplier: float = MA_CUT_MULTIPLIER,
) -> pd.Series:
    """
    Per-date exposure multiplier from the SPY 200MA filter.

    Returns a Series indexed by `dates`: `multiplier` where SPY is below its
    MA AND (VIX > threshold when VIX data is available), else 1.0.
    When vix_close is None the cut fires on the MA alone (legacy behavior);
    missing VIX values within a provided series default to 25.0 (cut allowed),
    matching the production pipeline.
    """
    spy_close = spy_close.sort_index()
    spy_close = spy_close[~spy_close.index.duplicated(keep="last")]
    ma = spy_close.rolling(ma_window, min_periods=MA_MIN_PERIODS).mean()
    below = spy_close < ma

    if vix_close is not None and len(vix_close) > 0:
        vix_aligned = (
            vix_close.sort_index()
            .reindex(below.index, method="ffill")
            .fillna(25.0)
        )
        below = below & vix_aligned.gt(_vix_threshold())

    aligned = below.reindex(dates, method="ffill").fillna(False)
    return pd.Series(np.where(aligned, multiplier, 1.0), index=dates)


def realized_vol_scale(
    returns: pd.Series,
    dates: pd.Index,
    target_vol: float = VOL_TARGET,
    lookback: int = VOL_LOOKBACK,
    floor: float = VOL_FLOOR,
    cap: float = VOL_CAP,
) -> pd.Series:
    """
    Per-date exposure multiplier targeting `target_vol` annualized against an
    arbitrary daily return series.

    scale(d) = clip(target_vol / realized_vol(d), floor, cap), where
    realized_vol(d) uses returns strictly before d (fully causal).
    Dates with <5 trailing observations get scale 1.0.

    Barroso & Santa-Clara (2015) and Moreira & Muir (2017) both scale a
    factor by ITS OWN trailing realized volatility. Passing SPY returns here
    reproduces the legacy market-referenced behaviour; passing the strategy's
    own returns implements the papers.
    """
    if len(dates) == 0:
        return pd.Series(dtype=float)

    rets = returns.sort_index().dropna()

    scales = []
    for d in dates:
        past = rets[rets.index < d].iloc[-lookback:]
        if len(past) < 5:
            scales.append(1.0)
            continue
        realized = float(past.std() * np.sqrt(252))
        if realized < 1e-6:
            scales.append(1.0)
            continue
        scales.append(float(np.clip(target_vol / realized, floor, cap)))
    return pd.Series(scales, index=dates)


def vol_target_scale(
    spy_close: pd.Series,
    dates: pd.Index,
    target_vol: float = VOL_TARGET,
    lookback: int = VOL_LOOKBACK,
    floor: float = VOL_FLOOR,
    cap: float = VOL_CAP,
) -> pd.Series:
    """
    Market-referenced vol targeting: `realized_vol_scale` over SPY returns.

    Retained with its original signature so existing callers and tests are
    unaffected. New code should prefer `realized_vol_scale` with the
    strategy's own return series — see `strategy_return_proxy`.
    """
    spy_close = spy_close.sort_index()
    spy_close = spy_close[~spy_close.index.duplicated(keep="last")]
    return realized_vol_scale(
        spy_close.pct_change().dropna(), dates,
        target_vol=target_vol, lookback=lookback, floor=floor, cap=cap,
    )


def strategy_return_proxy(
    weights: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.Series:
    """
    Daily return series of the book: r(t) = sum_i w_i(t-1) * ret_i(t).

    Causal by construction — yesterday's weights against today's returns.
    Used as the reference series for strategy-own volatility targeting
    (Barroso & Santa-Clara 2015, Moreira & Muir 2017), because the realized
    return series is not available inside the strategy before it runs.

    Symbols present in `weights` but absent from `close` contribute zero
    rather than NaN, so one missing ticker cannot blank the whole series.
    """
    if weights is None or weights.empty or close is None or close.empty:
        return pd.Series(dtype=float)

    cols = [c for c in weights.columns if c in close.columns]
    if not cols:
        return pd.Series(0.0, index=weights.index)

    w = weights[cols].astype(float).fillna(0.0)
    rets = close[cols].reindex(w.index).pct_change().fillna(0.0)

    # shift(1): weights known at the close of t-1 earn t's return.
    return (w.shift(1).fillna(0.0) * rets).sum(axis=1)


CRASH_BEAR_LOOKBACK    = 504   # ~2 trading years
CRASH_REBOUND_LOOKBACK = 21    # ~1 trading month
CRASH_MULTIPLIER       = 0.50


def momentum_crash_scale(
    spy_close: pd.Series,
    dates: pd.Index,
    bear_lookback: int = CRASH_BEAR_LOOKBACK,
    rebound_lookback: int = CRASH_REBOUND_LOOKBACK,
    multiplier: float = CRASH_MULTIPLIER,
) -> pd.Series:
    """
    Per-date exposure multiplier for the momentum-crash state.

    Daniel & Moskowitz (2016): momentum crashes cluster in panic periods —
    a prolonged market decline followed by a rebound. Winners carry beta
    acquired during the decline, and the rebound repriced them violently.

    Fires `multiplier` when BOTH hold, using only data strictly before `d`:
      * bear:    cumulative SPY return over the trailing `bear_lookback` < 0
      * rebound: SPY return over the trailing `rebound_lookback` > 0
    Otherwise 1.0.

    Deliberately NOT the full dynamic model. Daniel & Moskowitz scale
    continuously by a forecast conditional Sharpe ratio; estimating that on
    this book's history is an overfitting risk. This is the low-parameter
    state indicator at the core of the same result.

    Distinct from the 200MA cut, which fires on "market is below trend".
    This fires on "market fell for a long time and is now bouncing" — the
    200MA filter is typically OFF in exactly that state, which is the point.
    """
    if len(dates) == 0:
        return pd.Series(dtype=float)
    if multiplier >= 1.0:
        return pd.Series(1.0, index=dates)

    try:
        s = spy_close.sort_index()
        s = s[~s.index.duplicated(keep="last")].dropna()
    except Exception as exc:
        log.warning("[Exposure] momentum_crash_scale: bad SPY series (%s) "
                    "— no cut applied", exc)
        return pd.Series(1.0, index=dates)

    scales = []
    for d in dates:
        past = s[s.index < d]
        if len(past) < bear_lookback + 1:
            scales.append(1.0)
            continue

        bear_window = past.iloc[-(bear_lookback + 1):]
        bear_ret = float(bear_window.iloc[-1] / bear_window.iloc[0] - 1.0)

        reb_window = past.iloc[-(rebound_lookback + 1):]
        reb_ret = float(reb_window.iloc[-1] / reb_window.iloc[0] - 1.0)

        scales.append(float(multiplier) if (bear_ret < 0.0 and reb_ret > 0.0)
                      else 1.0)

    out = pd.Series(scales, index=dates)
    n_cut = int((out < 1.0).sum())
    if n_cut:
        log.info("[Exposure] Momentum-crash state on %d/%d dates — exposure "
                 "x%.2f (Daniel & Moskowitz 2016)", n_cut, len(out), multiplier)
    return out


def apply_exposure_overlays(
    weights: pd.DataFrame,
    spy_close: pd.Series,
    vix_close: Optional[pd.Series] = None,
    rebalance_only: bool = True,
    target_vol: float = VOL_TARGET,
    vol_floor: float = VOL_FLOOR,
    vol_cap: float = VOL_CAP,
    ma_multiplier: float = MA_CUT_MULTIPLIER,
    vol_targeting_enabled: bool = True,
    vol_reference: str = "spy",
    close: Optional[pd.DataFrame] = None,
    crash_overlay_enabled: bool = False,
    crash_multiplier: float = CRASH_MULTIPLIER,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply MA filter then vol targeting to a (dates × symbols) weights frame.

    With rebalance_only=True (default) the combined scale is computed on dates
    where the underlying weights change (rebalance rows) and held constant
    between them — matching live behavior, where exposure is set at rebalance
    and not adjusted daily. Returns (scaled_weights, meta).

    `vol_reference` selects the series vol targeting is measured against:
    "spy" (market-referenced, legacy default) or "strategy" (the book's own
    trailing realized vol — Barroso & Santa-Clara 2015, Moreira & Muir 2017),
    which requires a `close` price panel. Unknown values and a missing panel
    both fall back to "spy" with a warning; this never raises.
    """
    ref = str(vol_reference or "spy").lower()
    if ref not in ("spy", "strategy"):
        log.warning("[Exposure] Unknown vol_reference %r — using 'spy'",
                    vol_reference)
        ref = "spy"
    if ref == "strategy" and (close is None or close.empty):
        log.warning("[Exposure] vol_reference='strategy' needs a close panel "
                    "— falling back to 'spy'")
        ref = "spy"

    if weights.empty:
        return weights, {"ma_cut_dates": 0, "mean_vol_scale": 1.0,
                         "vol_reference": ref, "crash_cut_dates": 0}

    dates = weights.index

    ma_scale = ma_filter_scale(spy_close, dates, vix_close=vix_close,
                               multiplier=ma_multiplier)
    if vol_targeting_enabled:
        if ref == "strategy":
            strat_rets = strategy_return_proxy(weights, close)
            v_scale = realized_vol_scale(strat_rets, dates,
                                         target_vol=target_vol,
                                         floor=vol_floor, cap=vol_cap)
        else:
            v_scale = vol_target_scale(spy_close, dates, target_vol=target_vol,
                                       floor=vol_floor, cap=vol_cap)
    else:
        v_scale = pd.Series(1.0, index=dates)

    combined = ma_scale * v_scale

    if crash_overlay_enabled:
        c_scale = momentum_crash_scale(spy_close, dates,
                                       multiplier=crash_multiplier)
    else:
        c_scale = pd.Series(1.0, index=dates)
    combined = combined * c_scale

    if rebalance_only and len(weights) > 1:
        changed = weights.diff().abs().sum(axis=1) > 1e-12
        changed.iloc[0] = True
        combined = combined.where(changed).ffill().fillna(1.0)

    scaled = weights.mul(combined, axis=0)
    meta = {
        "ma_cut_dates": int((ma_scale < 1.0).sum()),
        "mean_vol_scale": round(float(v_scale.mean()), 4),
        "min_vol_scale": round(float(v_scale.min()), 4),
        "vol_reference": ref,
        "crash_cut_dates": int((c_scale < 1.0).sum()),
    }
    return scaled, meta


def load_vix_from_macro_cache() -> Optional[pd.Series]:
    """
    Load a VIX close series (indexed by date) from the macro cache.
    Returns None when unavailable. Used by the WF framework so research
    applies the same VIX-confirmed MA cut as production.
    """
    try:
        from ascent.data.store.parquet import load_parquet, has_data
        for cache in ("macro_live", "macro_simulated"):
            if not has_data(cache):
                continue
            df = load_parquet(cache)
            if not {"series_id", "date", "value"}.issubset(df.columns):
                continue
            vix = df[df["series_id"].isin(["VIXCLS", "vix"])].copy()
            if vix.empty:
                continue
            vix["date"] = pd.to_datetime(vix["date"])
            if getattr(vix["date"].dt, "tz", None) is not None:
                vix["date"] = vix["date"].dt.tz_localize(None)
            s = (vix.sort_values("date")
                    .drop_duplicates("date", keep="last")
                    .set_index("date")["value"]
                    .astype(float))
            return s
    except Exception as exc:
        log.debug("[Exposure] VIX load from macro cache failed: %s", exc)
    return None
