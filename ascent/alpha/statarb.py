"""
Ascent Capital — Alpha: Sector-Relative Statistical Arbitrage
Cross-sectional residual mean-reversion factor.

Formula (per date, per symbol):
  residual_Nd = mom_Nd(symbol) - mean(mom_Nd) over same-sector symbols
  signal_Nd   = -residual_Nd / vol_adj          (reversal: losers score +)
  composite   = 0.50 * signal_5d + 0.30 * signal_10d + 0.20 * signal_21d
  final       = cs_normalize(composite) * liquidity_weight

No look-ahead bias: all inputs are past returns computed at time t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────

def _cs_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score, clipped to [-3, 3]."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0)


def _winsorize(df: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Winsorize at ±clip cross-sectional standard deviations per row."""
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    lo = mean - clip * std
    hi = mean + clip * std
    return df.clip(lower=lo, upper=hi, axis=0)


def _sector_residual(
    mom: pd.DataFrame,
    sector_map: dict[str, str] | None,
) -> pd.DataFrame:
    """
    For each date and symbol, subtract the equal-weight sector mean return.
    Symbols with no sector mapping use the universe mean (graceful fallback).
    """
    if not sector_map:
        # No sector info — subtract universe mean (degrades gracefully)
        universe_mean = mom.mean(axis=1)
        return mom.sub(universe_mean, axis=0)

    residual = mom.copy()

    # Group columns by sector
    sectors: dict[str, list[str]] = {}
    for sym in mom.columns:
        sec = sector_map.get(sym, "__unknown__")
        sectors.setdefault(sec, []).append(sym)

    for sec, syms in sectors.items():
        available = [s for s in syms if s in mom.columns]
        if not available:
            continue
        sec_mean = mom[available].mean(axis=1)          # equal-weight sector mean
        residual[available] = mom[available].sub(sec_mean, axis=0)

    return residual


def _derive_mom(
    features: dict[str, pd.DataFrame],
    window: int,
) -> pd.DataFrame | None:
    """
    Return mom_Nd if present; attempt to derive from close prices if not.
    Returns None if unavailable.
    """
    key = f"mom_{window}d"
    if key in features:
        return features[key].copy()

    # Try to derive from a longer momentum via differencing (approximate)
    # e.g. mom_10d ≈ mom_21d - mom_5d is not exact; better to just skip.
    # We do attempt one level of derivation: if close is stored.
    if "close" in features:
        close = features["close"]
        derived = close.pct_change(window)
        if not derived.empty:
            return derived

    return None


# ── public API ─────────────────────────────────────────────────────────

def statarb_alpha(
    features: dict[str, pd.DataFrame],
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Sector-relative residual mean-reversion alpha.

    Args:
        features:   Dict of feature DataFrames (dates × symbols).
                    Expected keys (all optional but at least one must exist):
                      mom_5d, mom_10d, mom_21d   – past-N-day returns
                      vol_21d or vol_10d          – annualised rolling vol
                      dollar_vol_rank_21d         – liquidity rank [0,1]
        sector_map: {symbol: sector_name}. Partial coverage is fine;
                    unknown symbols fall back to universe mean.

    Returns:
        DataFrame(dates × symbols) with composite stat-arb alpha scores,
        or empty DataFrame if no momentum features are available.
    """
    HORIZONS = [
        (5,  0.50),
        (10, 0.30),
        (21, 0.20),
    ]

    # ── vol adjustment ────────────────────────────────────────────────
    if "vol_21d" in features:
        vol_adj = features["vol_21d"].copy().replace(0, np.nan)
    elif "vol_10d" in features:
        vol_adj = features["vol_10d"].copy().replace(0, np.nan)
    else:
        vol_adj = None  # skip vol-scaling; still produce signal

    # ── build per-horizon residual reversal signals ───────────────────
    components: list[tuple[pd.DataFrame, float]] = []

    for window, weight in HORIZONS:
        mom = _derive_mom(features, window)
        if mom is None or mom.empty:
            continue

        mom = _winsorize(mom, clip=3.0)

        # Sector-relative residual
        resid = _sector_residual(mom, sector_map)

        # Vol-adjust to avoid rewarding high-vol collapses
        if vol_adj is not None:
            # Align index/columns defensively
            v = vol_adj.reindex(index=resid.index, columns=resid.columns)
            # Use trailing vol from previous row (no look-ahead)
            v_lagged = v.shift(1).ffill()
            resid = resid.div(v_lagged).replace([np.inf, -np.inf], np.nan)

        # Reversal: flip sign so underperformers score positively
        signal = -resid

        # Winsorize again after division (vol can be tiny)
        signal = _winsorize(signal, clip=3.0).fillna(0)

        components.append((signal, weight))

    if not components:
        return pd.DataFrame()

    # ── blend horizons ────────────────────────────────────────────────
    total_w = sum(w for _, w in components)
    composite: pd.DataFrame | None = None
    for sig, w in components:
        scaled = sig * (w / total_w)
        if composite is None:
            composite = scaled
        else:
            idx = composite.index.intersection(sig.index)
            cols = composite.columns.intersection(sig.columns)
            composite = composite.reindex(index=idx, columns=cols)
            composite = composite + scaled.reindex(index=idx, columns=cols)

    if composite is None or composite.empty:
        return pd.DataFrame()

    # ── liquidity gate ────────────────────────────────────────────────
    # Scale signal by dollar_vol_rank so illiquid names are dampened.
    if "dollar_vol_rank_21d" in features:
        liq = features["dollar_vol_rank_21d"].reindex(
            index=composite.index, columns=composite.columns
        ).fillna(0.5)                       # unknown → neutral
        # Soft multiplier: rank in [0,1] → multiplier in [0.5, 1.0]
        liq_mult = 0.5 + 0.5 * liq
        composite = composite * liq_mult

    # ── final cross-sectional normalization ───────────────────────────
    composite = _cs_normalize(composite)

    return composite