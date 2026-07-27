# ascent/portfolio/stop_loss.py
"""
Position-level stop-loss — single source of truth.

Implements Han, Zhou & Zhu (2014), "Taming Momentum Crashes: A Simple
Stop-Loss Strategy": exit a position that has fallen more than `threshold`
below its entry price, and block re-entry for a cooldown window.

Both the production path (run_all_agents.py, entry prices from the live
Alpaca book) and the walk-forward framework (ascent/research/wf_framework/
ascent_strategy.py, entry prices reconstructed from the price panel) MUST
go through this module. See ascent/portfolio/exposure.py for the precedent:
research and production previously carried separate overlay implementations
and silently diverged.

Design notes:
  * Stopped weight goes to CASH by default. Redistributing into the
    remaining book re-risks into the same factor that just broke.
  * The stop must be applied LAST, after every cap and overlay, because
    _water_fill_cap / enforce_cluster_cap / enforce_risk_budget_cap /
    apply_exposure_overlays all renormalize and would refill the name.
  * Missing data never triggers a stop (fail-open), matching
    enforce_cluster_cap's never-raise contract.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

STOP_THRESHOLD = 0.10   # Han, Zhou & Zhu (2014) headline stop level
COOLDOWN_DAYS  = 30     # calendar days, ~21 trading days


def compute_stop_breaches(
    entry_prices: pd.Series,
    current_prices: pd.Series,
    threshold: float = STOP_THRESHOLD,
) -> pd.Series:
    """
    Boolean Series (indexed by symbol) marking positions that have fallen
    `threshold` or more below their entry price.

    A breach requires a positive entry price and a positive current price.
    Anything unresolvable (missing on either side, non-positive, NaN) is
    reported as NOT breached and logged — an unknown price must never force
    a liquidation.

    The comparison is inclusive: exactly -threshold breaches.
    """
    idx = entry_prices.index.union(current_prices.index)
    if len(idx) == 0:
        return pd.Series(dtype=bool)

    entry = pd.to_numeric(entry_prices.reindex(idx), errors="coerce")
    now   = pd.to_numeric(current_prices.reindex(idx), errors="coerce")

    resolvable = entry.notna() & now.notna() & (entry > 0) & (now > 0)
    unresolved = list(idx[~resolvable])
    if unresolved:
        log.warning(
            "[StopLoss] Cannot evaluate stop for %s (missing/invalid entry or "
            "current price) — treating as NOT breached", unresolved,
        )

    pct = pd.Series(np.nan, index=idx, dtype=float)
    pct.loc[resolvable] = now.loc[resolvable] / entry.loc[resolvable] - 1.0

    # Inclusive at the threshold; 1e-12 absorbs float representation error.
    breached = resolvable & (pct <= -abs(threshold) + 1e-12)
    return breached.fillna(False).astype(bool)


def apply_stop_loss(
    weights: pd.Series,
    breached: pd.Series,
    redistribute: bool = False,
) -> pd.Series:
    """
    Zero out breached names.

    redistribute=False (default): freed weight becomes cash, gross exposure
    falls. This is the actual risk reduction and the faithful reading of the
    paper.

    redistribute=True: freed weight is spread pro-rata across survivors,
    preserving gross. Provided for research comparison only.
    """
    if weights is None or len(weights) == 0:
        return weights

    w = weights.astype(float).copy()
    mask = breached.reindex(w.index).fillna(False).astype(bool)
    if not mask.any():
        return w

    freed = float(w[mask].sum())
    out = w.copy()
    out[mask] = 0.0

    if redistribute:
        survivors = ~mask
        surv_total = float(out[survivors].sum())
        if surv_total > 0:
            out[survivors] = out[survivors] / surv_total * (surv_total + freed)
        # else: every name breached — everything is already cash, nothing to
        # redistribute into. Returning all-zeros is correct.

    log.info(
        "[StopLoss] Stopped %d position(s) %s — %.4f of gross moved to %s",
        int(mask.sum()), list(w.index[mask]), freed,
        "survivors" if redistribute else "cash",
    )
    return out
