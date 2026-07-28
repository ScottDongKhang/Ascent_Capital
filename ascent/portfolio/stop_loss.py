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

import json
import logging
from datetime import date, timedelta
from pathlib import Path

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

    Never raises: any unexpected internal failure is logged and reported as
    an all-False Series (no stops) — the fail-open direction required of a
    risk overlay, matching enforce_cluster_cap's never-raise contract.
    """
    try:
        return _compute_stop_breaches_impl(entry_prices, current_prices, threshold)
    except Exception as exc:
        log.warning(
            "[StopLoss] compute_stop_breaches failed unexpectedly (%s) — "
            "reporting no breaches (fail-open)", exc,
        )
        try:
            idx = entry_prices.index.union(current_prices.index)
        except Exception:
            idx = pd.Index([])
        return pd.Series(False, index=idx, dtype=bool)


def _compute_stop_breaches_impl(
    entry_prices: pd.Series,
    current_prices: pd.Series,
    threshold: float,
) -> pd.Series:
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

    Never raises: any unexpected internal failure is logged and `weights` is
    returned unchanged (fail-open), matching enforce_cluster_cap's
    never-raise contract.
    """
    try:
        return _apply_stop_loss_impl(weights, breached, redistribute)
    except Exception as exc:
        log.warning(
            "[StopLoss] apply_stop_loss failed unexpectedly (%s) — "
            "returning weights unchanged (fail-open)", exc,
        )
        return weights


def _apply_stop_loss_impl(
    weights: pd.Series,
    breached: pd.Series,
    redistribute: bool = False,
) -> pd.Series:
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


DEFAULT_STATE_PATH = "data_cache/stop_loss_state.json"


def load_stop_state(path: str = DEFAULT_STATE_PATH) -> dict:
    """
    Load {symbol: ISO date of last stop}. A missing or corrupt file yields
    an empty state — a broken state file must never block trading.
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"expected a dict, got {type(data).__name__}")
        return {str(k): str(v) for k, v in data.items()}
    except Exception as exc:
        log.warning("[StopLoss] Could not read stop state %s (%s) — treating "
                    "as empty", path, exc)
        return {}


def record_stops(symbols: list, today: str,
                 path: str = DEFAULT_STATE_PATH) -> dict:
    """
    Record `symbols` as stopped on `today` (ISO YYYY-MM-DD) and persist.
    Re-stopping a symbol refreshes its date. Returns the new state.
    """
    state = load_stop_state(path)
    for s in symbols:
        state[str(s)] = str(today)
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2, sort_keys=True))
    except Exception as exc:
        log.warning("[StopLoss] Could not persist stop state to %s: %s",
                    path, exc)
    return state


def blocked_symbols(state: dict, today: str,
                    cooldown_days: int = COOLDOWN_DAYS) -> set:
    """
    Symbols still inside their re-entry cooldown as of `today`.

    Cooldown is measured in CALENDAR days: bdate_range(end="today") returns
    empty on weekends (known repo gotcha) and a trading-calendar dependency
    buys nothing here. The boundary is exclusive — exactly `cooldown_days`
    after the stop, the symbol is tradeable again.

    An unparseable date does not block (fail-open): a corrupt entry must not
    freeze a symbol out of the book forever.

    NOTE: the cooldown-boundary rule ("< " not "<=", exclusive at exactly
    `cooldown_days` later) is intentionally re-implemented here AND in
    `apply_stop_loss_panel`'s internal `blocked_until` dict — this one on
    ISO strings via `date`/`timedelta` for the persisted live-trading state,
    that one on `pd.Timestamp`/`pd.Timedelta` for a single backtest pass with
    no disk state. If `COOLDOWN_DAYS` semantics ever change (e.g. to trading
    days, or to an inclusive boundary), update both.
    """
    if not state:
        return set()
    try:
        today_d = date.fromisoformat(str(today))
    except Exception as exc:
        log.warning("[StopLoss] Bad 'today' value %r (%s) — blocking nothing",
                    today, exc)
        return set()

    out = set()
    for sym, stopped_on in state.items():
        try:
            d = date.fromisoformat(str(stopped_on))
        except Exception:
            log.warning("[StopLoss] Unparseable stop date %r for %s — not "
                        "blocking", stopped_on, sym)
            continue
        if today_d < d + timedelta(days=int(cooldown_days)):
            out.add(sym)
    return out


def apply_stop_loss_panel(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    threshold: float = STOP_THRESHOLD,
    cooldown_days: int = COOLDOWN_DAYS,
    redistribute: bool = False,
) -> tuple:
    """
    Apply the stop-loss rule across a (dates x symbols) weights panel.

    Research counterpart to the production path: there is no broker to ask
    for entry prices, so they are reconstructed by walking the panel forward
    and recording the close on the day each name enters the book.

    Entry price is the close on the entry date. In the live path the fill is
    the next open, so research is a close-to-close approximation of the same
    rule — a deliberate, documented simplification for a risk overlay.

    threshold <= 0 disables the rule entirely (exact no-op).

    Returns (stopped_weights, events).

    Never raises: any unexpected internal failure is logged and `(weights,
    [])` is returned unchanged (fail-open), matching enforce_cluster_cap's
    never-raise contract.
    """
    try:
        return _apply_stop_loss_panel_impl(
            weights, close, threshold, cooldown_days, redistribute,
        )
    except Exception as exc:
        log.warning(
            "[StopLoss] apply_stop_loss_panel failed unexpectedly (%s) — "
            "returning weights unchanged, no events (fail-open)", exc,
        )
        return weights, []


def _apply_stop_loss_panel_impl(
    weights: pd.DataFrame,
    close: pd.DataFrame,
    threshold: float,
    cooldown_days: int,
    redistribute: bool,
) -> tuple:
    if weights is None or weights.empty or threshold <= 0:
        return (weights.astype(float) if weights is not None
                and not weights.empty else weights), []

    out = weights.astype(float).copy()
    px = close.reindex(index=out.index, columns=out.columns)

    entry: dict = {}          # symbol -> entry price
    # NOTE: `blocked_until` re-implements the cooldown-boundary rule from
    # `blocked_symbols` above (exclusive at exactly `cooldown_days` later),
    # but on pd.Timestamp/pd.Timedelta for a single backtest pass rather than
    # ISO strings against persisted state. Keep both in sync if COOLDOWN_DAYS
    # semantics ever change.
    blocked_until: dict = {}  # symbol -> pd.Timestamp
    events: list = []

    for dt in out.index:
        row = out.loc[dt]
        held = [s for s in out.columns if float(row.get(s, 0.0)) > 0.0]

        # 1. Names still inside their cooldown cannot be re-entered. This
        #    mutates out.loc[dt, s] to 0.0, so `row`/`held` below must be
        #    recomputed from the post-mutation state before step 2 decides
        #    which names are still "held" — the duplication is load-bearing
        #    for that ordering, not accidental repetition.
        for s in held:
            until = blocked_until.get(s)
            if until is not None and dt < until:
                out.loc[dt, s] = 0.0
        row = out.loc[dt]
        held = [s for s in out.columns if float(row.get(s, 0.0)) > 0.0]

        # 2. Names that left the book (by the strategy or by cooldown) lose
        #    their recorded entry, so a later re-entry prices fresh.
        for s in list(entry):
            if s not in held:
                entry.pop(s, None)

        # 3. New entries record their price and cannot breach on day one.
        fresh = []
        for s in held:
            if s not in entry:
                p = px.at[dt, s] if s in px.columns else np.nan
                if pd.notna(p) and float(p) > 0:
                    entry[s] = float(p)
                    fresh.append(s)

        # 4. Evaluate breaches for names held since a prior date.
        seasoned = [s for s in held if s in entry and s not in fresh]
        if seasoned:
            breached = compute_stop_breaches(
                pd.Series({s: entry[s] for s in seasoned}),
                pd.Series({s: px.at[dt, s] for s in seasoned}),
                threshold=threshold,
            )
            hits = [s for s in seasoned if bool(breached.get(s, False))]
            if hits:
                for s in hits:
                    exit_px = float(px.at[dt, s])
                    events.append({
                        "date":           str(dt),
                        "symbol":         s,
                        "entry_price":    entry[s],
                        "exit_price":     exit_px,
                        "pct_from_entry": exit_px / entry[s] - 1.0,
                    })
                    blocked_until[s] = dt + pd.Timedelta(days=int(cooldown_days))
                    entry.pop(s, None)
                out.loc[dt] = apply_stop_loss(
                    out.loc[dt],
                    pd.Series(True, index=hits),
                    redistribute=redistribute,
                )

    return out, events
