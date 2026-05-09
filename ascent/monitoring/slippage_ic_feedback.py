"""
ascent/monitoring/slippage_ic_feedback.py

Slippage-adjusted IC feedback loop.

Weekly: reads slippage_log.jsonl + agent PnL logs, computes gross IC
vs net-of-slippage IC, writes drag coefficient to active_alpha_config.json.

This module is a passive logger until ~60 fills accumulate (MIN_FILLS=50).
Self-improve integration is a future TODO — do not wire the drag coefficient
into self-improve scoring until sufficient data exists.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

log = logging.getLogger(__name__)

SLIPPAGE_LOG = Path("logs/slippage_log.jsonl")
PNL_LOGS     = {
    "us_equities":   Path("logs/us_equities_pnl.jsonl"),
    "macro":         Path("logs/macro_pnl.jsonl"),
    "international": Path("logs/international_pnl.jsonl"),
    "alternatives":  Path("logs/alternatives_pnl.jsonl"),
}
ACTIVE_CONFIG_PATH = Path("data_cache/active_alpha_config.json")
MIN_FILLS = 50
# NOTE: Do not wire slippage_ic_drag into self-improve scoring until >= 60 fills
# have accumulated (expected ~July 2026). With fewer fills the Spearman IC estimate
# has SE ~0.3, making the drag coefficient noise rather than signal.


def _load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def compute_slippage_ic_drag(lookback_days: int = 90) -> Dict[str, float]:
    """
    Compute slippage IC drag over recent fills.

    For each fill: compute gross 10-day forward return and net return
    (gross minus realized slippage). Compare Spearman IC of signal
    vs gross returns to IC vs net returns. The difference is the drag.

    Returns dict with slippage_ic_drag, gross_ic, net_ic, n_fills, mean_slippage_bps.
    """
    slippage_rows = _load_jsonl(SLIPPAGE_LOG)
    if len(slippage_rows) < MIN_FILLS:
        log.info("[SlippageIC] Insufficient fills (%d < %d), skipping",
                 len(slippage_rows), MIN_FILLS)
        return {"slippage_ic_drag": 0.0, "gross_ic": 0.0, "net_ic": 0.0,
                "n_fills": len(slippage_rows), "mean_slippage_bps": 0.0}

    # Build symbol→date→return lookup from PnL logs
    fwd_lookup: Dict[str, Dict[str, float]] = {}
    for path in PNL_LOGS.values():
        for row in _load_jsonl(path):
            sym = row.get("symbol") or row.get("ticker")
            dt  = row.get("date")
            ret = row.get("daily_return") or row.get("return")
            if sym and dt and ret is not None:
                fwd_lookup.setdefault(sym, {})[dt] = float(ret)

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=lookback_days)
    signals, gross_fwds, net_fwds, slippages_bps = [], [], [], []

    for row in slippage_rows:
        try:
            dt  = pd.Timestamp(row["date"])
            sym = row.get("symbol", "")
            slip_bps      = float(row.get("slippage_bps", 0.0))
            signal_price  = float(row.get("signal_price", 1.0))
            fill_price    = float(row.get("fill_price", 1.0))
        except (KeyError, ValueError, TypeError):
            continue

        if dt < cutoff or sym not in fwd_lookup:
            continue

        sym_rets = {pd.Timestamp(d): r for d, r in fwd_lookup[sym].items()}
        future   = sorted(d for d in sym_rets if d > dt)[:10]
        if len(future) < 5:
            continue

        fwd_return   = sum(sym_rets[d] for d in future)
        slip_return  = slip_bps / 10_000
        signal_score = (signal_price - fill_price) / max(signal_price, 1e-8)

        signals.append(signal_score)
        gross_fwds.append(fwd_return)
        net_fwds.append(fwd_return - slip_return)
        slippages_bps.append(slip_bps)

    if len(signals) < MIN_FILLS:
        return {"slippage_ic_drag": 0.0, "gross_ic": 0.0, "net_ic": 0.0,
                "n_fills": len(signals), "mean_slippage_bps": 0.0}

    gross_ic, _ = spearmanr(signals, gross_fwds)
    net_ic,   _ = spearmanr(signals, net_fwds)
    gross_ic    = float(gross_ic) if not np.isnan(gross_ic) else 0.0
    net_ic      = float(net_ic)   if not np.isnan(net_ic)   else 0.0
    drag        = (gross_ic - net_ic) / max(abs(gross_ic), 1e-6)

    return {
        "slippage_ic_drag":  round(drag, 4),
        "gross_ic":          round(gross_ic, 4),
        "net_ic":            round(net_ic, 4),
        "n_fills":           len(signals),
        "mean_slippage_bps": round(float(np.mean(slippages_bps)), 2),
    }


def update_active_config_with_slippage_feedback(metrics: Dict[str, float]) -> None:
    """Write slippage IC drag to active_alpha_config.json slippage_feedback section."""
    config = {}
    if ACTIVE_CONFIG_PATH.exists():
        try:
            config = json.loads(ACTIVE_CONFIG_PATH.read_text())
        except Exception:
            pass
    config["slippage_feedback"] = {
        **{k: metrics.get(k, 0.0) for k in
           ["slippage_ic_drag", "gross_ic", "net_ic", "n_fills", "mean_slippage_bps"]},
        "updated_at": str(pd.Timestamp.today().date()),
    }
    ACTIVE_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_CONFIG_PATH.write_text(json.dumps(config, indent=2))
    log.info("[SlippageIC] drag=%.4f gross_ic=%.4f net_ic=%.4f fills=%d",
             metrics.get("slippage_ic_drag", 0), metrics.get("gross_ic", 0),
             metrics.get("net_ic", 0), metrics.get("n_fills", 0))


def run_slippage_ic_feedback(lookback_days: int = 90) -> Dict[str, float]:
    """Top-level entry point called from run_all_agents.py on Sundays."""
    metrics = compute_slippage_ic_drag(lookback_days=lookback_days)
    if metrics["n_fills"] >= MIN_FILLS:
        update_active_config_with_slippage_feedback(metrics)
    return metrics
