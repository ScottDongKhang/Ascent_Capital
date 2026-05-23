# ascent/monitoring/position_health.py
"""
Pure-Python position health metrics for live holdings.
No LLM — just price data and alpha signals.

Computes per-position:
  - return since last rebalance (cumulative close-to-close)
  - current 252d / 21d momentum
  - alpha rank percentile in universe today (0=top, 1=bottom)
  - flag: OK / WATCH / DETERIORATING

Used by adversarial_daily and position_thesis to give LLM monitors real numbers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_WEDGE_LOG = Path("logs/alpha_wedge.jsonl")


@dataclass
class PositionHealth:
    symbol: str
    weight: float
    return_since_rebalance: Optional[float]   # cumulative since last rebalance date
    momentum_252d: Optional[float]
    momentum_21d: Optional[float]
    alpha_rank_pct: Optional[float]           # 0.0 = top of universe, 1.0 = bottom
    flag: str                                 # "OK" | "WATCH" | "DETERIORATING"

    def as_prompt_line(self) -> str:
        ret_str  = f"{self.return_since_rebalance:+.1%}" if self.return_since_rebalance is not None else "n/a"
        mom_str  = f"{self.momentum_252d:+.0%}" if self.momentum_252d is not None else "n/a"
        rank_str = f"{self.alpha_rank_pct:.0%}" if self.alpha_rank_pct is not None else "n/a"
        return (
            f"{self.symbol} ({self.weight:.1%}) [{self.flag}]"
            f"  ret_since_rebal={ret_str}"
            f"  mom252d={mom_str}"
            f"  alpha_rank={rank_str} in universe"
        )


def _last_rebalance_date() -> Optional[str]:
    """Read last rebalance date from alpha_wedge.jsonl, fallback to merged_weights.json."""
    try:
        if _WEDGE_LOG.exists():
            rows = [json.loads(l) for l in _WEDGE_LOG.read_text().splitlines() if l.strip()]
            if rows:
                return rows[-1]["rebalance_date"]
    except Exception:
        pass
    try:
        mw = Path("execution/merged_weights.json")
        if mw.exists():
            return json.loads(mw.read_text()).get("date")
    except Exception:
        pass
    return None


def _load_prices_wide():
    """Load prices_live parquet and pivot to wide format. Returns None on failure."""
    try:
        from ascent.data.store.parquet import load_parquet
        df = load_parquet("prices_live")
        if df is None or df.empty:
            return None
        return df.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    except Exception as e:
        log.warning("[PositionHealth] Price load failed: %s", e)
        return None


def _alpha_rank_map(agent_outputs: list) -> Dict[str, float]:
    """
    Return {symbol: rank_percentile} for every symbol in today's US equities alpha scores.
    0.0 = top of universe, 1.0 = bottom.
    """
    us = next((ao for ao in agent_outputs if ao.agent_id == "us_equities"), None)
    if us is None or us.alpha_scores is None or us.alpha_scores.empty:
        return {}
    try:
        latest = us.alpha_scores.iloc[-1].dropna()
        # ascending=True: low alpha score → low rank → high rank_pct (bottom of universe)
        # 1.0 - pct: best alpha → rank_pct ≈ 0.0 (top), worst → rank_pct ≈ 1.0 (bottom)
        ranked = latest.rank(ascending=True, pct=True)
        return {str(sym): float(1.0 - pct) for sym, pct in ranked.items()}
    except Exception as e:
        log.warning("[PositionHealth] Alpha rank computation failed: %s", e)
        return {}


def compute_position_health(
    date_str: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
) -> Dict[str, PositionHealth]:
    """
    Compute health metrics for every live position.
    Returns {} on complete failure — never blocks the daily run.
    """
    if not merged_weights:
        return {}

    prices = _load_prices_wide()
    rank_map = _alpha_rank_map(agent_outputs)
    last_rb = _last_rebalance_date()

    result: Dict[str, PositionHealth] = {}

    for sym, weight in merged_weights.items():
        ret_since_rb: Optional[float] = None
        mom_252d: Optional[float] = None
        mom_21d: Optional[float] = None
        rank_pct: Optional[float] = rank_map.get(sym)

        if prices is not None and sym in prices.columns:
            col = prices[sym].dropna()
            if not col.empty:
                # Return since last rebalance
                if last_rb:
                    try:
                        rb = date.fromisoformat(last_rb)
                        rb_prices = col[col.index.date >= rb]
                        if not rb_prices.empty and len(rb_prices) >= 2:
                            p0 = float(rb_prices.iloc[0])
                            p1 = float(rb_prices.iloc[-1])
                            if p0 != 0:
                                ret_since_rb = (p1 - p0) / p0
                    except Exception:
                        pass

                # 252d momentum (skip last 21d — standard skip-month)
                if len(col) >= 252:
                    try:
                        p_252 = float(col.iloc[-252])
                        p_21  = float(col.iloc[-22]) if len(col) >= 22 else float(col.iloc[-1])
                        if p_252 != 0:
                            mom_252d = (p_21 - p_252) / p_252
                    except Exception:
                        pass

                # 21d momentum
                if len(col) >= 22:
                    try:
                        p0 = float(col.iloc[-22])
                        p1 = float(col.iloc[-1])
                        if p0 != 0:
                            mom_21d = (p1 - p0) / p0
                    except Exception:
                        pass

        # Flag logic
        flag = "OK"
        deteriorating = (
            (rank_pct is not None and rank_pct > 0.60) and
            (ret_since_rb is not None and ret_since_rb < -0.03)
        )
        watch = (
            (rank_pct is not None and rank_pct > 0.45) or
            (ret_since_rb is not None and ret_since_rb < -0.02) or
            (mom_252d is not None and mom_252d < 0.05)
        )
        if deteriorating:
            flag = "DETERIORATING"
        elif watch:
            flag = "WATCH"

        result[sym] = PositionHealth(
            symbol=sym,
            weight=weight,
            return_since_rebalance=ret_since_rb,
            momentum_252d=mom_252d,
            momentum_21d=mom_21d,
            alpha_rank_pct=rank_pct,
            flag=flag,
        )

    return result


def format_health_summary(health: Dict[str, PositionHealth]) -> str:
    """One-line summary for each position, sorted worst-first."""
    if not health:
        return "No position health data."
    order = {"DETERIORATING": 0, "WATCH": 1, "OK": 2}
    sorted_h = sorted(health.values(), key=lambda h: (order.get(h.flag, 3), h.return_since_rebalance or 0))
    return "\n".join(h.as_prompt_line() for h in sorted_h)
