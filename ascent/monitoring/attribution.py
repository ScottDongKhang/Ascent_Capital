"""
ascent/monitoring/attribution.py
Daily position-level P&L attribution.

Computes per-position contribution (weight × return), portfolio total return,
SPY benchmark return, alpha vs SPY, top 5 contributors and drags.

Called by run_all_agents._log_holdings() after each run.
Output appended to logs/attribution_log.jsonl.
"""

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict

ATTRIBUTION_LOG = Path("logs/attribution_log.jsonl")


def _fetch_returns(symbols: List[str]) -> Dict[str, float]:
    """Fetch latest close-to-close returns for given symbols + SPY."""
    syms = list(set(symbols) | {"SPY"})
    try:
        import yfinance as yf
        data = yf.download(syms, period="5d", auto_adjust=True, progress=False)
        if data.empty or len(data) < 2:
            return {}
        import pandas as pd
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data[["Close"]]
            closes.columns = syms[:1]
        rets = closes.pct_change().iloc[-1]
        return {s: float(rets[s]) for s in syms if s in rets.index and not math.isnan(float(rets[s]))}
    except Exception as e:
        print(f"[Attribution] Price fetch failed: {e}")
        return {}


def run_attribution(positions: List[Dict], run_date: date) -> Dict:
    """
    Compute daily attribution from a list of position dicts.
    Each position dict must have: symbol, weight, market_value.
    Returns attribution dict and appends to logs/attribution_log.jsonl.
    """
    symbols = [p["symbol"] for p in positions]
    returns = _fetch_returns(symbols)

    if not returns:
        print("[Attribution] No returns fetched — skipping")
        return {}

    spy_ret = returns.get("SPY", 0.0)

    contribs = []
    for pos in positions:
        sym = pos["symbol"]
        w   = float(pos.get("weight", 0))
        r   = returns.get(sym, 0.0)
        contrib = w * r
        contribs.append({
            "symbol":       sym,
            "weight":       round(w, 4),
            "return":       round(r, 6),
            "contribution": round(contrib, 6),
        })

    port_ret = sum(c["contribution"] for c in contribs)
    alpha    = port_ret - spy_ret

    contribs_sorted  = sorted(contribs, key=lambda x: -x["contribution"])
    top_contributors = [c for c in contribs_sorted if c["contribution"] > 0][:5]
    top_drags        = [c for c in reversed(contribs_sorted) if c["contribution"] < 0][:5]

    report = {
        "date":             run_date.isoformat(),
        "timestamp":        datetime.now().isoformat(),
        "portfolio_return": round(port_ret, 6),
        "spy_return":       round(spy_ret, 6),
        "alpha_vs_spy":     round(alpha, 6),
        "n_positions":      len(positions),
        "top_contributors": top_contributors,
        "top_drags":        top_drags,
        "all_positions":    contribs_sorted,
    }

    ATTRIBUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ATTRIBUTION_LOG, "a") as f:
        f.write(json.dumps(report) + "\n")

    sign = "▲" if alpha >= 0 else "▼"
    print(f"[Attribution] {run_date} | portfolio {port_ret:+.2%} vs SPY {spy_ret:+.2%} ({sign} {alpha:+.2%})")
    if top_contributors:
        best = top_contributors[0]
        print(f"  Best:  {best['symbol']} +{best['contribution']:.3%} (wt {best['weight']:.1%} × ret {best['return']:+.1%})")
    if top_drags:
        worst = top_drags[0]
        print(f"  Worst: {worst['symbol']} {worst['contribution']:.3%} (wt {worst['weight']:.1%} × ret {worst['return']:+.1%})")

    return report
