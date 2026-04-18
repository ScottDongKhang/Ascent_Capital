"""
ascent/monitoring/quant_context.py
Pre-compute quantitative context for LLM debate agents.
Called by debate_runner.py before the debate starts.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

FACTOR_BUCKETS = {
    "rates_long":   {"TLT", "IEF", "LQD"},
    "rates_short":  {"HYG", "JNK"},
    "dollar_long":  {"UUP"},
    "commodities":  {"PDBC", "USO", "DBA", "DBB"},
    "gold":         {"GLD", "IAU"},
    "vol_long":     {"VIXY", "VXX"},
    "vol_short":    {"SVXY"},
    "em_equity":    {"EEM", "VWO", "EWT", "EWZ", "AAXJ", "EWY", "INDA"},
    "us_tech":      {"QQQ", "XLK", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "AVGO"},
    "us_defensive": {"XLU", "XLP", "XLV", "NEE", "WMT", "MRK", "JNJ", "PG"},
    "reits":        {"VNQ", "IYR", "EQIX"},
    "energy":       {"XLE", "MPC", "PSX", "XOM"},
}


def _compute_factor_exposures(weights: Dict[str, float]) -> Dict[str, float]:
    exposures = {}
    for factor, syms in FACTOR_BUCKETS.items():
        total = sum(weights.get(s, 0.0) for s in syms)
        if total > 0.001:
            exposures[factor] = round(total, 4)
    return exposures


def _compute_historical_var(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    lookback: int = 63,
    confidence_levels: tuple = (0.95, 0.99),
) -> Dict[str, float]:
    common = [s for s in weights if s in prices.columns]
    if not common:
        return {"var_95": 0.0, "var_99": 0.0}
    subset = prices[common].dropna(how="all").tail(lookback + 1)
    if len(subset) < 10:
        return {"var_95": 0.0, "var_99": 0.0}
    rets = subset.pct_change().dropna()
    w_arr = np.array([weights.get(s, 0.0) for s in common])
    total_w = w_arr.sum()
    if total_w > 0:
        w_arr /= total_w
    port_rets = rets.values @ w_arr
    result = {}
    for cl in confidence_levels:
        pctile = np.percentile(port_rets, (1 - cl) * 100)
        key = f"var_{int(cl*100)}"
        result[key] = round(float(pctile), 6)
    return result


def _compute_sector_concentration(weights: Dict[str, float]) -> Dict[str, float]:
    exposures = _compute_factor_exposures(weights)
    total = sum(weights.values()) or 1.0
    classified = sum(exposures.values())
    unclassified = total - classified
    if unclassified > 0.01:
        exposures["unclassified"] = round(unclassified, 4)
    return {k: round(v / total, 4) for k, v in sorted(exposures.items(), key=lambda x: -x[1])}


def _compute_top_correlations(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    lookback: int = 63,
    top_n: int = 3,
) -> List[Dict]:
    common = [s for s in weights if s in prices.columns and weights[s] > 0.02]
    if len(common) < 2:
        return []
    subset = prices[common].dropna(how="all").tail(lookback + 1)
    if len(subset) < 21:
        return []
    corr_matrix = subset.pct_change().dropna().corr()
    pairs = []
    for i, s1 in enumerate(common):
        for s2 in common[i+1:]:
            if s1 in corr_matrix.index and s2 in corr_matrix.columns:
                c = corr_matrix.loc[s1, s2]
                if not np.isnan(c):
                    pairs.append({"sym1": s1, "sym2": s2, "correlation": round(float(c), 3)})
    pairs.sort(key=lambda x: -abs(x["correlation"]))
    return pairs[:top_n]


def _build_summary_text(
    weights: Dict[str, float],
    var_data: Dict[str, float],
    factor_exposures: Dict[str, float],
    top_corr: List[Dict],
) -> str:
    lines = ["QUANTITATIVE RISK CONTEXT (pre-computed):"]
    var_95 = var_data.get("var_95", 0.0)
    var_99 = var_data.get("var_99", 0.0)
    lines.append(f"\nPortfolio VaR (historical simulation, 63 trading days):")
    lines.append(f"  95th percentile (1-day): {var_95:.2%}  (1 in 20 chance of loss >= this)")
    lines.append(f"  99th percentile (1-day): {var_99:.2%}  (1 in 100 chance of loss >= this)")
    lines.append("\nFactor exposures (sum of position weights per factor):")
    for factor, exp in sorted(factor_exposures.items(), key=lambda x: -x[1]):
        lines.append(f"  {factor:<18} {exp:.1%}")
    high_corr = [p for p in top_corr if p["correlation"] > 0.65]
    if high_corr:
        lines.append("\nHigh-correlation pairs (potential double-counting of risk):")
        for p in high_corr:
            lines.append(f"  {p['sym1']} <-> {p['sym2']}: {p['correlation']:.2f}")
    else:
        lines.append("\nCorrelation: No high-correlation pairs (>0.65) detected.")
    top_factor = max(factor_exposures.items(), key=lambda x: x[1]) if factor_exposures else None
    if top_factor and top_factor[1] > 0.30:
        lines.append(f"\nConcentration warning: {top_factor[0]} is {top_factor[1]:.1%} of portfolio")
    return "\n".join(lines)


def build_quant_context(
    weights: Dict[str, float],
    prices: Optional[pd.DataFrame],
    lookback: int = 63,
) -> Dict:
    """Build quantitative context for debate agents. prices must be WIDE format (symbol columns)."""
    if not weights:
        return {
            "portfolio_var_95": 0.0, "portfolio_var_99": 0.0,
            "factor_exposures": {}, "sector_concentration": {},
            "top_correlated_pairs": [],
            "summary_text": "QUANTITATIVE RISK CONTEXT: No positions provided.",
        }
    factor_exposures = _compute_factor_exposures(weights)
    sector_concentration = _compute_sector_concentration(weights)
    top_corr = []
    var_data = {"var_95": 0.0, "var_99": 0.0}
    if prices is not None and not prices.empty:
        raw_var = _compute_historical_var(weights, prices, lookback)
        var_data = {"var_95": raw_var.get("var_95", 0.0), "var_99": raw_var.get("var_99", 0.0)}
        top_corr = _compute_top_correlations(weights, prices, lookback)
    summary = _build_summary_text(weights, var_data, factor_exposures, top_corr)
    return {
        "portfolio_var_95": var_data["var_95"],
        "portfolio_var_99": var_data["var_99"],
        "factor_exposures": factor_exposures,
        "sector_concentration": sector_concentration,
        "top_correlated_pairs": top_corr,
        "summary_text": summary,
    }
