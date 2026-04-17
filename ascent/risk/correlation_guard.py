"""
ascent/risk/correlation_guard.py
Cross-agent correlation guard.

Checks whether instruments across different agents are too correlated,
preventing the orchestrator from creating hidden concentration risk.

CORRELATION_CAP = 0.70 (63-day trailing)

Usage:
    Called by orchestrator/central_intelligence.py (v2).
    Also callable standalone for diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from ascent.data.store.parquet import load_parquet, has_data

CORRELATION_CAP = 0.70   # max acceptable trailing 63-day correlation
LOOKBACK        = 63     # trading days


# ── Price loading ──────────────────────────────────────────────────────────────

def _load_combined_prices(symbols: List[str]) -> pd.DataFrame:
    """
    Load close prices for a combined set of symbols across agents.

    Equity prices: prices_live.parquet
    Macro prices:  prices_macro.parquet
    Combines and deduplicates columns.
    """
    frames = []

    if has_data("prices_live"):
        try:
            eq = load_parquet("prices_live")
            if isinstance(eq.columns, pd.MultiIndex):
                lvl0 = eq.columns.get_level_values(0)
                eq   = eq["Close"] if "Close" in lvl0 else eq.iloc[:, :len(symbols)]
            overlap = [s for s in symbols if s in eq.columns]
            if overlap:
                frames.append(eq[overlap])
        except Exception as e:
            print(f"[CorrGuard] Could not load prices_live: {e}")

    if has_data("prices_macro"):
        try:
            macro = load_parquet("prices_macro")
            if isinstance(macro.columns, pd.MultiIndex):
                lvl0  = macro.columns.get_level_values(0)
                macro = macro["Close"] if "Close" in lvl0 else macro.iloc[:, :len(symbols)]
            overlap = [s for s in symbols if s in macro.columns]
            if overlap:
                frames.append(macro[overlap])
        except Exception as e:
            print(f"[CorrGuard] Could not load prices_macro: {e}")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined


# ── Correlation check ──────────────────────────────────────────────────────────

def check_cross_agent_correlation(
    agent_weights: Dict[str, Dict[str, float]],
    cap: float = CORRELATION_CAP,
    lookback: int = LOOKBACK,
) -> List[Tuple[str, str, float]]:
    """
    Check for high correlations between instruments held by DIFFERENT agents.

    Args:
        agent_weights: {agent_id: {symbol: weight}}
        cap:           Maximum acceptable absolute correlation
        lookback:      Trailing window in trading days

    Returns:
        List of (symbol_a, symbol_b, correlation) tuples exceeding the cap.
        Empty list if data is insufficient.
    """
    all_symbols: set = set()
    for weights in agent_weights.values():
        all_symbols.update(weights.keys())

    if len(all_symbols) < 2:
        return []

    prices = _load_combined_prices(list(all_symbols))
    if prices.empty or len(prices) < lookback:
        print(f"[CorrGuard] Insufficient price data ({len(prices)} rows, need {lookback})")
        return []

    returns = prices.pct_change().dropna()
    if len(returns) < lookback:
        return []

    recent_returns = returns.iloc[-lookback:]
    corr_matrix    = recent_returns.corr()

    # Only check pairs from DIFFERENT agents
    violations: List[Tuple[str, str, float]] = []
    agent_ids = list(agent_weights.keys())

    for i, agent_a in enumerate(agent_ids):
        for agent_b in agent_ids[i + 1:]:
            syms_a = set(agent_weights[agent_a].keys())
            syms_b = set(agent_weights[agent_b].keys())

            for sym_a in syms_a:
                for sym_b in syms_b:
                    if sym_a == sym_b:
                        continue  # same symbol in both agents — handled by weight merging
                    if sym_a in corr_matrix.columns and sym_b in corr_matrix.columns:
                        corr = float(corr_matrix.loc[sym_a, sym_b])
                        if abs(corr) > cap:
                            violations.append((sym_a, sym_b, round(corr, 4)))

    if violations:
        print(f"[CorrGuard] {len(violations)} cross-agent correlation violation(s) (cap={cap}):")
        for sym_a, sym_b, corr in violations:
            print(f"  {sym_a} <-> {sym_b}: {corr:+.3f}")
    else:
        print(f"[CorrGuard] No cross-agent correlation violations (cap={cap}, lookback={lookback}d)")

    return violations


# ── Weight adjustment ──────────────────────────────────────────────────────────

def apply_correlation_adjustments(
    merged_weights: Dict[str, float],
    violations: List[Tuple[str, str, float]],
) -> Dict[str, float]:
    """
    Reduce weights for correlated pairs.
    For each violation, halve the weight of the SMALLER position.
    Renormalizes the result.

    Args:
        merged_weights: Combined portfolio {symbol: weight}
        violations:     Output of check_cross_agent_correlation()

    Returns:
        Adjusted and renormalized weights dict.
    """
    if not violations:
        return merged_weights

    adjusted = dict(merged_weights)

    for sym_a, sym_b, corr in violations:
        w_a = adjusted.get(sym_a, 0.0)
        w_b = adjusted.get(sym_b, 0.0)

        if w_a > 0 and w_b > 0:
            if w_a <= w_b:
                adjusted[sym_a] = w_a * 0.5
                print(f"[CorrGuard] Halved {sym_a} {w_a:.2%} → {adjusted[sym_a]:.2%} "
                      f"(corr={corr:+.3f} with {sym_b})")
            else:
                adjusted[sym_b] = w_b * 0.5
                print(f"[CorrGuard] Halved {sym_b} {w_b:.2%} → {adjusted[sym_b]:.2%} "
                      f"(corr={corr:+.3f} with {sym_a})")

    # Renormalize, drop < 0.5%
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {
            k: round(v / total, 6)
            for k, v in adjusted.items()
            if v / total > 0.005
        }

    return adjusted
