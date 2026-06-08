"""ascent/causal/causal_discovery.py

Runs the PC constraint-based causal discovery algorithm on FRED macro
data + sector ETF weekly returns to produce the macro causal DAG.

Output written to data_cache/macro_causal_dag.json.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MACRO_DAG_PATH = Path("data_cache/macro_causal_dag.json")

MACRO_SERIES = ["fed_funds_rate", "hy_spread", "vix", "unemployment"]
SECTOR_ETFS  = ["XLF", "XLK", "XLV", "XLE", "XLP"]


def run_pc(
    data: np.ndarray,
    node_names: List[str],
    alpha: float = 0.05,
) -> dict:
    """
    Run the PC algorithm on a T×N data matrix.
    Returns a dict with nodes, edges (directed), and active_transmission_chains.
    """
    from causallearn.search.ConstraintBased.PC import pc as run_pc_alg

    cg = run_pc_alg(data, alpha=alpha, indep_test="fisherz", show_progress=False)
    adj = cg.G.graph  # N×N: adj[i,j]==-1 and adj[j,i]==1 means i→j

    # Compute pairwise correlations for strength annotation
    corr = np.corrcoef(data.T)

    n = len(node_names)
    edges = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Directed edge i→j: adj[i,j]==-1 and adj[j,i]==1
            if adj[i, j] == -1 and adj[j, i] == 1:
                key = (node_names[i], node_names[j])
                if key in seen:
                    continue
                seen.add(key)
                r = corr[i, j]
                strength = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
                direction = "positive" if r > 0 else "negative"
                edges.append({
                    "from": node_names[i],
                    "to": node_names[j],
                    "strength": strength,
                    "direction": direction,
                })

    chains = _find_transmission_chains(edges, node_names)
    return {
        "nodes": node_names,
        "edges": edges,
        "active_transmission_chains": chains,
    }


def _find_transmission_chains(edges: list, node_names: list) -> List[str]:
    """
    Find paths of length 2+ through the DAG and return them as strings.
    Limited to paths starting from macro nodes (fed_rate, hy_spread, vix).
    """
    adj_map: dict = {}
    for e in edges:
        adj_map.setdefault(e["from"], []).append(e["to"])

    chains = []
    source_nodes = [n for n in node_names if n in MACRO_SERIES[:3]]
    for src in source_nodes:
        for mid in adj_map.get(src, []):
            for dst in adj_map.get(mid, []):
                if dst != src:
                    chains.append(f"{src} → {mid} → {dst}")
    return chains[:10]  # cap at 10 to keep JSON small


def _load_macro_data(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Convert macro_df to weekly frequency using last observation per week."""
    numeric = macro_df.select_dtypes(include=[np.number])
    weekly = numeric.resample("W-FRI").last().dropna(how="all")
    return weekly


def _load_sector_data(sector_df: pd.DataFrame) -> pd.DataFrame:
    """Compute weekly returns from sector ETF price data."""
    prices = sector_df.resample("W-FRI").last()
    returns = prices.pct_change().dropna(how="all")
    return returns


def discover_macro_dag(
    macro_df: pd.DataFrame,
    sector_df: pd.DataFrame,
    regime: str = "calm_bull",
    output_path: Optional[Path] = None,
) -> dict:
    """
    Build the macro causal DAG from FRED macro data + sector ETF returns.

    Args:
        macro_df: DataFrame indexed by date with columns matching MACRO_SERIES names
        sector_df: DataFrame indexed by date with columns for SECTOR_ETFS prices or returns
        regime: current regime label for metadata
        output_path: where to write JSON (default: MACRO_DAG_PATH)

    Returns:
        The DAG dict (also written to output_path).
    """
    if output_path is None:
        output_path = MACRO_DAG_PATH

    macro_weekly = _load_macro_data(macro_df)

    # Normalize timezone on macro (FRED is tz-naive but guard against future changes)
    if macro_weekly.index.tz is not None:
        macro_weekly.index = macro_weekly.index.tz_localize(None)

    if sector_df.empty:
        combined = macro_weekly.iloc[-104:].dropna()
    else:
        # sector_df may already be returns (small values) or prices (large values)
        sample = sector_df.select_dtypes(include=[np.number])
        if sample.empty or sample.abs().mean().mean() < 0.1:
            sector_returns = sample.resample("W-FRI").last().dropna(how="all")
        else:
            sector_returns = _load_sector_data(sector_df)

        # Normalize timezone: strip tz-awareness before joining (Yahoo prices are tz-aware)
        if sector_returns.index.tz is not None:
            sector_returns.index = sector_returns.index.tz_localize(None)

        # Align on common dates, keep last 2 years (~104 weekly observations)
        combined = macro_weekly.join(sector_returns, how="inner").dropna()

    combined = combined.iloc[-104:]

    if len(combined) < 30:
        log.warning("[CausalDiscovery] Insufficient data (%d rows) for PC algorithm", len(combined))
        return {}

    node_names = list(combined.columns)
    data = combined.values.astype(float)

    log.info("[CausalDiscovery] Running PC on %d nodes, %d observations", len(node_names), len(data))
    dag = run_pc(data, node_names, alpha=0.05)
    dag["as_of"] = str(date.today())
    dag["regime"] = regime

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dag, indent=2))
    log.info(
        "[CausalDiscovery] DAG written: %d nodes, %d edges, %d chains",
        len(dag["nodes"]), len(dag["edges"]), len(dag["active_transmission_chains"]),
    )
    return dag


def run_discovery(regime: str = "calm_bull") -> dict:
    """
    Entry point for the weekend runner.
    Loads data from standard parquet caches and writes macro_causal_dag.json.
    """
    macro_path = Path("data_cache/macro_live.parquet")
    prices_path = Path("data_cache/prices_live.parquet")

    if not macro_path.exists():
        log.warning("[CausalDiscovery] macro_live.parquet not found — skipping")
        return {}

    # Load FRED macro: pivot from long to wide format
    macro_raw = pd.read_parquet(macro_path)
    macro_pivot = (
        macro_raw[macro_raw["name"].isin(MACRO_SERIES)]
        .pivot_table(index="date", columns="name", values="value", aggfunc="last")
    )
    macro_pivot.index = pd.to_datetime(macro_pivot.index)

    # Load sector ETF prices — check prices_live first, then fetch from yfinance
    sector_prices = pd.DataFrame()
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        if "symbol" in prices.columns:
            subset = prices[prices["symbol"].isin(SECTOR_ETFS)]
            if not subset.empty:
                sector_prices = subset.pivot_table(
                    index="date", columns="symbol", values="close", aggfunc="last"
                )
                sector_prices.index = pd.to_datetime(sector_prices.index)

    if sector_prices.empty:
        try:
            import yfinance as yf
            raw = yf.download(SECTOR_ETFS, period="3y", progress=False, auto_adjust=True)
            sector_prices = raw["Close"] if "Close" in raw.columns else raw
            sector_prices.index = pd.to_datetime(sector_prices.index)
            log.info("[CausalDiscovery] Sector ETF prices fetched from yfinance")
        except Exception as exc:
            log.warning("[CausalDiscovery] Could not load sector ETFs (%s) — macro-only mode", exc)
            sector_prices = macro_pivot.copy().iloc[:, :0]  # empty, triggers macro-only path

    return discover_macro_dag(macro_pivot, sector_prices, regime=regime)
