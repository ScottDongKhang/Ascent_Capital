"""
scenario_library_20in20.py
===========================
Named scenario definitions for 20in20 demo.
Drop this in the repo root alongside run_20in20.py.

Each Scenario defines return shocks by theme bucket.
The runner in run_20in20.py applies these to the watchlist
and estimates portfolio-level P&L impact.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    shocks: Dict[str, float]   # theme -> return shock (e.g. -0.06 = -6%)
    horizon_days: int
    severity: str              # "mild" | "moderate" | "severe"


def build_scenario_library_20in20() -> List[Scenario]:
    """
    Returns the canonical list of 20in20 scenarios.
    Shocks are approximate 5-day return impacts by theme bucket.
    Keys must match theme names in watchlists_20in20.py.
    """
    return [
        Scenario(
            name="EM_RISK_OFF",
            description=(
                "Global risk-off driven by EM stress — capital flight from "
                "emerging markets, USD strengthens, commodities drop."
            ),
            shocks={
                "EM_Asia":     -0.06,
                "Consumer":    -0.02,
                "Fintech":     -0.03,
                "Rates_Macro": +0.02,   # TLT/IEF rally, USD up
                "US_Broad":    -0.02,
                "Sectors":     -0.01,
                "Logistics":   -0.01,
            },
            horizon_days=5,
            severity="moderate",
        ),
        Scenario(
            name="USD_UP",
            description=(
                "Sharp USD appreciation — hurts EM assets and multinationals, "
                "benefits USD-denominated safe havens."
            ),
            shocks={
                "EM_Asia":     -0.04,
                "Consumer":    -0.01,
                "Fintech":     -0.02,
                "Rates_Macro": +0.01,
                "US_Broad":    -0.01,
                "Sectors":      0.00,
                "Logistics":   -0.01,
            },
            horizon_days=5,
            severity="mild",
        ),
        Scenario(
            name="RATES_SPIKE",
            description=(
                "Sudden spike in long-end rates — duration assets sell off, "
                "growth/tech multiple compression, financials mixed."
            ),
            shocks={
                "Rates_Macro": -0.05,   # TLT/IEF crushed
                "Sectors":     -0.03,   # growth sectors hit
                "US_Broad":    -0.03,
                "Fintech":     -0.04,
                "Consumer":    -0.02,
                "EM_Asia":     -0.03,
                "Logistics":   -0.01,
            },
            horizon_days=5,
            severity="moderate",
        ),
        Scenario(
            name="TECH_UNWIND",
            description=(
                "Rotation out of mega-cap tech — Nasdaq-led selloff, "
                "momentum unwind, defensive sectors outperform."
            ),
            shocks={
                "US_Broad":    -0.04,   # QQQ/SPY drag
                "Sectors":     -0.05,   # XLK hardest hit
                "Fintech":     -0.04,
                "Consumer":    -0.02,
                "EM_Asia":     -0.02,
                "Logistics":   -0.01,
                "Rates_Macro": +0.01,   # flight to bonds
            },
            horizon_days=5,
            severity="moderate",
        ),
        Scenario(
            name="COMMODITY_SHOCK",
            description=(
                "Oil/commodity spike — inflation re-acceleration fear, "
                "consumer margins squeezed, energy sector outperforms."
            ),
            shocks={
                "Rates_Macro": +0.04,   # oil/gold up
                "Logistics":   -0.02,   # cost pressure
                "Consumer":    -0.03,   # margin squeeze
                "EM_Asia":     +0.01,   # commodity exporters benefit
                "US_Broad":    -0.02,
                "Sectors":     +0.01,   # XLE up
                "Fintech":     -0.01,
            },
            horizon_days=5,
            severity="mild",
        ),
    ]


def run_scenario_brief(
    prices,              # pd.DataFrame indexed by date, columns=tickers
    watchlists,          # WatchlistSet
    scenario_lib: List[Scenario],
    cfg,                 # Config20in20
):
    """
    Applies each scenario's theme-level shocks to the watchlist
    and returns a summary DataFrame.

    Assumes equal weight across all tickers for simplicity.
    """
    import pandas as pd

    table = watchlists.table
    n_total = len(table)
    if n_total == 0:
        return pd.DataFrame()

    rows = []
    for sc in scenario_lib:
        pnl = 0.0
        exposed_tickers = []

        for _, meta in table.iterrows():
            shock = sc.shocks.get(meta["theme"], 0.0)
            pnl += shock / n_total
            if abs(shock) >= 0.03 and meta["ticker"] not in exposed_tickers:
                exposed_tickers.append(meta["ticker"])

        rows.append({
            "scenario":     sc.name,
            "description":  sc.description,
            "severity":     sc.severity,
            "horizon_days": sc.horizon_days,
            "pnl_est":      round(pnl, 4),
            "most_exposed": ", ".join(exposed_tickers[:6]),
        })

    df = (
        pd.DataFrame(rows)
        .sort_values("pnl_est")
        .reset_index(drop=True)
    )
    return df


if __name__ == "__main__":
    lib = build_scenario_library_20in20()
    for s in lib:
        print(f"{s.name:<20}  severity={s.severity:<8}  horizon={s.horizon_days}d")
        print(f"  {s.description[:80]}")
        print()
