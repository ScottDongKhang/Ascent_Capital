"""
simulation/scenario_simulator.py
Parametric scenario shock simulator for Ascent Capital.

Takes current portfolio weights and applies predefined shock scenarios.
Computes first-order portfolio impact (no Monte Carlo — pure shock arithmetic).
Writes to outputs/scenarios/scenarios_YYYY-MM-DD.json.

No external dependencies — uses weights already computed by the orchestrator.

Usage:
    python3 -m simulation.scenario_simulator
"""

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List
import numpy as np

OUTPUT_DIR = Path("outputs/scenarios")

SCENARIOS = {
    "em_risk_off": {
        "description": "EM equities -15%, USD +5%, flight to safety",
        "shocks": {
            "EEM": -0.15, "VWO": -0.15, "EWZ": -0.20,
            "EWT": -0.12, "AAXJ": -0.12, "EWJ": -0.08,
            "UUP": 0.05, "TLT": 0.03, "GLD": 0.05, "BIL": 0.01,
        },
    },
    "rates_spike": {
        "description": "Rates spike — TLT -8%, rate-sensitive equities repriced",
        "shocks": {
            "TLT": -0.08, "IEF": -0.04, "LQD": -0.04,
            "VNQ": -0.06, "EQIX": -0.05, "AMT": -0.05,
            "NEE": -0.04, "D": -0.03,
        },
    },
    "tech_unwind": {
        "description": "Nasdaq -12%, tech correlation spike",
        "shocks": {
            "AAPL": -0.12, "MSFT": -0.12, "NVDA": -0.15,
            "GOOGL": -0.12, "META": -0.12, "AMZN": -0.10,
            "CRM": -0.10, "ADBE": -0.10, "NOW": -0.10,
        },
    },
    "commodity_shock": {
        "description": "Oil spike +15%, gold rallies +8%, inflation fears",
        "shocks": {
            "USO": 0.15, "GLD": 0.08, "DBA": 0.10,
            "XOM": 0.08, "COP": 0.08, "MPC": 0.06,
            "TLT": -0.03, "IEF": -0.02,
        },
    },
    "correlation_crisis": {
        "description": "All risky assets fall simultaneously — correlation breakdown",
        "shocks": {sym: -0.10 for sym in [
            "AAPL", "MSFT", "GOOGL", "META", "AMZN",
            "EEM", "VWO", "VNQ", "USO", "HYG",
            "EQIX", "CAT", "MPC",
        ]},
    },
}


def compute_scenario_impact(
    portfolio_weights: Dict[str, float],
    scenario_name: str,
    shocks: Dict[str, float],
    description: str = "",
) -> Dict:
    """
    Compute first-order portfolio impact of a scenario shock.

    Returns dict with:
        scenario, description, portfolio_impact_pct,
        affected_positions, unaffected_weight_pct, n_affected
    """
    total_impact = 0.0
    affected     = []

    for sym, weight in portfolio_weights.items():
        if sym in shocks:
            impact = weight * shocks[sym]
            total_impact += impact
            affected.append({
                "symbol":       sym,
                "weight":       round(weight, 4),
                "shock":        round(shocks[sym], 4),
                "contribution": round(impact, 6),
            })

    unaffected_weight = sum(w for sym, w in portfolio_weights.items() if sym not in shocks)

    return {
        "scenario":              scenario_name,
        "description":           description,
        "portfolio_impact_pct":  round(total_impact * 100, 2),
        "affected_positions":    sorted(affected, key=lambda x: x["contribution"]),
        "unaffected_weight_pct": round(unaffected_weight * 100, 2),
        "n_affected":            len(affected),
    }


def compute_scenario_percentiles(
    portfolio_weights,
    scenario_name,
    shocks,
    description="",
    n_sims=1000,
    noise_std=0.30,
):
    import numpy as np
    rng = np.random.default_rng(seed=42)
    sim_impacts = []
    for _ in range(n_sims):
        total = 0.0
        for sym, weight in portfolio_weights.items():
            if sym in shocks:
                base_shock = shocks[sym]
                noise = rng.normal(0, abs(base_shock) * noise_std)
                total += weight * (base_shock + noise)
        sim_impacts.append(total * 100)
    arr = __import__("numpy").array(sim_impacts)
    pc = {
        "p5":  round(float(__import__("numpy").percentile(arr, 5)), 2),
        "p25": round(float(__import__("numpy").percentile(arr, 25)), 2),
        "p50": round(float(__import__("numpy").percentile(arr, 50)), 2),
        "p75": round(float(__import__("numpy").percentile(arr, 75)), 2),
        "p95": round(float(__import__("numpy").percentile(arr, 95)), 2),
    }
    point = sum(portfolio_weights.get(s, 0) * sh for s, sh in shocks.items()) * 100
    return {
        "scenario": scenario_name,
        "description": description,
        "point_impact_pct": round(point, 2),
        "percentiles": pc,
        "worst_case": pc["p5"],
        "base_case": pc["p50"],
        "best_case": pc["p95"],
    }


def run_all_scenarios(
    portfolio_weights: Dict[str, float],
    run_date: date = None,
) -> List[Dict]:
    """
    Run all predefined scenarios against current portfolio weights.
    Writes results to JSON and returns the list.
    """
    run_date = run_date or date.today()
    results  = []

    print(f"\n[Scenario] Running {len(SCENARIOS)} scenarios | {len(portfolio_weights)} positions")

    for name, spec in SCENARIOS.items():
        result = compute_scenario_impact(
            portfolio_weights, name, spec["shocks"], spec["description"]
        )
        mc = compute_scenario_percentiles(
            portfolio_weights, name, spec["shocks"], spec["description"]
        )
        result["percentiles"] = mc["percentiles"]
        result["worst_case"]  = mc["worst_case"]
        result["base_case"]   = mc["base_case"]
        result["best_case"]   = mc["best_case"]
        results.append(result)
        sign = "+" if result["portfolio_impact_pct"] >= 0 else ""
        print(f"  {name:<22} {sign}{result['portfolio_impact_pct']:.2f}%  p5={mc['worst_case']:+.1f}%  p50={mc['base_case']:+.1f}%  p95={mc['best_case']:+.1f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = OUTPUT_DIR / f"scenarios_{run_date.isoformat()}.json"

    payload = {
        "date":         run_date.isoformat(),
        "generated_at": datetime.now().isoformat(),
        "n_positions":  len(portfolio_weights),
        "portfolio":    portfolio_weights,
        "scenarios":    results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[Scenario] Written to {out_path}")
    return results


if __name__ == "__main__":
    # Test with sample weights matching today's dry run output
    test_weights = {
        "APD": 0.084, "CAT": 0.084, "EQIX": 0.084, "MPC": 0.084,
        "MRK": 0.084, "NEE": 0.084, "SEDG": 0.084, "T": 0.084,
        "UUP": 0.082, "GLD": 0.082, "USO": 0.082,
        "HAS": 0.036, "SCHW": 0.026, "HYG": 0.019,
    }
    run_all_scenarios(test_weights)
