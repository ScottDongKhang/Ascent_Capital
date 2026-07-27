"""
ascent/monitoring/scenario_planner.py

Adversarial weekend scenario planning.

Stress-tests the current portfolio against 5 fixed scenarios + 1 dynamic
scenario derived from this week's debrief. For each scenario, computes
portfolio dollar impact and asks Sonnet to assess probability and
recommend a pre-emptive adjustment.

Flags any scenario with estimated probability > 40% for console alert.

Writes: data_cache/scenario_plan.json
Called by weekend_runner.py every weekend run.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

SCENARIO_PLAN_PATH  = Path("data_cache/scenario_plan.json")
MERGED_WEIGHTS_PATH = Path("execution/merged_weights.json")
HOLDINGS_LOG        = Path("logs/holdings_log.jsonl")

PROBABILITY_ALERT_THRESHOLD = 0.40


def _load_current_portfolio() -> dict:
    """Load current weights from merged_weights.json."""
    if MERGED_WEIGHTS_PATH.exists():
        try:
            data = json.loads(MERGED_WEIGHTS_PATH.read_text())
            return {k: v for k, v in data.items()
                    if isinstance(v, float) and v > 0.001}
        except Exception:
            pass
    return {}


def _load_nav() -> float:
    """Get latest NAV from holdings log."""
    if not HOLDINGS_LOG.exists():
        return 100_000.0
    try:
        lines = [l for l in HOLDINGS_LOG.read_text().splitlines() if l.strip()]
        last = json.loads(lines[-1])
        return float(last.get("equity", 100_000.0)) + float(last.get("cash", 0.0))
    except Exception:
        return 100_000.0


def _compute_portfolio_impact(weights: dict, shocks: dict, nav: float) -> dict:
    """
    For each symbol in shocks, apply shock return.
    Returns: {total_impact_pct, total_impact_usd, per_position}
    shocks: {symbol: return_shock}  e.g. {"SPY": -0.03, ...}
    """
    impact_total = 0.0
    per_position = []

    for sym, w in weights.items():
        # Apply symbol-specific shock if present, else apply market shock if provided
        shock = shocks.get(sym, shocks.get("__market__", 0.0))
        contrib = w * shock
        impact_total += contrib
        if abs(contrib) > 0.0005:
            per_position.append({
                "symbol": sym,
                "weight": round(w, 4),
                "shock": round(shock, 4),
                "impact_pct": round(contrib * 100, 3),
            })

    per_position.sort(key=lambda x: x["impact_pct"])
    return {
        "total_impact_pct": round(impact_total * 100, 2),
        "total_impact_usd": round(impact_total * nav, 0),
        "worst_positions": per_position[:5],
    }


def _build_scenarios(weights: dict, dynamic_risk: Optional[str] = None) -> list:
    """
    Build 5 fixed + 1 dynamic scenario definitions.
    Each scenario: {name, description, shocks, rationale}
    """
    # Identify largest position
    largest = max(weights, key=lambda s: weights[s]) if weights else "SPY"
    em_names = {"EWY", "EEM", "EWT", "EWC", "EWZ", "VWO", "AAXJ", "INDA"}
    em_held  = [s for s in weights if s in em_names]

    scenarios = [
        {
            "name": "risk_off_shock",
            "description": "SPY opens -3% Monday — broad risk-off",
            "shocks": {"__market__": -0.03, "GLD": 0.01, "TLT": 0.015, "BIL": 0.0},
            "rationale": "Tariff shock, Fed surprise, geopolitical event",
        },
        {
            "name": "regime_flip_stressed",
            "description": "Regime shifts to stressed mid-week",
            "shocks": {
                "__market__": -0.015,
                "HYG": -0.02, "LQD": -0.01,
                "GLD": 0.01, "TLT": 0.02, "UUP": 0.01,
            },
            "rationale": "VIX spike, credit spread widening, SPY breaks 50MA",
        },
        {
            "name": "largest_position_blowup",
            "description": f"{largest} drops -15% (idiosyncratic blow-up)",
            "shocks": {largest: -0.15},
            "rationale": "Earnings miss, fraud, CFO departure, sector regulatory shock",
        },
        {
            "name": "rates_spike_50bps",
            "description": "10Y yield spikes 50bps — duration shock",
            "shocks": {
                "TLT": -0.05, "IEF": -0.03, "LQD": -0.025,
                "HYG": -0.02, "VNQ": -0.03, "IFRA": -0.02,
                "__market__": -0.01,
            },
            "rationale": "Hot CPI print, Fed hawkish pivot, Treasury supply shock",
        },
        {
            "name": "em_selloff",
            "description": "EM selloff: all EM positions -5%",
            "shocks": {s: -0.05 for s in em_held} if em_held else {"EWY": -0.05},
            "rationale": "USD strength, China slowdown, EM credit event",
        },
    ]

    # Dynamic scenario from debrief
    if dynamic_risk:
        scenarios.append({
            "name": "dynamic_debrief_risk",
            "description": f"Debrief-identified risk: {dynamic_risk[:80]}",
            "shocks": {"__market__": -0.02},
            "rationale": "Identified by weekly debrief synthesis as top risk",
        })

    return scenarios


def _assess_scenarios_with_llm(
    scenarios_with_impact: list,
    weights: dict,
    regime: str,
) -> list:
    """
    Call Sonnet once with all scenarios. Returns probability + recommendation per scenario.
    Single call to minimize cost.
    """
    try:
        from ascent.llm.client import SONNET_MODEL, extract_text
        import anthropic
        client = anthropic.Anthropic()

        scenario_text = "\n\n".join([
            f"SCENARIO: {s['name']}\n"
            f"Description: {s['description']}\n"
            f"Portfolio impact: {s['impact']['total_impact_pct']:+.2f}% (${s['impact']['total_impact_usd']:,.0f})\n"
            f"Rationale: {s['rationale']}"
            for s in scenarios_with_impact
        ])

        top_positions = sorted(weights.items(), key=lambda x: -x[1])[:10]
        positions_text = ", ".join(f"{s} {w:.1%}" for s, w in top_positions)

        prompt = f"""You are a risk manager at a quantitative hedge fund.
Current regime: {regime}
Current portfolio (top positions): {positions_text}

Assess each scenario's probability over the next 5 trading days and recommend
a pre-emptive portfolio adjustment if warranted.

{scenario_text}

Respond with a JSON array, one object per scenario, in the same order:
[
  {{
    "name": "<scenario name>",
    "probability": <0.0–1.0 float>,
    "probability_rationale": "<one sentence why>",
    "pre_emptive_action": "<one concrete action or 'no action needed'>",
    "urgency": "high" | "medium" | "low"
  }},
  ...
]"""

        resp = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4096,   # headroom: thinking shares this budget
            messages=[{"role": "user", "content": prompt}],
        )
        text = extract_text(resp).strip()
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[ScenarioPlanner] LLM assessment failed: {e}")
        return [{"name": s["name"], "probability": None, "pre_emptive_action": "assessment unavailable"}
                for s in scenarios_with_impact]


def run_scenario_planning(debrief: Optional[dict] = None) -> dict:
    """
    Run all scenarios, assess probabilities, write scenario_plan.json.
    Returns the plan dict.
    """
    print("[ScenarioPlanner] Running adversarial scenario planning...")

    weights = _load_current_portfolio()
    nav     = _load_nav()

    if not weights:
        print("[ScenarioPlanner] No portfolio weights found — skipping")
        return {}

    # Get current regime
    regime = "unknown"
    try:
        sig = json.loads(Path("dashboard/regime_signal.json").read_text())
        regime = sig.get("regime", sig.get("label", "unknown"))
    except Exception:
        pass

    # Dynamic risk from debrief
    dynamic_risk = None
    if debrief:
        dynamic_risk = debrief.get("synthesis", {}).get("top_risk")

    scenarios = _build_scenarios(weights, dynamic_risk)

    # Compute portfolio impact for each scenario
    scenarios_with_impact = []
    for s in scenarios:
        impact = _compute_portfolio_impact(weights, s["shocks"], nav)
        scenarios_with_impact.append({**s, "impact": impact})

    # LLM probability assessment
    assessments = _assess_scenarios_with_llm(scenarios_with_impact, weights, regime)

    # Merge impact + assessment
    assessment_by_name = {a["name"]: a for a in assessments}
    final_scenarios = []
    flagged = []

    for s in scenarios_with_impact:
        assessment = assessment_by_name.get(s["name"], {})
        prob = assessment.get("probability")
        entry = {
            "name": s["name"],
            "description": s["description"],
            "rationale": s["rationale"],
            "impact": s["impact"],
            "probability": prob,
            "probability_rationale": assessment.get("probability_rationale", ""),
            "pre_emptive_action": assessment.get("pre_emptive_action", ""),
            "urgency": assessment.get("urgency", "low"),
            "flagged": bool(prob and prob >= PROBABILITY_ALERT_THRESHOLD),
        }
        final_scenarios.append(entry)
        if entry["flagged"]:
            flagged.append(entry)

    plan = {
        "as_of": date.today().isoformat(),
        "regime": regime,
        "nav": nav,
        "scenarios": final_scenarios,
        "flagged_count": len(flagged),
    }

    import tempfile, os
    tmp = Path(tempfile.mktemp(dir=SCENARIO_PLAN_PATH.parent, suffix=".tmp"))
    SCENARIO_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(plan, indent=2))
    os.replace(tmp, SCENARIO_PLAN_PATH)

    # Console alerts for high-probability scenarios
    if flagged:
        print(f"\n{'!'*60}")
        print(f"[ScenarioPlanner] ⚠  {len(flagged)} SCENARIO(S) FLAGGED (prob ≥ {PROBABILITY_ALERT_THRESHOLD:.0%})")
        for f in flagged:
            prob_str = f"{f['probability']:.0%}" if f['probability'] else "?"
            print(f"  {f['name']}: {prob_str} probability | "
                  f"impact {f['impact']['total_impact_pct']:+.2f}%")
            print(f"  Action: {f['pre_emptive_action']}")
        print(f"{'!'*60}\n")
    else:
        print(f"[ScenarioPlanner] All {len(final_scenarios)} scenarios below alert threshold. "
              f"Worst impact: {min(s['impact']['total_impact_pct'] for s in final_scenarios):+.2f}%")

    return plan
