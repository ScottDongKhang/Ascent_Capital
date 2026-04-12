"""
debate/debate_runner.py
Orchestrates the full LLM debate session before each rebalance.

Called by run_all_agents.py on rebalance days.
Writes full debate record to outputs/debate_log/verdict_YYYY-MM-DD.json.
Advisory only — halt_and_review skips execution but doesn't require human unlock.

Learning loop:
- At the start of each debate, scores any pending past verdicts
- Rebuilds agent credibility tracker
- Agents and judge then see that credibility data in their prompts

Usage:
    python3 -c "from debate.debate_runner import run_debate; run_debate({...})"
"""

import json
import os
from datetime import date, datetime
from pathlib import Path

from debate.agents import run_bull_agent, run_bear_agent, run_devils_advocate, run_regime_specialist, run_quant_sanity_check
from debate.judge import run_judge
from debate.outcome_tracker import score_pending_verdicts
from simulation.scenario_simulator import run_all_scenarios
from ascent.reporting.debrief import run_pending_debriefs, load_debrief_context
from ascent.reporting.blind_spot_detector import detect_blind_spots, load_blind_spot_context

DEBATE_LOG_DIR   = Path("outputs/debate_log")
MULTI_AGENT_LOG  = Path("logs/multi_agent_run.jsonl")
HALT_STATE_PATH  = Path("execution/halt_state.json")


def load_latest_run_state(as_of_date=None) -> dict:
    if not MULTI_AGENT_LOG.exists():
        raise FileNotFoundError(f"No run log at {MULTI_AGENT_LOG}. Run run_all_agents.py first.")
    runs = []
    with open(MULTI_AGENT_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                runs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not runs:
        raise ValueError("Run log exists but is empty.")
    if as_of_date:
        cutoff = as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else str(as_of_date)
        eligible = [r for r in runs if r.get("date", "") <= cutoff]
        if not eligible:
            raise ValueError(f"No entries on or before {as_of_date}. Earliest: {runs[0].get('date','?')}")
        run = eligible[-1]
    else:
        run = runs[-1]
    data_date  = run.get("date", "unknown")
    weights    = run.get("weights", {})
    agents_raw = run.get("agents", {})
    allocation = run.get("allocation", {})
    def _regime(key):
        v = agents_raw.get(key, {})
        return v.get("regime", "unknown") if isinstance(v, dict) else "unknown"
    print(f"[Debate] Auto-loaded run data as of: {data_date}  ({len(weights)} positions)")
    return {
        "date": data_date, "us_regime": _regime("us_equities"),
        "macro_regime": _regime("macro"), "n_positions": len(weights),
        "allocation": allocation, "weights": weights, "_data_as_of": data_date,
    }


def run_debate(portfolio_state: dict = None, run_date: date = None, as_of_date=None) -> dict:
    """
    Run the full pre-rebalance debate session.

    Args:
        portfolio_state: Dict with keys:
            date, us_regime, macro_regime, n_positions, allocation, weights
        run_date: Override for testing. Defaults to today.

    Returns:
        Verdict dict from the judge:
            confidence, recommendation, key_risks, reasoning
    """
    run_date = run_date or date.today()

    if portfolio_state is None:
        portfolio_state = load_latest_run_state(as_of_date=as_of_date)

    data_as_of = portfolio_state.get("_data_as_of", portfolio_state.get("date", "unknown"))
    portfolio_state["_data_as_of"] = data_as_of

    print(f"\n{'='*60}")
    print(f"[Debate] Pre-rebalance debate | run_date={run_date}")
    print(f"[Debate] Data as of: {data_as_of}")
    if data_as_of != run_date.isoformat():
        print(f"[Debate] NOTE: agents reason using info as of {data_as_of}, not today.")
    print(f"{'='*60}")

    # Score any past verdicts that are now old enough to evaluate
    try:
        scored = score_pending_verdicts()
        if scored:
            print(f"[Debate] Scored {scored} pending verdict(s) — agent credibility updated")
        else:
            print("[Debate] No new verdicts to score")
    except Exception as e:
        print(f"[Debate] Outcome scoring failed (non-fatal): {e}")

    # Write debriefs for any scored verdicts that don't have one yet
    try:
        debriefed = run_pending_debriefs()
        if debriefed:
            print(f"[Debate] Wrote {debriefed} debrief(s) — lessons now in debate memory")
    except Exception as e:
        print(f"[Debate] Debrief generation failed (non-fatal): {e}")

    # Detect systematic blind spots across all past verdicts
    try:
        detect_blind_spots()
    except Exception as e:
        print(f"[Debate] Blind spot detection failed (non-fatal): {e}")

    # Inject blind spot context into portfolio state for agents
    blind_spot_context = load_blind_spot_context()
    if blind_spot_context:
        portfolio_state["blind_spot_context"] = blind_spot_context
        print(f"[Debate] Blind spot context injected into debate")

    # Run scenario sim before agents so devil's advocate has numbers
    print("[Debate] Running scenario simulation...")
    try:
        weights = portfolio_state.get("weights", {})
        if weights:
            scenario_results = run_all_scenarios(weights, run_date)
            worst_scenarios = sorted(scenario_results, key=lambda x: x["worst_case"])[:2]
            scenario_summary = " | ".join(
                f"{s['scenario']}: p5={s['worst_case']:+.1f}% p50={s['base_case']:+.1f}%"
                for s in worst_scenarios
            )
            portfolio_state["scenario_summary"] = scenario_summary
            portfolio_state["scenario_results"] = worst_scenarios
            print(f"[Debate] Worst scenarios: {scenario_summary}")
        else:
            portfolio_state["scenario_summary"] = ""
            portfolio_state["scenario_results"] = []
            print("[Debate] No weights — skipping scenario sim")
    except Exception as e:
        portfolio_state["scenario_summary"] = ""
        portfolio_state["scenario_results"] = []
        print(f"[Debate] Scenario sim failed (non-fatal): {e}")

    # Bull agent
    print("[Debate] Bull agent arguing...")
    try:
        bull = run_bull_agent(portfolio_state)
        print(f"[Debate] Bull: {bull[:120]}...")
    except Exception as e:
        bull = f"Bull agent failed: {e}"
        print(f"[Debate] Bull FAILED: {e}")

    # Bear agent
    print("[Debate] Bear agent arguing...")
    try:
        bear = run_bear_agent(portfolio_state)
        print(f"[Debate] Bear: {bear[:120]}...")
    except Exception as e:
        bear = f"Bear agent failed: {e}"
        print(f"[Debate] Bear FAILED: {e}")

    # Devil's advocate
    print("[Debate] Devil's advocate arguing...")
    try:
        devil = run_devils_advocate(portfolio_state)
        print(f"[Debate] Devil: {devil[:120]}...")
    except Exception as e:
        devil = f"Devil's advocate failed: {e}"
        print(f"[Debate] Devil FAILED: {e}")

    # Regime specialist
    print("[Debate] Regime specialist arguing...")
    try:
        regime_arg = run_regime_specialist(portfolio_state)
        print(f"[Debate] Regime: {regime_arg[:120]}...")
    except Exception as e:
        regime_arg = f"Regime specialist failed: {e}"
        print(f"[Debate] Regime FAILED: {e}")

    # Quant sanity check (no LLM — pure Python)
    print("[Debate] Quant sanity check running...")
    try:
        quant_check = run_quant_sanity_check(portfolio_state)
        print(f"[Debate] Quant: {quant_check[:120]}...")
    except Exception as e:
        quant_check = f"Quant sanity check failed: {e}"
        print(f"[Debate] Quant FAILED: {e}")

    # Judge synthesizes — now with memory
    print("[Debate] Judge synthesizing verdict...")
    verdict = run_judge(bull, bear, devil, portfolio_state, regime_arg=regime_arg, quant_check=quant_check)

    print(f"\n[Debate] VERDICT: {verdict['recommendation'].upper()} "
          f"(confidence: {verdict.get('confidence', '?')})")
    for risk in verdict.get("key_risks", []):
        print(f"  Risk: {risk}")
    if verdict.get("reasoning"):
        print(f"  Reasoning: {verdict['reasoning'][:150]}...")

    # Save full debate record
    record = {
        "date":            run_date.isoformat(),
        "data_as_of":      data_as_of,
        "timestamp":       datetime.now().isoformat(),
        "portfolio_state": portfolio_state,
        "arguments": {
            "bull":             bull,
            "bear":             bear,
            "devils_advocate":  devil,
            "regime_specialist": regime_arg,
            "quant_sanity":     quant_check,
        },
        "verdict":         verdict,
        "outcome_scored":  False,  # will be updated by outcome_tracker in a future run
    }

    os.makedirs(DEBATE_LOG_DIR, exist_ok=True)
    out_path = DEBATE_LOG_DIR / f"verdict_{run_date.isoformat()}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"[Debate] Full record written to {out_path}")

    # Write persistent halt state if verdict requires human override
    if verdict.get("recommendation") == "halt_and_review":
        halt_record = {
            "halted":          True,
            "halt_date":       run_date.isoformat(),
            "reason":          verdict.get("reasoning", "")[:200],
            "key_risks":       verdict.get("key_risks", []),
            "verdict_path":    str(out_path),
            "requires_override": True,
            "created_at":      datetime.now().isoformat(),
        }
        HALT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        HALT_STATE_PATH.write_text(json.dumps(halt_record, indent=2))
        print(f"[Debate] HALT state written to {HALT_STATE_PATH} — "
              "create execution/halt_override.json to resume trading")

    return verdict
