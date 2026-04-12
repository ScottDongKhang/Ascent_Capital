"""
run_all_agents.py
Top-level daily runner for the Ascent Capital multi-agent platform.

Non-rebalance day:  agents → orchestrator → write weights → log (no debate, no execution)
Rebalance day:      agents → orchestrator → write weights → debate → execute via eod_runner

Usage:
    python3 run_all_agents.py                # live execution
    python3 run_all_agents.py --dry-run      # no order submission
"""

import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path

from agents.us_equities_agent import run_us_equities_agent
from agents.macro_agent import run_macro_agent
from orchestrator.central_intelligence import run_orchestrator
from ascent.monitoring.skill_tracker import export_skill_scores
from ascent.monitoring.forward_pnl_tracker import run_forward_pnl_cycle
from ascent.monitoring.pre_rebalance_checklist import run_checklist

try:
    from agents.international_agent import run_international_agent
    _HAS_INTERNATIONAL = True
except ImportError:
    _HAS_INTERNATIONAL = False

try:
    from agents.alternatives_agent import run_alternatives_agent
    _HAS_ALTERNATIVES = True
except ImportError:
    _HAS_ALTERNATIVES = False


def main():
    dry_run = "--dry-run" in sys.argv
    today   = date.today()

    print(f"\n{'#'*60}")
    print(f"# ASCENT CAPITAL — Multi-Agent Daily Run")
    print(f"# Date:  {today}")
    print(f"# Mode:  {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'#'*60}\n")

    # ── Step 1: Run all agents in parallel ───────────────────────────────────
    agent_tasks = [
        ("us_equities", run_us_equities_agent),
        ("macro",       run_macro_agent),
    ]
    if _HAS_INTERNATIONAL:
        agent_tasks.append(("international", run_international_agent))
    if _HAS_ALTERNATIVES:
        agent_tasks.append(("alternatives", run_alternatives_agent))

    agent_outputs = []
    with ThreadPoolExecutor(max_workers=len(agent_tasks)) as executor:
        futures = {
            executor.submit(fn, dry_run, today): name
            for name, fn in agent_tasks
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                agent_outputs.append(future.result())
            except Exception as e:
                print(f"[Runner] {name} agent FAILED: {e}")

    if not agent_outputs:
        print("[Runner] No agent outputs — aborting")
        return

    # ── Step 1b: Pre-rebalance checklist (rebalance days only) ───────────────
    from pathlib import Path as _Path
    import pandas as _pd
    cal_path     = _Path("rebalance_calendar.csv")
    is_rebalance = False
    if cal_path.exists():
        try:
            _cal         = _pd.read_csv(cal_path)
            is_rebalance = today.isoformat() in _cal["rebalance_date"].values
        except Exception:
            pass

    if is_rebalance:
        try:
            held_symbols = []
            try:
                from ascent.execution.alpaca_broker import get_positions
                pos = get_positions()
                if not pos.empty and "symbol" in pos.columns:
                    held_symbols = list(pos["symbol"])
            except Exception:
                pass

            checklist = run_checklist(
                agent_outputs=agent_outputs,
                held_symbols=held_symbols,
            )
            if not checklist.passed:
                print("[Runner] ✗ Checklist has blocking failures — aborting execution")
                print("[Runner] Review logs/checklist_log.jsonl for details")
                return
        except Exception as e:
            print(f"[Runner] Checklist failed ({e}) — proceeding anyway")
    # ── Steps 2/3/4: Sequential pipeline — DO NOT parallelize ───────────────
    # Each step depends on the previous: PnL log → skill scores → orchestrator.
    # run_forward_pnl_cycle writes today's NAV to the PnL log.
    # export_skill_scores reads that log to compute the 63-day rolling Sharpe.
    # run_orchestrator reads the fresh Sharpe to weight capital allocation.
    try:
        run_forward_pnl_cycle(agent_outputs, today=today)
    except Exception as e:
        print(f"[Runner] Forward PnL cycle failed: {e} — continuing")

    try:
        export_skill_scores()
    except Exception as e:
        print(f"[Runner] Skill score update failed: {e} — continuing with stale scores")

    # ── Step 5: Run orchestrator (reads fresh skill scores written above) ─────
    merged_weights = run_orchestrator(agent_outputs)

    if not merged_weights:
        print("[Runner] Orchestrator returned empty weights — aborting execution")
        return

    # ── Step 4: Write merged weights to file ──────────────────────────────────
    weights_path = Path("execution/merged_weights.json")
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "date":         today.isoformat(),
        "weights":      merged_weights,
        "agents":       [ao.agent_id for ao in agent_outputs],
        "generated_at": datetime.now().isoformat(),
    }
    with open(weights_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[Runner] Merged weights written to {weights_path}")
    print(f"[Runner] {len(merged_weights)} positions, total weight: {sum(merged_weights.values()):.4f}")

    # ── Non-rebalance day: stop here ──────────────────────────────────────────
    if not is_rebalance:
        print("[Runner] Non-rebalance day — weights updated, no debate, no execution.")
        _log_run(today, merged_weights, agent_outputs, dry_run)
        return

    # ── Rebalance day: debate → execute ───────────────────────────────────────
    print(f"\n[Runner] Rebalance day — running debate layer...")
    verdict = None
    try:
        from debate.debate_runner import run_debate
        import json as _json
        from pathlib import Path as _Path
        _regime_path  = _Path("dashboard/regime_signal.json")
        _saved_regime = "unknown"
        try:
            _rdata        = _json.loads(_regime_path.read_text())
            _saved_regime = (_rdata[-1] if isinstance(_rdata, list) else _rdata).get("label", "unknown")
        except Exception:
            pass
        portfolio_state = {
            "date":         today.isoformat(),
            "us_regime":    next((ao.regime_signal for ao in agent_outputs if ao.agent_id == "us_equities" and ao.regime_signal), _saved_regime),
            "macro_regime": next((ao.regime_signal for ao in agent_outputs if ao.agent_id == "macro" and ao.regime_signal), "unknown"),
            "n_positions":  len(merged_weights),
            "allocation":   {ao.agent_id: round(
                next((v for k, v in {
                    "us_equities": 0.60, "macro": 0.15,
                    "international": 0.15, "alternatives": 0.10
                }.items() if k == ao.agent_id), 0.0), 2)
                for ao in agent_outputs},
            "weights":      merged_weights,
        }
        verdict = run_debate(portfolio_state, run_date=today)

        if verdict.get("recommendation") == "halt_and_review":
            print("[Runner] DEBATE VERDICT: halt_and_review — skipping execution")
            print("[Runner] Review at outputs/debate_log/")
            _log_run(today, merged_weights, agent_outputs, dry_run)
            return

    except Exception as e:
        print(f"[Runner] Debate failed ({e}) — proceeding to execution anyway")

    # ── Step 5: Execute via eod_runner ────────────────────────────────────────
    if dry_run:
        print(f"\n[Runner] DRY RUN — would submit {len(merged_weights)} positions:")
        for sym, w in sorted(merged_weights.items(), key=lambda x: -x[1])[:15]:
            print(f"  {sym}: {w:.2%}")
    else:
        print("[Runner] LIVE MODE — calling eod_runner with merged weights")
        try:
            from ascent.execution.eod_runner import run_eod_with_weights
            run_eod_with_weights(merged_weights, run_date=today, dry_run=False, precomputed_verdict=verdict)
        except ImportError:
            print("[Runner] WARNING: run_eod_with_weights not yet available — weights file written only")
        except Exception as e:
            print(f"[Runner] Execution failed: {e}")

    # ── Step 6: Log the run ───────────────────────────────────────────────────
    _log_run(today, merged_weights, agent_outputs, dry_run)


def _log_run(today, merged_weights, agent_outputs, dry_run):
    def _regime_str(val):
        if val is None:
            return "unknown"
        return str(val).split(".")[-1].lower() if "." in str(val) else str(val)

    run_log = {
        "date":             today.isoformat(),
        "weights":          merged_weights,
        "agents": {
            ao.agent_id: {
                "n_positions": ao.n_positions if hasattr(ao, "n_positions") else len(getattr(ao, "target_weights", {})),
                "regime":      _regime_str(ao.regime_signal),
            }
            for ao in agent_outputs
        },
        "allocation": {
            ao.agent_id: round(
                {"us_equities": 0.60, "macro": 0.15, "international": 0.15, "alternatives": 0.10}.get(ao.agent_id, 0.0), 2
            )
            for ao in agent_outputs
        },
        "merged_positions": len(merged_weights),
        "mode":             "dry_run" if dry_run else "live",
        "timestamp":        datetime.now().isoformat(),
    }

    log_path = Path("logs/multi_agent_run.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(run_log) + "\n")

    print(f"\n[Runner] Run logged to {log_path}")
    print(f"[Runner] Done.\n")


if __name__ == "__main__":
    main()
