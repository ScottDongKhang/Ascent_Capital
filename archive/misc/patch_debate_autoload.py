"""
Patches two files:
1. run_all_agents.py  — write full state (weights, regimes, allocation) to run log
2. debate/debate_runner.py — auto-load from run log instead of requiring manual portfolio_state

Run from project root:
    python3 patch_debate_autoload.py
"""

import ast
from pathlib import Path

BASE = Path("/Users/kdong/Downloads/ascent capital v2 up to phase 5.1")

# ── PATCH 1: run_all_agents.py ────────────────────────────────────────────────

RAL = BASE / "run_all_agents.py"
src = RAL.read_text()

OLD_LOG = '    run_log = {\n        "date":              today.isoformat(),\n        "agents":            {ao.agent_id: ao.n_positions for ao in agent_outputs},\n        "merged_positions":  len(merged_weights),\n        "mode":              "dry_run" if dry_run else "live",\n        "timestamp":         datetime.now().isoformat(),\n    }'

NEW_LOG = '''    def _regime_str(val):
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
    }'''

if OLD_LOG not in src:
    print("PATCH 1 FAILED: run_log block not found — check indentation in run_all_agents.py")
    raise SystemExit(1)

src = src.replace(OLD_LOG, NEW_LOG, 1)
ast.parse(src)
RAL.write_text(src)
print("PATCH 1 done: run_all_agents.py")

# ── PATCH 2a: debate_runner.py — add loader function ─────────────────────────

DR = BASE / "debate" / "debate_runner.py"
src2 = DR.read_text()

OLD_IMPORTS = 'DEBATE_LOG_DIR = Path("outputs/debate_log")'

NEW_IMPORTS = '''DEBATE_LOG_DIR   = Path("outputs/debate_log")
MULTI_AGENT_LOG  = Path("logs/multi_agent_run.jsonl")


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
    }'''

if OLD_IMPORTS not in src2:
    print("PATCH 2a FAILED: DEBATE_LOG_DIR line not found in debate_runner.py")
    raise SystemExit(1)
src2 = src2.replace(OLD_IMPORTS, NEW_IMPORTS, 1)

# ── PATCH 2b: replace function signature ─────────────────────────────────────

OLD_SIG = 'def run_debate(portfolio_state: dict, run_date: date = None) -> dict:'
NEW_SIG  = 'def run_debate(portfolio_state: dict = None, run_date: date = None, as_of_date=None) -> dict:'

if OLD_SIG not in src2:
    print("PATCH 2b FAILED: run_debate signature not found")
    raise SystemExit(1)
src2 = src2.replace(OLD_SIG, NEW_SIG, 1)

# ── PATCH 2c: inject auto-load + header right after run_date line ─────────────

OLD_RUNDATE = '    run_date = run_date or date.today()\n\n    print(f"\\n{\'=\'*60}")\n    print(f"[Debate] Pre-rebalance debate | {run_date}")\n    print(f"{\'=\'*60}")'

NEW_RUNDATE = '''    run_date = run_date or date.today()

    if portfolio_state is None:
        portfolio_state = load_latest_run_state(as_of_date=as_of_date)

    data_as_of = portfolio_state.get("_data_as_of", portfolio_state.get("date", "unknown"))
    portfolio_state["_data_as_of"] = data_as_of

    print(f"\\n{'='*60}")
    print(f"[Debate] Pre-rebalance debate | run_date={run_date}")
    print(f"[Debate] Data as of: {data_as_of}")
    if data_as_of != run_date.isoformat():
        print(f"[Debate] NOTE: agents reason using info as of {data_as_of}, not today.")
    print(f"{'='*60}")'''

if OLD_RUNDATE not in src2:
    print("PATCH 2c FAILED: run_date block not found — will try alternate whitespace")
    raise SystemExit(1)
src2 = src2.replace(OLD_RUNDATE, NEW_RUNDATE, 1)

# ── PATCH 2d: stamp data_as_of into saved record ─────────────────────────────

OLD_RECORD = '        "date":            run_date.isoformat(),\n        "timestamp":       datetime.now().isoformat(),'
NEW_RECORD  = '        "date":            run_date.isoformat(),\n        "data_as_of":      data_as_of,\n        "timestamp":       datetime.now().isoformat(),'

if OLD_RECORD not in src2:
    print("PATCH 2d FAILED: record dict not found")
    raise SystemExit(1)
src2 = src2.replace(OLD_RECORD, NEW_RECORD, 1)

ast.parse(src2)
DR.write_text(src2)
print("PATCH 2 done: debate/debate_runner.py")

print("\nAll patches applied.")
print('Run: python3 -c "from debate.debate_runner import run_debate; run_debate()"')
