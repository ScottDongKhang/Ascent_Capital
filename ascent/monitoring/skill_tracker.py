"""
ascent/monitoring/skill_tracker.py
Per-agent rolling Sharpe tracker.

Reads each agent's daily PnL from logs, computes 63-day rolling Sharpe,
writes to dashboard/agent_skill_scores.json.
Consumed by the orchestrator for capital allocation decisions.

Usage:
    python3 -m ascent.monitoring.skill_tracker
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

SKILL_LOG_PATH    = Path("logs/skill_scores_log.jsonl")
SKILL_OUTPUT_PATH = Path("dashboard/agent_skill_scores.json")
ROLLING_WINDOW    = 63  # trading days


# ── PnL loading ────────────────────────────────────────────────────────────────

def _load_agent_pnl(agent_id: str) -> pd.Series:
    """
    Load daily returns for a specific agent from its PnL log.

    us_equities: reads portfolio_value from logs/eod_log.jsonl
    macro:       reads nav from logs/macro_pnl.jsonl
    others:      reads nav from logs/{agent_id}_pnl.jsonl
    """
    if agent_id == "us_equities":
        log_path = Path("logs/eod_log.jsonl")
    else:
        log_path = Path(f"logs/{agent_id}_pnl.jsonl")

    if not log_path.exists():
        return pd.Series(dtype=float)

    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                d   = entry.get("date") or entry.get("run_date")
                nav = entry.get("portfolio_value") or entry.get("nav")
                if d and nav:
                    records.append({"date": pd.Timestamp(d), "nav": float(nav)})
            except (json.JSONDecodeError, ValueError):
                continue

    if not records:
        return pd.Series(dtype=float)

    df = (pd.DataFrame(records)
            .set_index("date")
            .sort_index())
    df = df[~df.index.duplicated(keep="last")]
    returns = df["nav"].pct_change().dropna()
    return returns


# ── Sharpe computation ─────────────────────────────────────────────────────────

def compute_rolling_sharpe(returns: pd.Series, window: int = ROLLING_WINDOW) -> float:
    """
    Annualized Sharpe over the most recent `window` trading days.
    Returns None if insufficient data.
    """
    if len(returns) < window:
        return None

    recent  = returns.iloc[-window:]
    mean_r  = recent.mean()
    std_r   = recent.std()

    if std_r == 0 or np.isnan(std_r):
        return 0.0

    return float(mean_r / std_r * np.sqrt(252))


# ── All agents ─────────────────────────────────────────────────────────────────

def compute_all_skill_scores(agent_ids: list = None) -> dict:
    """
    Compute skill scores for all registered agents.
    Returns dict: {agent_id: {"skill_score", "n_days", "status", "latest_date"}}
    """
    if agent_ids is None:
        agent_ids = ["us_equities", "macro", "international", "alternatives"]

    scores = {}
    for agent_id in agent_ids:
        returns = _load_agent_pnl(agent_id)
        n_days  = len(returns)

        if n_days < 10:
            scores[agent_id] = {
                "skill_score":  None,
                "n_days":       n_days,
                "status":       "insufficient_data",
                "latest_date":  returns.index[-1].isoformat() if n_days > 0 else None,
            }
            continue

        sharpe = compute_rolling_sharpe(returns)
        scores[agent_id] = {
            "skill_score":  round(sharpe, 4) if sharpe is not None else None,
            "n_days":       n_days,
            "status":       "active" if sharpe is not None else "warming_up",
            "latest_date":  returns.index[-1].isoformat(),
        }

    return scores


# ── Export ─────────────────────────────────────────────────────────────────────

def export_skill_scores(agent_ids: list = None) -> dict:
    """
    Main entry point. Compute scores and write to JSON + JSONL log.
    Returns the scores dict.
    """
    scores = compute_all_skill_scores(agent_ids)

    payload = {
        "generated_at":      datetime.now().isoformat(),
        "skill_score_as_of": datetime.now().date().isoformat(),
        "agents":            scores,
    }

    os.makedirs(SKILL_OUTPUT_PATH.parent, exist_ok=True)
    with open(SKILL_OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    os.makedirs(SKILL_LOG_PATH.parent, exist_ok=True)
    log_entry = {"timestamp": datetime.now().isoformat(), **scores}
    with open(SKILL_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"[SkillTracker] Agent skill scores ({datetime.now().date()}):")
    for agent_id, data in scores.items():
        score_str = f"{data['skill_score']:.3f}" if data["skill_score"] is not None else "N/A"
        print(f"  {agent_id}: Sharpe={score_str} | {data['n_days']} days | {data['status']}")

    return scores


if __name__ == "__main__":
    export_skill_scores()
