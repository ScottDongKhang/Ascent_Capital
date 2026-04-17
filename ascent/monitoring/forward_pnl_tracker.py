"""
ascent/monitoring/forward_pnl_tracker.py
Forward PnL tracker for specialist agents.

True OOS performance tracking:
  1. At the start of each run, load yesterday's proposed weights per agent
  2. Fetch today's prices
  3. Compute what those weights actually returned (close-to-close)
  4. Write the return + new NAV to each agent's PnL log
  5. Save today's proposed weights as the snapshot for tomorrow

This gives the skill tracker honest forward-looking returns,
not in-sample backtest returns.

Called by run_all_agents.py after agents run, before skill scores update.

Snapshot files: logs/snapshots/{agent_id}_weights_{YYYY-MM-DD}.json
PnL logs:
    us_equities → logs/us_equities_pnl.jsonl
    macro       → logs/macro_pnl.jsonl
    international → logs/international_pnl.jsonl
    alternatives  → logs/alternatives_pnl.jsonl
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

SNAPSHOT_DIR = Path("logs/snapshots")
PNL_LOGS = {
    "us_equities":   Path("logs/us_equities_pnl.jsonl"),
    "macro":         Path("logs/macro_pnl.jsonl"),
    "international": Path("logs/international_pnl.jsonl"),
    "alternatives":  Path("logs/alternatives_pnl.jsonl"),
}
STARTING_NAV = 100_000.0  # normalized base NAV for agents without real capital


# ── Price fetching ─────────────────────────────────────────────────────────────

def _fetch_latest_returns(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch yesterday→today close-to-close returns for a list of symbols.
    Returns {symbol: return} or empty dict on failure.
    """
    if not symbols:
        return {}
    try:
        import yfinance as yf
        import pandas as pd

        # Need 3 days of data to ensure we get 2 closes
        raw = yf.download(
            symbols,
            period="5d",
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            return {}

        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
        elif "Close" in raw.columns:
            closes = raw[["Close"]]
            closes.columns = symbols[:1]
        else:
            closes = raw

        closes = closes.reindex(columns=symbols).dropna(how="all")
        if len(closes) < 2:
            return {}

        # Latest close-to-close return
        today_close = closes.iloc[-1]
        prev_close  = closes.iloc[-2]

        returns = {}
        for sym in symbols:
            t = today_close.get(sym)
            p = prev_close.get(sym)
            if t is not None and p is not None and p != 0:
                returns[sym] = float((t - p) / p)

        return returns

    except Exception as e:
        print(f"[ForwardPnL] Price fetch failed: {e}")
        return {}


# ── NAV loading ────────────────────────────────────────────────────────────────

def _load_latest_nav(agent_id: str) -> float:
    """Load the most recent NAV from agent's PnL log."""
    log_path = PNL_LOGS.get(agent_id)
    if not log_path or not log_path.exists():
        return STARTING_NAV

    latest_nav = STARTING_NAV
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
            nav   = entry.get("nav") or entry.get("portfolio_value")  # portfolio_value = legacy key
            if nav:
                latest_nav = float(nav)
        except Exception:
            pass

    return latest_nav


# ── Snapshot management ────────────────────────────────────────────────────────

def save_weight_snapshot(agent_id: str, weights: Dict[str, float], run_date: date):
    """Save today's proposed weights as a snapshot for tomorrow's scoring."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{agent_id}_weights_{run_date.isoformat()}.json"
    payload = {
        "agent_id":   agent_id,
        "date":       run_date.isoformat(),
        "weights":    weights,
        "saved_at":   datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _load_previous_snapshot(agent_id: str, as_of: date) -> Optional[dict]:
    """
    Load the most recent weight snapshot before as_of date.
    Looks back up to 5 trading days to handle weekends/holidays.
    """
    if not SNAPSHOT_DIR.exists():
        return None

    for days_back in range(1, 6):
        prev_date = as_of - timedelta(days=days_back)
        path = SNAPSHOT_DIR / f"{agent_id}_weights_{prev_date.isoformat()}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass

    return None


# ── PnL writing ────────────────────────────────────────────────────────────────

def _write_pnl_entry(agent_id: str, run_date: date, nav: float, daily_return: float):
    """Append a PnL entry to the agent's log."""
    log_path = PNL_LOGS.get(agent_id)
    if not log_path:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "date":      run_date.isoformat(),
        "nav":       round(nav, 2),
        "return":    round(daily_return, 6),
        "agent_id":  agent_id,
        "source":    "forward_pnl_tracker",
        "timestamp": datetime.now().isoformat(),
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Main scoring function ──────────────────────────────────────────────────────

def score_yesterday_weights(
    agent_id: str,
    today: date,
    prefetched_returns: Optional[Dict[str, float]] = None,
) -> Optional[float]:
    """
    Score yesterday's proposed weights against today's price move.

    Args:
        agent_id:           Agent to score.
        today:              Date of scoring.
        prefetched_returns: Pre-fetched {symbol: return} dict. If provided,
                            skips the yfinance call (used for batched fetching).

    Returns:
        Daily portfolio return, or None if no snapshot found.
    """
    snapshot = _load_previous_snapshot(agent_id, today)
    if not snapshot:
        return None

    weights: Dict[str, float] = snapshot.get("weights", {})
    if not weights:
        return None

    snap_date = snapshot.get("date", "unknown")
    symbols   = list(weights.keys())

    if prefetched_returns is not None:
        returns = prefetched_returns
    else:
        returns = _fetch_latest_returns(symbols)

    if not returns:
        print(f"[ForwardPnL] {agent_id}: no price returns available for scoring")
        return None

    # Weighted portfolio return
    portfolio_return = sum(
        weights.get(sym, 0.0) * returns.get(sym, 0.0)
        for sym in symbols
    )

    # Update NAV
    prev_nav  = _load_latest_nav(agent_id)
    new_nav   = prev_nav * (1 + portfolio_return)

    _write_pnl_entry(agent_id, today, new_nav, portfolio_return)

    print(
        f"[ForwardPnL] {agent_id}: {portfolio_return:+.2%} "
        f"(NAV {prev_nav:,.0f} → {new_nav:,.0f}, weights from {snap_date})"
    )

    return portfolio_return


# ── Main entry point ───────────────────────────────────────────────────────────

def run_forward_pnl_cycle(agent_outputs: list, today: date = None) -> dict:
    """
    Full forward PnL cycle:
      1. Score yesterday's weights for all agents
      2. Save today's weights as tomorrow's snapshot

    Args:
        agent_outputs: List of AgentOutput from today's run
        today:         Date override for testing

    Returns:
        Dict {agent_id: daily_return} for agents that were scored
    """
    today = today or date.today()
    results = {}

    print(f"\n[ForwardPnL] Running forward PnL cycle | {today}")

    # Step 1 — batch-fetch all symbols across all agents (one yfinance call)
    agent_ids = [ao.agent_id for ao in agent_outputs]
    all_snapshots: Dict[str, Optional[dict]] = {
        aid: _load_previous_snapshot(aid, today) for aid in agent_ids
    }
    all_symbols: set = set()
    for snap in all_snapshots.values():
        if snap:
            all_symbols.update(snap.get("weights", {}).keys())

    all_symbols.add("SPY")  # always fetch SPY for benchmark
    batch_returns = _fetch_latest_returns(list(all_symbols))
    print(f"[ForwardPnL] Batch-fetched returns for {len(all_symbols)} symbols in one call")
    spy_return = batch_returns.get("SPY", 0.0)
    print(f"[ForwardPnL] SPY benchmark: {spy_return:+.2%}")

    # Score each agent using the pre-fetched returns
    for agent_id in agent_ids:
        try:
            ret = score_yesterday_weights(agent_id, today, prefetched_returns=batch_returns or None)
            if ret is not None:
                results[agent_id] = ret
        except Exception as e:
            print(f"[ForwardPnL] {agent_id} scoring failed: {e}")

    if not results:
        print("[ForwardPnL] No agents scored today (first run or no snapshots yet)")

    # Step 2 — save today's weights as tomorrow's snapshot
    saved = 0
    for ao in agent_outputs:
        if ao.target_weights:
            try:
                save_weight_snapshot(ao.agent_id, ao.target_weights, today)
                saved += 1
            except Exception as e:
                print(f"[ForwardPnL] Failed to save snapshot for {ao.agent_id}: {e}")

    print(f"[ForwardPnL] Saved {saved} weight snapshot(s) for tomorrow's scoring")

    return {"agent_returns": results, "spy_return": batch_returns.get("SPY", 0.0)}
