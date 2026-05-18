"""
memory/regime_memory.py
Regime-aware episodic memory — logs portfolio episodes and realized outcomes.

Each episode records the regime + portfolio weights at run time. After 21
calendar days the realized 21-day portfolio return is filled in via
update_outcomes(). The AI PM queries this to calibrate expected returns
before proposing a new portfolio.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

EPISODE_LOG = Path("logs/regime_episodes.jsonl")


def log_episode(
    run_date: str,
    regime: str,
    quant_weights: dict,
    ai_weights: Optional[dict] = None,
) -> None:
    """
    Append a new episode to EPISODE_LOG.

    Skips silently if an entry for run_date already exists (idempotent
    — safe to call multiple times per day).
    """
    EPISODE_LOG.parent.mkdir(parents=True, exist_ok=True)

    # Deduplicate: read existing dates
    existing_dates: set[str] = set()
    if EPISODE_LOG.exists():
        with open(EPISODE_LOG) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "date" in row:
                        existing_dates.add(row["date"])
                except Exception:
                    pass

    if run_date in existing_dates:
        return

    entry = {
        "date": run_date,
        "regime": regime,
        "quant_weights": quant_weights,
        "ai_weights": ai_weights,
        "realized_return_21d": None,
    }

    with open(EPISODE_LOG, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def query_episodes(regime: str, n: int = 5) -> str:
    """
    Return last n episodes where the stored regime starts with the queried
    regime prefix (case-insensitive). Exact match also works.

    Returns formatted plain text for LLM consumption:
      Episode 2026-04-15 (calm_bull): quant top5=AAPL=5.0%,... | realized 21d return: +1.23%
    Returns "No episodes found for regime '{regime}'." if empty.
    """
    regime_lower = regime.strip().lower()

    if not EPISODE_LOG.exists():
        return f"No episodes found for regime '{regime}'."

    matching: list[dict] = []
    with open(EPISODE_LOG) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                stored = str(row.get("regime", "")).lower()
                if stored.startswith(regime_lower):
                    matching.append(row)
            except Exception:
                pass

    # Take last n (most recent by file order)
    matching = matching[-n:]

    if not matching:
        return f"No episodes found for regime '{regime}'."

    lines: list[str] = []
    for ep in reversed(matching):  # newest first
        ep_date = ep.get("date", "?")
        ep_regime = ep.get("regime", "?")
        qw = ep.get("quant_weights") or {}
        top5 = sorted(qw.items(), key=lambda x: -x[1])[:5]
        top5_str = ",".join(f"{s}={v * 100:.1f}%" for s, v in top5) if top5 else "none"
        realized = ep.get("realized_return_21d")
        if realized is None:
            ret_str = "pending"
        else:
            ret_str = f"{realized:+.2%}"
        lines.append(
            f"Episode {ep_date} ({ep_regime}): quant top5={top5_str} | realized 21d return: {ret_str}"
        )

    return "\n".join(lines)


def update_outcomes(price_returns: dict[str, float]) -> int:
    """
    For each episode where realized_return_21d is None and the episode is
    >= 21 calendar days old, compute:

        realized_return_21d = sum(weight * price_returns[symbol])

    price_returns is a dict of symbol -> total return since episode date
    (NOT daily — total cumulative return). Symbols not present in
    price_returns are treated as 0.

    Rewrites EPISODE_LOG in-place. Returns the count of episodes updated.
    """
    if not EPISODE_LOG.exists():
        return 0

    today = date.today()
    rows: list[dict] = []
    with open(EPISODE_LOG) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                rows.append({"_raw": line})

    n_updated = 0
    for row in rows:
        if "_raw" in row:
            continue
        if row.get("realized_return_21d") is not None:
            continue
        ep_date_str = row.get("date", "")
        if not ep_date_str:
            continue
        try:
            ep_date = date.fromisoformat(ep_date_str)
        except ValueError:
            continue

        if (today - ep_date).days < 21:
            continue

        # Compute weighted return
        qw = row.get("quant_weights") or {}
        total_w = sum(qw.values()) or 1.0
        realized = sum(
            (w / total_w) * price_returns.get(sym, 0.0)
            for sym, w in qw.items()
        )
        row["realized_return_21d"] = realized
        n_updated += 1

    if n_updated > 0:
        tmp = EPISODE_LOG.parent / (EPISODE_LOG.name + ".tmp")
        with open(tmp, "w") as fh:
            for row in rows:
                if "_raw" in row:
                    fh.write(row["_raw"] + "\n")
                else:
                    fh.write(json.dumps(row) + "\n")
        os.replace(tmp, EPISODE_LOG)

    return n_updated
