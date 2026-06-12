"""
One-time backfill of counterfactual tracking data.

Writes:
  logs/counterfactual_quant_star_snapshots.jsonl  — one entry per rebalance date
  logs/counterfactual_ai_snapshots.jsonl          — one entry per AI PM thesis date
  logs/counterfactual_daily.jsonl                 — rewritten with real Track A★/D/C returns

Run: .venv/bin/python scripts/backfill_counterfactual.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load_prices() -> pd.DataFrame:
    """Return wide daily close prices (date × symbol), date as string YYYY-MM-DD."""
    raw = pd.read_parquet(REPO / "data_cache" / "prices_live.parquet")
    wide = raw.pivot_table(index="date", columns="symbol", values="close")
    wide.index = pd.to_datetime(wide.index, utc=True).normalize().strftime("%Y-%m-%d")
    return wide.sort_index()


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def portfolio_return(weights: dict, returns_row: pd.Series) -> float:
    total = 0.0
    for sym, w in weights.items():
        if sym in returns_row.index:
            total += w * float(returns_row[sym])
    return total


def load_multi_agent() -> list[dict]:
    """Load multi_agent_run, keeping only the last entry per date."""
    by_date = {}
    for line in (REPO / "logs" / "multi_agent_run.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
            by_date[r["date"]] = r  # last write wins
        except Exception:
            pass
    return sorted(by_date.values(), key=lambda r: r["date"])


def load_rebalance_dates() -> set[str]:
    dates = set()
    for line in (REPO / "logs" / "eod_log.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
            if r.get("rebalanced"):
                dates.add(r["date"])
        except Exception:
            pass
    return dates


def load_ai_pm_theses() -> dict[str, dict]:
    """Returns {date_str: normalized_weights} for each thesis file."""
    result = {}
    thesis_dir = REPO / "outputs" / "ai_pm_theses"
    for f in sorted(thesis_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            date_str = f.stem[:10]  # YYYY-MM-DD from filename
            portfolio = data.get("ai_pm_portfolio", {})
            if not portfolio:
                continue
            # Values may be rank scores (e.g. 1-10) or actual weights — normalize either way
            total = sum(float(v) for v in portfolio.values())
            if total > 0:
                portfolio = {k: float(v) / total for k, v in portfolio.items()}
            result[date_str] = portfolio
        except Exception:
            pass
    return result


def load_existing_track_b() -> dict[str, float]:
    """Load correct Track B (actual Alpaca) returns from existing counterfactual_daily."""
    result = {}
    path = REPO / "logs" / "counterfactual_daily.jsonl"
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
            date = r.get("date")
            tb = r.get("track_b_return")
            if date and tb is not None and tb != 0.0:
                result[date] = tb  # last write wins (deduplication)
        except Exception:
            pass
    return result


def main():
    print("[Backfill] Loading prices...")
    prices = load_prices()
    rets = daily_returns(prices)

    print("[Backfill] Loading source data...")
    multi_agent = load_multi_agent()
    rebalance_dates = load_rebalance_dates()
    ai_pm_theses = load_ai_pm_theses()
    track_b = load_existing_track_b()

    print(f"[Backfill] {len(multi_agent)} daily runs | {len(rebalance_dates)} rebalances | {len(ai_pm_theses)} AI PM theses")
    print(f"[Backfill] Rebalance dates: {sorted(rebalance_dates)}")
    print(f"[Backfill] AI PM thesis dates: {sorted(ai_pm_theses.keys())}")

    # ── Write Track A★ snapshots (one per rebalance) ─────────────────────────
    astar_path = REPO / "logs" / "counterfactual_quant_star_snapshots.jsonl"
    # Build map: date → merged_weights from multi_agent_run
    weights_by_date = {r["date"]: r["weights"] for r in multi_agent}

    written_astar = []
    for d in sorted(rebalance_dates):
        if d in weights_by_date:
            written_astar.append({"date": d, "weights": weights_by_date[d]})

    # Also write latest weights as most recent snapshot (May 27)
    # so load_snapshots() returns something for daily scoring
    astar_path.write_text("\n".join(json.dumps(e) for e in written_astar) + "\n")
    print(f"[Backfill] Wrote {len(written_astar)} Track A★ snapshots → {astar_path.name}")

    # ── Write Track D snapshots (one per AI PM thesis) ────────────────────────
    ai_path = REPO / "logs" / "counterfactual_ai_snapshots.jsonl"
    written_ai = []
    for d in sorted(ai_pm_theses.keys()):
        portfolio = ai_pm_theses[d]
        total = sum(portfolio.values())
        if total > 0:
            portfolio = {k: v / total for k, v in portfolio.items()}
        written_ai.append({"date": d, "weights": portfolio})

    ai_path.write_text("\n".join(json.dumps(e) for e in written_ai) + "\n")
    print(f"[Backfill] Wrote {len(written_ai)} Track D snapshots → {ai_path.name}")

    # ── Recompute counterfactual_daily.jsonl ──────────────────────────────────
    # For each date, use weights held from last rebalance
    daily_records = []

    # Track which weights to use (held from last rebalance/thesis)
    current_astar_weights = {}
    current_d_weights = {}
    rebalance_sorted = sorted(rebalance_dates)
    thesis_sorted = sorted(ai_pm_theses.keys())

    for entry in multi_agent:
        date_str = entry["date"]

        # Update held weights when we hit a rebalance date
        if date_str in rebalance_dates:
            current_astar_weights = entry["weights"]

        # Update AI PM weights from last thesis on or before this date
        applicable_theses = [d for d in thesis_sorted if d <= date_str]
        if applicable_theses:
            current_d_weights = ai_pm_theses[applicable_theses[-1]]

        if date_str not in rets.index:
            continue

        row = rets.loc[date_str]

        astar_ret = portfolio_return(current_astar_weights, row) if current_astar_weights else 0.0
        d_ret = portfolio_return(current_d_weights, row) if current_d_weights else 0.0
        spy_ret = float(row["SPY"]) if "SPY" in row.index else 0.0
        b_ret = track_b.get(date_str, 0.0)

        daily_records.append({
            "date": date_str,
            "track_astar_return": round(astar_ret, 6),
            "track_a_return": round(astar_ret, 6),  # same as A★ for now (no Phase 1 separation yet)
            "track_b_return": round(b_ret, 6),
            "track_c_return": round(spy_ret, 6),
            "track_d_return": round(d_ret, 6),
        })

    daily_path = REPO / "logs" / "counterfactual_daily.jsonl"
    daily_path.write_text("\n".join(json.dumps(r) for r in daily_records) + "\n")
    print(f"[Backfill] Rewrote {len(daily_records)} daily records → {daily_path.name}")

    # ── Patch earned_authority.json with last 21 days ────────────────────────
    auth_path = REPO / "data_cache" / "earned_authority.json"
    state = json.loads(auth_path.read_text())
    last_21 = daily_records[-21:]
    state["track_d_returns"] = [r["track_d_return"] for r in last_21]
    state["track_astar_returns"] = [r["track_astar_return"] for r in last_21]
    state["ai_returns_21d"] = state["track_d_returns"]
    state["quant_returns_21d"] = state["track_astar_returns"]
    auth_path.write_text(json.dumps(state, indent=2))
    print(f"[Backfill] Patched earned_authority.json with {len(last_21)} days of real returns")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[Backfill] Sample output (last 5 days):")
    for r in daily_records[-5:]:
        print(f"  {r['date']} | A★={r['track_astar_return']:+.4f} | D={r['track_d_return']:+.4f} | B={r['track_b_return']:+.4f} | SPY={r['track_c_return']:+.4f}")

    print("\n[Backfill] Done. Run run_all_agents.py to verify counterfactual scoring works.")


if __name__ == "__main__":
    main()
