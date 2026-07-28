"""scripts/backfill_holdings_log.py

Backfill logs/holdings_log.jsonl using Alpaca paper portfolio history.
Merges with existing entries (which have position detail) — does not overwrite them.
Adds SPY daily return and alpha for each backfilled day.
"""
import os
import json
import requests
import yfinance as yf
from datetime import datetime
from pathlib import Path

from ascent.utils.market_time import market_date_from_epoch

os.environ.setdefault("ALPACA_KEY", "PK6F4S764IDFRNF7ZCHHRJE7IJ")
os.environ.setdefault("ALPACA_SECRET", "G7FJ2rYPBMTyxW8cJDRUQX1GUaQLA8h8SGLww5RA2f6W")

BASE = "https://paper-api.alpaca.markets"
HEADERS = {
    "APCA-API-KEY-ID": os.environ["ALPACA_KEY"],
    "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET"],
}
LOG_PATH = Path("logs/holdings_log.jsonl")


def fetch_alpaca_history() -> dict:
    r = requests.get(
        f"{BASE}/v2/account/portfolio/history",
        headers=HEADERS,
        params={"period": "3M", "timeframe": "1D", "extended_hours": False},
    )
    r.raise_for_status()
    hist = r.json()
    out = {}
    for ts, eq, pl, plp in zip(
        hist["timestamp"], hist["equity"], hist["profit_loss"], hist["profit_loss_pct"]
    ):
        if not eq:
            continue
        dt = market_date_from_epoch(ts).isoformat()
        day_return = (eq / (eq - pl) - 1) if pl and (eq - pl) > 0 else 0.0
        out[dt] = {
            "equity": eq,
            "profit_loss": pl or 0,
            "day_return": day_return,
        }
    return out


def fetch_spy_returns(start: str, end: str) -> dict:
    import pandas as pd
    spy = yf.download("SPY", start=start, end=end, progress=False, auto_adjust=True)
    closes = spy["Close"]
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    pct = closes.pct_change().dropna()
    result = {}
    for idx, val in pct.items():
        if hasattr(idx, "date"):
            key = str(idx.date())
        else:
            key = str(pd.Timestamp(str(idx)).date())
        result[key] = float(val)
    return result


def load_existing_log() -> dict:
    existing = {}
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    e = json.loads(line)
                    existing[e["date"]] = e
                except Exception:
                    pass
    return existing


def main():
    print("[Backfill] Fetching Alpaca portfolio history...")
    alpaca = fetch_alpaca_history()
    print(f"[Backfill] {len(alpaca)} days from Alpaca ({min(alpaca)} → {max(alpaca)})")

    print("[Backfill] Fetching SPY returns...")
    spy_rets = fetch_spy_returns("2026-03-20", "2026-05-28")
    print(f"[Backfill] {len(spy_rets)} SPY daily returns")

    existing = load_existing_log()
    print(f"[Backfill] {len(existing)} existing log entries")

    all_dates = sorted(set(alpaca.keys()) | set(existing.keys()))
    merged = []
    backfilled = 0

    for dt in all_dates:
        if dt in existing:
            merged.append(existing[dt])
        elif dt in alpaca:
            a = alpaca[dt]
            spy_ret = spy_rets.get(dt, 0.0)
            merged.append({
                "date": dt,
                "timestamp": dt + "T16:00:00",
                "equity": a["equity"],
                "cash": 1272.76,
                "day_return": a["day_return"],
                "spy_return": spy_ret,
                "alpha_vs_spy": a["day_return"] - spy_ret,
                "n_positions": None,
                "positions": [],
                "_source": "alpaca_backfill",
            })
            backfilled += 1

    merged.sort(key=lambda x: x["date"])

    with open(LOG_PATH, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry) + "\n")

    print(f"[Backfill] Written {len(merged)} entries ({backfilled} backfilled, {len(existing)} existing)")
    print()

    # Print full table
    print(f"{'Date':<12} {'NAV':>10} {'Day Ret':>9} {'SPY':>8} {'Alpha':>8}")
    print("-" * 55)
    cum_port = 1.0
    cum_spy = 1.0
    for e in merged:
        dr = e.get("day_return", 0) or 0
        sr = e.get("spy_return", 0) or 0
        alpha = e.get("alpha_vs_spy", 0) or 0
        cum_port *= (1 + dr)
        cum_spy *= (1 + sr)
        src = " ←" if e.get("_source") == "alpaca_backfill" else "  "
        print(f"{e['date']:<12} ${e['equity']:>9,.0f} {dr*100:>+8.2f}% {sr*100:>+7.2f}% {alpha*100:>+7.2f}%{src}")

    print("-" * 55)
    print(f"{'TOTAL':<12} ${merged[-1]['equity']:>9,.0f} {(cum_port-1)*100:>+8.2f}% {(cum_spy-1)*100:>+7.2f}% {(cum_port-cum_spy)*100:>+7.2f}%")


if __name__ == "__main__":
    main()
