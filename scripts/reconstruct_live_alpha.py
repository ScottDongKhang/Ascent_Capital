"""
scripts/reconstruct_live_alpha.py

Reconstructs the complete daily alpha history by combining:
  - Alpaca portfolio/history endpoint (actual NAV per day)
  - SPY daily returns from yfinance

Writes two outputs:
  - logs/live_alpha_history.jsonl   — full record per trading day
  - logs/live_alpha_summary.json    — cumulative stats

Usage:
  .venv/bin/python scripts/reconstruct_live_alpha.py
"""

import json
import os
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, date
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
HISTORY_LOG = BASE_DIR / "logs" / "live_alpha_history.jsonl"
SUMMARY_FILE = BASE_DIR / "logs" / "live_alpha_summary.json"

try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except Exception:
    pass

ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets/v2")
ALPACA_KEY      = os.environ.get("ALPACA_KEY") or os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET   = os.environ.get("ALPACA_SECRET") or os.environ.get("ALPACA_SECRET_KEY", "")

# Account went live (first real positions) April 1, 2026
LIVE_START = date(2026, 4, 1)


def _alpaca_headers():
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
    }


def fetch_alpaca_history() -> pd.DataFrame:
    r = requests.get(
        f"{ALPACA_BASE_URL}/account/portfolio/history",
        params={"period": "6M", "timeframe": "1D", "intraday_reporting": "market_hours"},
        headers=_alpaca_headers(),
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    rows = []
    for ts, eq, ret in zip(data["timestamp"], data["equity"], data["profit_loss_pct"]):
        dt = datetime.fromtimestamp(ts).date()
        if eq and eq > 0:
            rows.append({"date": dt, "nav": float(eq), "port_return": float(ret)})

    df = pd.DataFrame(rows).set_index("date").sort_index()
    # Only keep live period
    df = df[df.index >= LIVE_START]
    return df


def fetch_spy_returns(start: date, end: date) -> pd.Series:
    spy = yf.download(
        "SPY",
        start=start.isoformat(),
        end=(pd.Timestamp(end) + pd.offsets.BDay(1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    closes = spy["Close"].squeeze()
    rets = closes.pct_change().dropna()
    rets.index = rets.index.date
    return rets


def build_history(alpaca: pd.DataFrame, spy_rets: pd.Series) -> list:
    records = []
    port_cum = 1.0
    spy_cum = 1.0

    for dt in alpaca.index:
        port_ret = alpaca.loc[dt, "port_return"]
        nav = alpaca.loc[dt, "nav"]
        spy_ret = float(spy_rets.get(dt, float("nan")))

        port_cum *= (1 + port_ret)
        if not pd.isna(spy_ret):
            spy_cum *= (1 + spy_ret)
            alpha_day = port_ret - spy_ret
            cum_alpha = port_cum - spy_cum
        else:
            alpha_day = None
            cum_alpha = None

        records.append({
            "date": dt.isoformat(),
            "nav": round(nav, 2),
            "port_return": round(port_ret, 6),
            "spy_return": round(spy_ret, 6) if not pd.isna(spy_ret) else None,
            "alpha_vs_spy": round(alpha_day, 6) if alpha_day is not None else None,
            "port_cumulative": round(port_cum - 1, 6),
            "spy_cumulative": round(spy_cum - 1, 6),
            "cumulative_alpha": round(cum_alpha, 6) if cum_alpha is not None else None,
        })

    return records


def compute_summary(records: list) -> dict:
    valid = [r for r in records if r["alpha_vs_spy"] is not None]
    alphas = [r["alpha_vs_spy"] for r in valid]
    wins = sum(1 for a in alphas if a > 0)

    last = records[-1]
    first_nav = records[0]["nav"]
    last_nav = last["nav"]

    import math
    n_days = len(records)
    # Annualize: assume 252 trading days/year
    ann_factor = 252 / n_days if n_days > 0 else 1
    ann_port = (last["port_cumulative"] + 1) ** ann_factor - 1
    ann_spy = (last["spy_cumulative"] + 1) ** ann_factor - 1
    ann_alpha = ann_port - ann_spy

    daily_alpha_std = (sum((a - (sum(alphas)/len(alphas)))**2 for a in alphas) / len(alphas)) ** 0.5 if alphas else 0
    ann_alpha_std = daily_alpha_std * (252 ** 0.5)
    ir = (sum(alphas) / len(alphas)) / daily_alpha_std if daily_alpha_std > 0 else 0

    return {
        "as_of": last["date"],
        "live_start": records[0]["date"],
        "n_trading_days": n_days,
        "nav_start": first_nav,
        "nav_end": last_nav,
        "port_cumulative_pct": round(last["port_cumulative"] * 100, 2),
        "spy_cumulative_pct": round(last["spy_cumulative"] * 100, 2),
        "cumulative_alpha_pct": round(last["cumulative_alpha"] * 100, 2) if last["cumulative_alpha"] else None,
        "annualized_port_pct": round(ann_port * 100, 2),
        "annualized_spy_pct": round(ann_spy * 100, 2),
        "annualized_alpha_pct": round(ann_alpha * 100, 2),
        "win_rate_vs_spy": f"{wins}/{len(valid)} days",
        "information_ratio": round(ir * (252 ** 0.5), 3),
        "best_day_alpha": round(max(alphas) * 100, 2) if alphas else None,
        "worst_day_alpha": round(min(alphas) * 100, 2) if alphas else None,
    }


def main():
    print("Fetching Alpaca portfolio history...")
    alpaca = fetch_alpaca_history()
    print(f"  {len(alpaca)} trading days found (Apr 1 – present)")

    start = alpaca.index[0]
    end = alpaca.index[-1]
    print(f"Fetching SPY returns {start} – {end}...")
    spy_rets = fetch_spy_returns(start, end)

    records = build_history(alpaca, spy_rets)
    summary = compute_summary(records)

    HISTORY_LOG.parent.mkdir(exist_ok=True)
    with open(HISTORY_LOG, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print(f"LIVE ALPHA SUMMARY (Apr 1 – {summary['as_of']})")
    print(f"{'='*50}")
    print(f"Trading days:        {summary['n_trading_days']}")
    print(f"NAV:                 ${summary['nav_start']:,.0f} → ${summary['nav_end']:,.0f}")
    print(f"Portfolio return:    {summary['port_cumulative_pct']:+.2f}%")
    print(f"SPY return:          {summary['spy_cumulative_pct']:+.2f}%")
    print(f"Cumulative alpha:    {summary['cumulative_alpha_pct']:+.2f}%")
    print(f"Ann. alpha (proj):   {summary['annualized_alpha_pct']:+.2f}%")
    print(f"Information ratio:   {summary['information_ratio']:+.3f}")
    print(f"Win rate vs SPY:     {summary['win_rate_vs_spy']}")
    print(f"Best day alpha:      {summary['best_day_alpha']:+.2f}%")
    print(f"Worst day alpha:     {summary['worst_day_alpha']:+.2f}%")
    print(f"\nWrote: {HISTORY_LOG}")
    print(f"Wrote: {SUMMARY_FILE}")

    print(f"\n{'─'*50}")
    print("DAILY BREAKDOWN:")
    print(f"{'Date':<12} {'NAV':>10} {'Port':>7} {'SPY':>7} {'Alpha':>7} {'CumAlpha':>9}")
    print(f"{'─'*12} {'─'*10} {'─'*7} {'─'*7} {'─'*7} {'─'*9}")
    for r in records:
        spy_s = f"{r['spy_return']*100:+.2f}%" if r['spy_return'] is not None else "  n/a  "
        alp_s = f"{r['alpha_vs_spy']*100:+.2f}%" if r['alpha_vs_spy'] is not None else "  n/a  "
        cum_s = f"{r['cumulative_alpha']*100:+.2f}%" if r['cumulative_alpha'] is not None else "  n/a  "
        print(f"{r['date']:<12} ${r['nav']:>9,.0f} {r['port_return']*100:+.2f}%  {spy_s}  {alp_s}  {cum_s}")


if __name__ == "__main__":
    main()
