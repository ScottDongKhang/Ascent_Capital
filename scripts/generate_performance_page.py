#!/usr/bin/env python3
"""
Generate docs/index.html — static GitHub Pages performance dashboard.

Pulls live data from Alpaca paper trading API, benchmarks against SPY,
and annotates the equity curve with rebalance verdicts and regime changes.

Run from repo root:
    .venv/bin/python scripts/generate_performance_page.py [--push]

--push  Auto-commits and pushes docs/index.html after generation.
"""

import glob
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PAPER_BASE   = "https://paper-api.alpaca.markets"
LIVE_START   = date(2026, 4, 1)
DOCS_DIR     = Path("docs")
OUTPUT_PATH  = DOCS_DIR / "index.html"


# ── Alpaca helpers ────────────────────────────────────────────────────────────

def _alpaca_headers() -> dict:
    key    = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY", "")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("ALPACA_SECRET", "")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_KEY not set")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_portfolio_history() -> list[dict]:
    """Returns [{date, equity, day_return_pct}] from LIVE_START onward."""
    r = requests.get(
        f"{PAPER_BASE}/v2/account/portfolio/history",
        headers=_alpaca_headers(),
        params={"period": "6M", "timeframe": "1D", "extended_hours": "false"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    records = []
    timestamps = data.get("timestamp", [])
    equity     = data.get("equity", [])
    prev_eq    = None

    for ts, eq in zip(timestamps, equity):
        if eq is None or eq == 0:
            prev_eq = eq
            continue
        dt = datetime.fromtimestamp(ts).date()
        if dt < LIVE_START:
            prev_eq = eq
            continue
        day_ret = (eq / prev_eq - 1) if (prev_eq and prev_eq > 0) else 0.0
        records.append({"date": dt.isoformat(), "equity": round(eq, 2), "day_return": round(day_ret, 6)})
        prev_eq = eq

    return records


def fetch_current_positions() -> list[dict]:
    r = requests.get(f"{PAPER_BASE}/v2/positions", headers=_alpaca_headers(), timeout=10)
    r.raise_for_status()
    raw = r.json()
    total_mv = sum(float(p["market_value"]) for p in raw if float(p["market_value"]) > 0)
    out = []
    for p in sorted(raw, key=lambda x: -float(x["market_value"])):
        mv = float(p["market_value"])
        out.append({
            "symbol":          p["symbol"],
            "qty":             float(p["qty"]),
            "market_value":    round(mv, 2),
            "weight":          round(mv / total_mv * 100, 1) if total_mv else 0,
            "unrealized_plpc": round(float(p["unrealized_plpc"]) * 100, 2),
            "current_price":   round(float(p["current_price"]), 2),
            "avg_entry_price": round(float(p["avg_entry_price"]), 2),
        })
    return out


# ── Market data ───────────────────────────────────────────────────────────────

def fetch_spy(start: date, end: date) -> dict[str, float]:
    """Returns {date_str: close_price}."""
    df = yf.download(
        "SPY",
        start=start.isoformat(),
        end=(end + timedelta(days=3)).isoformat(),
        progress=False,
        auto_adjust=True,
    )
    if df.empty:
        return {}
    close = df["Close"]
    if isinstance(close.columns if hasattr(close, "columns") else None, object):
        close = close.iloc[:, 0]
    return {str(idx.date()): round(float(v), 4) for idx, v in close.items()}


# ── Annotations from local files ──────────────────────────────────────────────

VERDICT_COLORS = {
    "proceed":          "#3fb950",
    "reduce_size":      "#d29922",
    "halt_and_review":  "#f85149",
}

REGIME_COLORS = {
    "calm_bull":  "rgba(63,185,80,0.07)",
    "neutral":    "rgba(88,166,255,0.06)",
    "stressed":   "rgba(210,153,34,0.10)",
    "crisis":     "rgba(248,81,73,0.12)",
    "uncertain":  "rgba(139,148,158,0.08)",
}


def load_verdicts() -> list[dict]:
    events = []
    for f in sorted(glob.glob("outputs/debate_log/verdict_*.json")):
        try:
            with open(f) as fh:
                v = json.load(fh)
            verdict_block = v.get("verdict", {})
            if not isinstance(verdict_block, dict):
                continue
            rec   = verdict_block.get("recommendation", "proceed")
            reason = verdict_block.get("reasoning", "")
            risks  = verdict_block.get("key_risks", [])
            ps     = v.get("portfolio_state", {})
            n_pos  = ps.get("n_positions", "?")
            regime = ps.get("us_regime", "")
            if isinstance(regime, str):
                regime = regime.replace("RegimeLabel.", "")
            outcome = v.get("outcome_nav_change")

            short_reason = reason[:280].rstrip() + ("…" if len(reason) > 280 else "")
            events.append({
                "date":    v.get("date", Path(f).stem.replace("verdict_", "")),
                "type":    "rebalance",
                "verdict": rec,
                "label":   f"Rebalance — {rec.replace('_', ' ').title()}",
                "n_pos":   n_pos,
                "regime":  regime,
                "reason":  short_reason,
                "risks":   risks[:3],
                "outcome": outcome,
            })
        except Exception as e:
            print(f"  [WARN] Could not parse {f}: {e}")
    return events


def load_regime_bands(start: date) -> list[dict]:
    """Returns [{start, end, label}] for the live trading period."""
    csv = Path("dashboard/regime_labels.csv")
    if not csv.exists():
        return []
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    label_col = df.iloc[:, 0]

    bands, cur_label, cur_start = [], None, None
    for idx, lbl in label_col.items():
        d = idx.date()
        if d < start:
            cur_label, cur_start = str(lbl), d
            continue
        lbl = str(lbl)
        if cur_label is None:
            cur_label, cur_start = lbl, d
        elif lbl != cur_label:
            bands.append({"start": cur_start.isoformat(), "end": d.isoformat(), "label": cur_label})
            cur_label, cur_start = lbl, d
    if cur_label and cur_start:
        bands.append({"start": cur_start.isoformat(), "end": date.today().isoformat(), "label": cur_label})
    return bands


# ── Stats ─────────────────────────────────────────────────────────────────────

def compute_stats(records: list[dict], spy: dict[str, float]) -> dict:
    if not records:
        return {}
    eqs   = [r["equity"] for r in records]
    dates = [r["date"] for r in records]

    base, current = eqs[0], eqs[-1]
    total_ret = (current / base - 1) * 100

    peak, max_dd = base, 0.0
    for e in eqs:
        peak  = max(peak, e)
        max_dd = min(max_dd, (e - peak) / peak * 100)

    daily = [r["day_return"] for r in records if r["day_return"] != 0]
    sharpe = 0.0
    if len(daily) > 2:
        mu  = statistics.mean(daily)
        std = statistics.stdev(daily)
        sharpe = (mu / std * math.sqrt(252)) if std > 0 else 0.0

    spy_base = spy.get(dates[0])
    spy_cur  = spy.get(dates[-1])
    spy_ret  = ((spy_cur / spy_base - 1) * 100) if (spy_base and spy_cur) else None
    alpha    = (total_ret - spy_ret) if spy_ret is not None else None

    best  = max(daily) * 100 if daily else 0.0
    worst = min(daily) * 100 if daily else 0.0

    return {
        "total_return": round(total_ret, 2),
        "spy_return":   round(spy_ret, 2) if spy_ret is not None else None,
        "alpha":        round(alpha, 2) if alpha is not None else None,
        "max_drawdown": round(max_dd, 2),
        "sharpe":       round(sharpe, 3),
        "best_day":     round(best, 2),
        "worst_day":    round(worst, 2),
        "days_live":    len(dates),
        "current_nav":  round(current, 2),
        "start_nav":    round(base, 2),
    }


# ── Chart data ────────────────────────────────────────────────────────────────

def build_chart_data(records: list[dict], spy: dict[str, float]) -> tuple:
    """Both series start at same dollar value (portfolio base NAV)."""
    if not records:
        return [], [], []
    dates = [r["date"] for r in records]
    eqs   = [r["equity"] for r in records]
    base  = eqs[0]

    spy_base = spy.get(dates[0])
    spy_vals = []
    for d in dates:
        sc = spy.get(d)
        if sc and spy_base:
            spy_vals.append(round(sc / spy_base * base, 2))
        else:
            spy_vals.append(None)

    return dates, eqs, spy_vals


# ── HTML generation ───────────────────────────────────────────────────────────

def _pct_color(v) -> str:
    if v is None:
        return "#8b949e"
    return "#3fb950" if v >= 0 else "#f85149"


def _fmt_pct(v, show_sign=True) -> str:
    if v is None:
        return "N/A"
    sign = "+" if (v >= 0 and show_sign) else ""
    return f"{sign}{v:.2f}%"


def build_html(
    dates: list, port: list, spy: list,
    verdicts: list, regime_bands: list,
    stats: dict, positions: list,
    generated_at: str,
) -> str:

    # ── Chart.js annotation objects ──────────────────────────────────────────
    annotations = {}

    # Regime bands
    for i, band in enumerate(regime_bands):
        if band["start"] not in dates and band["end"] not in dates:
            continue
        color = REGIME_COLORS.get(band["label"], "rgba(100,100,100,0.05)")
        annotations[f"regime{i}"] = {
            "type":            "box",
            "xMin":            band["start"],
            "xMax":            band["end"],
            "backgroundColor": color,
            "borderWidth":     0,
            "label": {
                "display":         True,
                "content":         band["label"].replace("_", " "),
                "position":        {"x": "start", "y": "start"},
                "color":           "#8b949e",
                "font":            {"size": 9},
                "backgroundColor": "transparent",
                "yAdjust":         4,
                "xAdjust":         4,
            },
        }

    # Rebalance lines
    for i, ev in enumerate(verdicts):
        if ev["date"] not in dates:
            continue
        color = VERDICT_COLORS.get(ev["verdict"], "#8b949e")
        short = ev["verdict"].replace("_", " ").title()
        annotations[f"reb{i}"] = {
            "type":        "line",
            "xMin":        ev["date"],
            "xMax":        ev["date"],
            "borderColor": color,
            "borderWidth": 2,
            "borderDash":  [5, 4],
            "label": {
                "display":         True,
                "content":         short,
                "position":        "end",
                "backgroundColor": color + "22",
                "color":           color,
                "font":            {"size": 10, "weight": "600"},
                "padding":         3,
                "yAdjust":         6,
            },
        }

    # ── Stats cards ──────────────────────────────────────────────────────────
    def stat(label, value, extra_style=""):
        return f"""
        <div class="stat-card">
          <div class="stat-label">{label}</div>
          <div class="stat-value" style="{extra_style}">{value}</div>
        </div>"""

    nav  = stats.get("current_nav", 0)
    base = stats.get("start_nav", 0)
    tr   = stats.get("total_return")
    spy_r = stats.get("spy_return")
    alph  = stats.get("alpha")

    stats_html = (
        stat("Portfolio Return",   _fmt_pct(tr),   f"color:{_pct_color(tr)}")
      + stat("vs SPY (Alpha)",     _fmt_pct(alph),  f"color:{_pct_color(alph)}")
      + stat("SPY Return",         _fmt_pct(spy_r), f"color:{_pct_color(spy_r)}")
      + stat("Sharpe (Ann.)",      str(stats.get("sharpe", "N/A")))
      + stat("Max Drawdown",       _fmt_pct(stats.get("max_drawdown"), False),
             f"color:{_pct_color(stats.get('max_drawdown'))}")
      + stat("Best Day",           _fmt_pct(stats.get("best_day")), "color:#3fb950")
      + stat("Worst Day",          _fmt_pct(stats.get("worst_day"), False), "color:#f85149")
      + stat("Days Live",          str(stats.get("days_live", "?")))
      + stat("Current NAV",        f"${nav:,.0f}")
    )

    # ── Positions table ───────────────────────────────────────────────────────
    if positions:
        rows = ""
        for p in positions:
            c    = "#3fb950" if p["unrealized_plpc"] >= 0 else "#f85149"
            sign = "+" if p["unrealized_plpc"] >= 0 else ""
            rows += f"""
            <tr>
              <td class="sym">{p['symbol']}</td>
              <td>{p['weight']:.1f}%</td>
              <td>${p['market_value']:,.0f}</td>
              <td>${p['current_price']:.2f}</td>
              <td style="color:{c}">{sign}{p['unrealized_plpc']:.2f}%</td>
            </tr>"""
        pos_html = f"""
        <table>
          <thead><tr>
            <th>Symbol</th><th>Weight</th><th>Value</th><th>Price</th><th>Unrealized</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>"""
    else:
        pos_html = '<p class="empty">Position data unavailable — Alpaca API key not configured.</p>'

    # ── Timeline ──────────────────────────────────────────────────────────────
    all_events = sorted(verdicts, key=lambda x: x["date"])
    tl_html = ""
    for ev in reversed(all_events):   # newest first
        color = VERDICT_COLORS.get(ev["verdict"], "#8b949e")
        icon_map = {"proceed": "✓", "reduce_size": "↓", "halt_and_review": "✗"}
        icon   = icon_map.get(ev["verdict"], "↻")
        label  = ev["verdict"].replace("_", " ").title()

        risks_html = ""
        for r in ev.get("risks", []):
            risks_html += f'<li>{r}</li>'
        if risks_html:
            risks_html = f'<ul class="risks">{risks_html}</ul>'

        outcome_html = ""
        if ev.get("outcome") is not None:
            oc = ev["outcome"] * 100
            oc_color = "#3fb950" if oc >= 0 else "#f85149"
            outcome_html = f'<div class="outcome" style="color:{oc_color}">Realized +14d: {"+" if oc>=0 else ""}{oc:.2f}%</div>'

        tl_html += f"""
        <div class="tl-item">
          <div class="tl-dot" style="background:{color};color:#0d1117">{icon}</div>
          <div class="tl-body">
            <div class="tl-meta">
              <span class="tl-date">{ev['date']}</span>
              <span class="tl-badge" style="border-color:{color};color:{color}">{label}</span>
              <span class="tl-pos">{ev.get('n_pos','?')} positions · {ev.get('regime','').replace('_',' ')}</span>
            </div>
            <div class="tl-reason">{ev.get('reason','')}</div>
            {risks_html}
            {outcome_html}
          </div>
        </div>"""

    # ── Serialize for JS ──────────────────────────────────────────────────────
    dates_js       = json.dumps(dates)
    port_js        = json.dumps(port)
    spy_js         = json.dumps(spy)
    annotations_js = json.dumps(annotations, indent=2)
    start_nav_js   = json.dumps(round(base, 2))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ascent Capital — Live Performance</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.5}}
a{{color:#58a6ff;text-decoration:none}}

.wrap{{max-width:1280px;margin:0 auto;padding:40px 24px 60px}}

/* header */
header{{margin-bottom:32px}}
.hrow{{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}}
header h1{{font-size:30px;font-weight:800;letter-spacing:-0.5px}}
header h1 em{{color:#58a6ff;font-style:normal}}
.badge{{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;letter-spacing:.4px;text-transform:uppercase;border:1px solid #3fb95066;color:#3fb950;background:#3fb95011}}
.badge::before{{content:'';width:6px;height:6px;border-radius:50%;background:#3fb950;animation:pulse 1.6s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
header p{{color:#8b949e;font-size:14px;margin-top:6px}}

/* stats */
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(115px,1fr));gap:10px;margin-bottom:24px}}
.stat-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px}}
.stat-label{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.stat-value{{font-size:21px;font-weight:700}}

/* chart */
.chart-card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-bottom:24px}}
.chart-card h2{{font-size:15px;font-weight:600;color:#8b949e;margin-bottom:14px}}
.legend{{display:flex;gap:18px;margin-bottom:10px;flex-wrap:wrap}}
.leg{{display:flex;align-items:center;gap:6px;font-size:12px;color:#8b949e}}
.leg-dot{{width:10px;height:10px;border-radius:50%}}
.leg-dash{{width:18px;height:2px;background:repeating-linear-gradient(90deg,#6b7280 0,#6b7280 5px,transparent 5px,transparent 9px)}}
.chart-wrap{{position:relative;height:400px}}

/* two-col */
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px}}
.card h2{{font-size:15px;font-weight:600;color:#8b949e;margin-bottom:16px}}

/* table */
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead th{{text-align:left;color:#6e7681;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.4px;padding:6px 0;border-bottom:1px solid #21262d}}
tbody td{{padding:9px 0;border-bottom:1px solid #161b22;vertical-align:middle}}
tbody tr:last-child td{{border-bottom:none}}
.sym{{font-weight:700;color:#58a6ff;font-family:'SF Mono',Menlo,monospace;font-size:13px}}

/* timeline */
.timeline{{display:flex;flex-direction:column;gap:0;max-height:480px;overflow-y:auto;padding-right:6px;scrollbar-width:thin;scrollbar-color:#30363d transparent}}
.tl-item{{display:flex;gap:12px;padding-bottom:20px;position:relative}}
.tl-item:not(:last-child)::after{{content:'';position:absolute;left:13px;top:28px;bottom:0;width:1px;background:#21262d}}
.tl-dot{{width:27px;height:27px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;flex-shrink:0}}
.tl-body{{flex:1;padding-top:2px}}
.tl-meta{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:5px}}
.tl-date{{font-size:12px;color:#8b949e;font-family:monospace}}
.tl-badge{{font-size:11px;font-weight:600;border:1px solid;border-radius:12px;padding:1px 8px}}
.tl-pos{{font-size:11px;color:#6e7681}}
.tl-reason{{font-size:12px;color:#8b949e;line-height:1.6;margin-bottom:6px}}
.risks{{font-size:11px;color:#6e7681;padding-left:14px;margin-bottom:6px;display:flex;flex-direction:column;gap:2px}}
.outcome{{font-size:12px;font-weight:600}}
.empty{{color:#6e7681;font-size:13px;padding:8px 0}}

/* footer */
footer{{text-align:center;color:#21262d;font-size:12px;margin-top:40px;padding-top:20px;border-top:1px solid #21262d}}
</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="hrow">
    <h1>Ascent <em>Capital</em></h1>
    <div class="badge">Paper trading live</div>
  </div>
  <p>AI-native quant fund · Multi-agent + LLM debate · Alpaca paper trading · Live since April 1, 2026</p>
</header>

<div class="stats">{stats_html}</div>

<div class="chart-card">
  <h2>Equity Curve — Portfolio vs SPY</h2>
  <div class="legend">
    <div class="leg"><div class="leg-dot" style="background:#58a6ff"></div>Ascent Capital (NAV $)</div>
    <div class="leg"><div class="leg-dash"></div>SPY (same starting value)</div>
    <div class="leg"><div class="leg-dot" style="background:#3fb950"></div>Rebalance: Proceed</div>
    <div class="leg"><div class="leg-dot" style="background:#d29922"></div>Rebalance: Reduce Size</div>
    <div class="leg"><div class="leg-dot" style="background:#f85149"></div>Rebalance: Halt & Review</div>
  </div>
  <div class="chart-wrap"><canvas id="chart"></canvas></div>
</div>

<div class="grid2">
  <div class="card">
    <h2>Current Holdings</h2>
    {pos_html}
  </div>
  <div class="card">
    <h2>Rebalance History</h2>
    <div class="timeline">{tl_html}</div>
  </div>
</div>

<footer>
  Generated {generated_at} &nbsp;·&nbsp; Data: Alpaca paper trading API + SPY via Yahoo Finance &nbsp;·&nbsp;
  <a href="https://github.com/ScottDongKhang/Ascent_Capital">Source on GitHub</a>
</footer>

</div>

<script>
Chart.register(window['chartjs-plugin-annotation']);

const dates  = {dates_js};
const portNAV = {port_js};
const spyNAV  = {spy_js};
const startNAV = {start_nav_js};
const ANNOTATIONS = {annotations_js};

const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {{
  type: 'line',
  data: {{
    labels: dates,
    datasets: [
      {{
        label:           'Ascent Capital',
        data:            portNAV,
        borderColor:     '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.07)',
        borderWidth:     2.5,
        pointRadius:     0,
        pointHoverRadius: 5,
        fill:            true,
        tension:         0.3,
        yAxisID:         'y',
      }},
      {{
        label:           'SPY',
        data:            spyNAV,
        borderColor:     '#6b7280',
        backgroundColor: 'transparent',
        borderWidth:     1.5,
        pointRadius:     0,
        pointHoverRadius: 4,
        borderDash:      [6, 4],
        fill:            false,
        tension:         0.3,
        yAxisID:         'y',
      }},
    ]
  }},
  options: {{
    responsive:          true,
    maintainAspectRatio: false,
    interaction:         {{mode:'index',intersect:false}},
    plugins: {{
      legend:     {{display:false}},
      tooltip: {{
        backgroundColor: '#161b22',
        borderColor:     '#30363d',
        borderWidth:     1,
        titleColor:      '#8b949e',
        bodyColor:       '#e6edf3',
        callbacks: {{
          label: (ctx) => {{
            const v = ctx.raw;
            if (v === null || v === undefined) return '';
            const ret = ((v - startNAV) / startNAV * 100);
            const sign = ret >= 0 ? '+' : '';
            return ` ${{ctx.dataset.label}}: $${{v.toLocaleString('en-US',{{maximumFractionDigits:0}})}} (${{sign}}${{ret.toFixed(2)}}%)`;
          }}
        }}
      }},
      annotation: {{annotations: ANNOTATIONS}},
    }},
    scales: {{
      x: {{
        ticks: {{color:'#6e7681',maxTicksLimit:9,maxRotation:0}},
        grid:  {{color:'#21262d'}},
      }},
      y: {{
        ticks: {{
          color: '#6e7681',
          callback: (v) => '$' + Math.round(v).toLocaleString('en-US'),
        }},
        grid: {{color:'#21262d'}},
      }},
    }},
  }},
}});
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    push = "--push" in sys.argv

    print("=== Ascent Capital — Generate Performance Page ===")

    # Portfolio history
    print("\n[1/5] Fetching Alpaca portfolio history…")
    try:
        records = fetch_portfolio_history()
        print(f"      {len(records)} trading days from {records[0]['date']} → {records[-1]['date']}")
    except Exception as e:
        print(f"      ERROR: {e}")
        print("      Cannot generate page without portfolio data. Check ALPACA_API_KEY.")
        sys.exit(1)

    start_dt = date.fromisoformat(records[0]["date"])
    end_dt   = date.fromisoformat(records[-1]["date"])

    # SPY
    print("\n[2/5] Fetching SPY benchmark…")
    spy = fetch_spy(start_dt, end_dt)
    print(f"      {len(spy)} data points")

    # Annotations
    print("\n[3/5] Loading verdicts and regime data…")
    verdicts     = load_verdicts()
    regime_bands = load_regime_bands(start_dt)
    print(f"      {len(verdicts)} rebalance verdicts, {len(regime_bands)} regime bands")

    # Positions
    print("\n[4/5] Fetching current positions…")
    try:
        positions = fetch_current_positions()
        print(f"      {len(positions)} open positions")
    except Exception as e:
        print(f"      WARNING: {e} — positions section will be empty")
        positions = []

    # Build
    print("\n[5/5] Building HTML…")
    stats               = compute_stats(records, spy)
    dates, port, spy_v  = build_chart_data(records, spy)
    generated_at        = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    html                = build_html(dates, port, spy_v, verdicts, regime_bands,
                                     stats, positions, generated_at)

    DOCS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"      Wrote {OUTPUT_PATH} ({len(html):,} bytes)")

    # Key stats
    print(f"\n{'='*48}")
    print(f"  Total return : {_fmt_pct(stats.get('total_return'))}")
    print(f"  Alpha vs SPY : {_fmt_pct(stats.get('alpha'))}")
    print(f"  Sharpe       : {stats.get('sharpe')}")
    print(f"  Max DD       : {_fmt_pct(stats.get('max_drawdown'), False)}")
    print(f"  Current NAV  : ${stats.get('current_nav', 0):,.0f}")
    print(f"{'='*48}\n")

    if push:
        print("[Git] Committing and pushing…")
        today = date.today().isoformat()
        subprocess.run(["git", "add", str(OUTPUT_PATH)], check=True)
        subprocess.run([
            "git", "commit", "-m",
            f"chore: update performance dashboard {today}\n\nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
        ], check=True)
        subprocess.run(["git", "push"], check=True)
        print("[Git] Pushed.")
    else:
        print("Run with --push to auto-commit and push to GitHub Pages.")


if __name__ == "__main__":
    main()
