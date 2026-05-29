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


# Dates that resulted in actual Alpaca order execution.
# Apr 4/5/6/12 were debate-only sessions (no orders placed).
# May 5 / May 19 / May 27 are confirmed rebalances (May 27 in rebalance_calendar.csv).
ACTUAL_REBALANCE_DATES = {"2026-04-15", "2026-05-05", "2026-05-19", "2026-05-27"}

# Verdict files that are garbage test entries and should be skipped entirely.
EXCLUDE_VERDICT_DATES = {"2026-04-12"}  # n_positions=2, reasoning="ok" — early test artifact

# Known milestones without verdict files — added verbatim to the timeline.
HARDCODED_EVENTS = [
    {
        "date":    "2026-04-01",
        "type":    "milestone",
        "verdict": None,
        "label":   "Paper Trading Live — Initial Portfolio",
        "n_pos":   9,
        "regime":  "",
        "reason":  "29 orders placed on first day of live paper trading. "
                   "9 initial positions. Starting NAV ~$100,800.",
        "risks":   [],
        "outcome": None,
    },
    {
        "date":    "2026-05-05",
        "type":    "rebalance",
        "verdict": "proceed",
        "label":   "Rebalance #2 — Executed",
        "n_pos":   22,
        "regime":  "calm_bull",
        "reason":  "Full portfolio rotation. 40 orders placed. NAV ~$107,025 "
                   "after a strong end-of-April rally (+2.90% on Apr 30).",
        "risks":   [],
        "outcome": None,
    },
    {
        "date":    "2026-05-19",
        "type":    "rebalance",
        "verdict": "proceed",
        "label":   "Rebalance #3 — Executed · AI PM Shadow Begins",
        "n_pos":   18,
        "regime":  "calm_bull",
        "reason":  "30 orders placed. 18 positions. NAV $103,790 (portfolio "
                   "pulled back from $110K peak). AI PM Phase 0 shadow period "
                   "begins — tracks performance in parallel with 0% capital "
                   "authority. Needs 21 rebalances with Sharpe edge ≥0.05 to "
                   "earn any allocation.",
        "risks":   [],
        "outcome": None,
    },
]

# May 27 verdict file exists and is a real rebalance — override will be loaded from file.
# Listed here just for documentation; ACTUAL_REBALANCE_DATES handles the labeling.


def load_verdicts() -> list[dict]:
    events = list(HARDCODED_EVENTS)  # start with known milestones

    for f in sorted(glob.glob("outputs/debate_log/verdict_*.json")):
        try:
            with open(f) as fh:
                v = json.load(fh)
            verdict_block = v.get("verdict", {})
            if not isinstance(verdict_block, dict):
                continue

            ev_date = v.get("date", Path(f).stem.replace("verdict_", ""))

            # Skip garbage test entries
            if ev_date in EXCLUDE_VERDICT_DATES:
                continue

            # Skip if already covered by a hardcoded event
            if ev_date in {e["date"] for e in HARDCODED_EVENTS}:
                continue

            rec    = verdict_block.get("recommendation", "proceed")
            reason = verdict_block.get("reasoning", "")
            risks  = verdict_block.get("key_risks", [])
            ps     = v.get("portfolio_state", {})
            n_pos  = ps.get("n_positions", "?")
            regime = ps.get("us_regime", "")
            if isinstance(regime, str):
                regime = regime.replace("RegimeLabel.", "")
            outcome = v.get("outcome_nav_change")

            is_execution = ev_date in ACTUAL_REBALANCE_DATES
            if is_execution:
                label = f"Rebalance Executed — {rec.replace('_', ' ').title()}"
                ev_type = "rebalance"
            else:
                label = f"Debate Only (no orders) — {rec.replace('_', ' ').title()}"
                ev_type = "debate_only"

            short_reason = reason[:280].rstrip() + ("…" if len(reason) > 280 else "")
            args = v.get("arguments", {})
            events.append({
                "date":    ev_date,
                "type":    ev_type,
                "verdict": rec if is_execution else None,
                "label":   label,
                "n_pos":   n_pos,
                "regime":  regime,
                "reason":  short_reason,
                "risks":   risks[:3],
                "outcome": outcome,
                "args": {
                    "bull":   str(args.get("bull", ""))[:400],
                    "bear":   str(args.get("bear", ""))[:400],
                    "devil":  str(args.get("devils_advocate", ""))[:400],
                    "judge":  reason[:400],
                },
            })
        except Exception as e:
            print(f"  [WARN] Could not parse {f}: {e}")

    events.sort(key=lambda x: x["date"])
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


# ── Extra local data loaders ─────────────────────────────────────────────────

def load_earned_authority() -> dict:
    path = Path("data_cache/earned_authority.json")
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def load_latest_thesis() -> dict:
    files = sorted(glob.glob("outputs/ai_pm_theses/*.json"))
    if not files:
        return {}
    try:
        with open(files[-1]) as f:
            return json.load(f)
    except Exception:
        return {}


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

    # Use ALL trading days including flat days — filtering zeros inflates Sharpe
    daily = [r["day_return"] for r in records]
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

    # Standard error of annualized Sharpe: sqrt((1 + S²/2) * 252/n)
    sharpe_se = math.sqrt((1 + sharpe**2 / 2) * 252 / max(len(daily), 1)) if len(daily) > 2 else None

    return {
        "total_return": round(total_ret, 2),
        "spy_return":   round(spy_ret, 2) if spy_ret is not None else None,
        "alpha":        round(alpha, 2) if alpha is not None else None,
        "max_drawdown": round(max_dd, 2),
        "sharpe":       round(sharpe, 3),
        "sharpe_se":    round(sharpe_se, 2) if sharpe_se else None,
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
    if v is None: return "#8b949e"
    return "#3fb950" if v >= 0 else "#f85149"

def _fmt_pct(v, show_sign=True) -> str:
    if v is None: return "N/A"
    sign = "+" if (v >= 0 and show_sign) else ""
    return f"{sign}{v:.2f}%"

def _esc(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

# ── Section builders ──────────────────────────────────────────────────────────

def _earned_authority_html(auth: dict) -> str:
    if not auth:
        return '<p class="empty">data_cache/earned_authority.json not found</p>'
    phase      = auth.get("phase", 0)
    ai_rets    = auth.get("ai_returns_21d", [])
    qt_rets    = auth.get("quant_returns_21d", [])
    n_obs      = len(ai_rets)
    needed     = 21
    start_date = auth.get("phase_start_date", "")
    reverts    = auth.get("auto_revert_count", 0)
    phase_names = ["Shadow (0%)", "Phase 1 (25%)", "Phase 2 (50%)", "Phase 3 (75%)"]

    def cumret(rets):
        v = 1.0
        for r in rets: v *= (1 + r)
        return round((v - 1) * 100, 2)

    ai_cum = cumret(ai_rets) if ai_rets else 0.0
    qt_cum = cumret(qt_rets) if qt_rets else 0.0
    edge   = round(ai_cum - qt_cum, 2)
    edge_color = "#3fb950" if edge >= 0 else "#f85149"
    edge_sign  = "+" if edge >= 0 else ""

    pct_done = round(n_obs / needed * 100)

    phase_steps = ""
    for i, name in enumerate(phase_names):
        active = "active" if i == phase else ("done" if i < phase else "")
        phase_steps += f'<div class="ph-step {active}"><div class="ph-dot"></div><div class="ph-name">{name}</div></div>'
        if i < 3:
            phase_steps += f'<div class="ph-line {"done" if i < phase else ""}"></div>'

    return f"""
<div class="phase-steps">{phase_steps}</div>
<div class="phase-progress-bar"><div class="phase-fill" style="width:{pct_done}%"></div></div>
<div class="phase-detail">
  <span>{n_obs}/{needed} trading days evaluated</span>
  <span>Since {start_date}</span>
</div>
<div class="auth-stats">
  <div class="auth-stat"><div class="as-val" style="color:{_pct_color(ai_cum)}">{'+' if ai_cum>=0 else ''}{ai_cum:.2f}%</div><div class="as-lbl">AI PM (shadow)</div></div>
  <div class="auth-stat"><div class="as-val" style="color:{_pct_color(qt_cum)}">{'+' if qt_cum>=0 else ''}{qt_cum:.2f}%</div><div class="as-lbl">Quant baseline</div></div>
  <div class="auth-stat"><div class="as-val" style="color:{edge_color}">{edge_sign}{edge:.2f}%</div><div class="as-lbl">Edge</div></div>
  <div class="auth-stat"><div class="as-val">{reverts}</div><div class="as-lbl">Auto-reverts</div></div>
</div>
<p class="auth-note">AI PM earns 25% authority after 21 days with Sharpe edge ≥ 0.05. Auto-reverts to Phase 0 if AI drawdown exceeds quant by 5pp.</p>"""


def _thesis_html(thesis: dict) -> str:
    if not thesis:
        return '<p class="empty">No AI PM thesis available</p>'
    mv       = str(thesis.get("market_view", ""))[:200]
    overrides = thesis.get("quant_overrides", [])[:3]
    risks    = thesis.get("key_risks", [])[:2]
    wrong    = str(thesis.get("what_could_be_wrong", ""))[:180]
    as_of    = thesis.get("as_of_date", "")

    ov_html = ""
    for ov in overrides:
        sym    = ov.get("symbol","?")
        action = ov.get("ai_action","")
        reason = _esc(str(ov.get("reason",""))[:120])
        ov_type = ov.get("override_type","")
        ov_html += f'<div class="override-row"><span class="ov-sym">{sym}</span><span class="ov-action">{action}</span><span class="ov-type">{ov_type}</span><div class="ov-reason">{reason}</div></div>'

    risks_html = "".join(f'<li>{_esc(r[:120])}</li>' for r in risks)

    return f"""
<div class="thesis-meta">As of {as_of} · Claude Opus 4.6 · 24 tools</div>
<div class="thesis-mv">{_esc(mv)}{'…' if len(thesis.get('market_view',''))>200 else ''}</div>
{"<div class='override-section'><div class='ov-header'>Overrides vs Quant</div>" + ov_html + "</div>" if ov_html else ""}
{"<ul class='thesis-risks'>" + risks_html + "</ul>" if risks_html else ""}
{"<div class='thesis-wrong'><strong>What could be wrong:</strong> " + _esc(wrong) + "…</div>" if wrong else ""}"""


def _debate_html(verdicts: list) -> str:
    # Include any session (rebalance or debate-only) that has real argument text
    items = [v for v in verdicts if len(v.get("args", {}).get("bull", "")) > 50]
    if not items:
        return '<p class="empty">No debate records with full argument data yet.</p>'
    html = ""
    for ev in reversed(items):
        args     = ev.get("args", {})
        ev_type  = ev.get("type", "rebalance")
        verdict  = ev.get("verdict")
        is_exec  = ev_type == "rebalance"
        color    = VERDICT_COLORS.get(verdict, "#6e7681") if is_exec else "#6e7681"
        vname    = (verdict or "no orders").replace("_"," ").title()
        regime   = ev.get("regime","").replace("RegimeLabel.","").replace("_"," ")
        n_pos    = ev.get("n_pos","?")
        risks    = ev.get("risks",[])
        outcome  = ev.get("outcome")
        exec_tag = "" if is_exec else '<span style="font-size:10px;color:#6e7681;background:#21262d;padding:1px 6px;border-radius:4px">debate only · no orders</span>'
        oc_html  = ""
        if outcome is not None:
            oc = outcome * 100
            oc_html = f'<span class="oc-pill" style="color:{"#3fb950" if oc>=0 else "#f85149"}">14d outcome: {"+" if oc>=0 else ""}{oc:.2f}%</span>'

        risks_html = "".join(f"<li>{_esc(r[:120])}</li>" for r in risks[:3])

        def agent_block(label, color_cls, text):
            return f'<div class="agent-blk {color_cls}"><div class="ab-label">{label}</div><div class="ab-text">{_esc(text)}</div></div>' if text else ""

        html += f"""
<details class="debate-item">
  <summary class="debate-summary">
    <span class="di-date">{ev['date']}</span>
    <span class="di-verdict" style="border-color:{color};color:{color}">{vname}</span>
    {exec_tag}
    <span class="di-meta">{n_pos} positions · {regime}</span>
    {oc_html}
    <span class="di-caret">▸</span>
  </summary>
  <div class="debate-body">
    {agent_block("🐂 Bull", "bull", args.get("bull",""))}
    {agent_block("🐻 Bear", "bear", args.get("bear",""))}
    {agent_block("😈 Devil's Advocate", "devil", args.get("devil",""))}
    <div class="agent-blk judge">
      <div class="ab-label">⚖ Judge Verdict</div>
      <div class="ab-text">{_esc(args.get("judge",""))}</div>
      {"<ul class='debate-risks'>" + risks_html + "</ul>" if risks_html else ""}
    </div>
  </div>
</details>"""
    return html


def _timeline_html(verdicts: list) -> str:
    html = ""
    for ev in reversed(verdicts):
        ev_type = ev.get("type","rebalance")
        verdict = ev.get("verdict")
        if ev_type == "milestone":
            color, icon = "#58a6ff", "★"
        elif ev_type == "debate_only":
            color, icon = "#6e7681", "◌"
        else:
            color = VERDICT_COLORS.get(verdict,"#8b949e")
            icon  = {"proceed":"✓","reduce_size":"↓","halt_and_review":"✗"}.get(verdict,"↻")

        badge = ev["label"].split("—")[-1].strip() if "—" in ev["label"] else ev["label"]
        if ev_type == "debate_only":
            badge = "Debate Only — No Orders"
        regime_str = ev.get("regime","").replace("_"," ").replace("RegimeLabel.","")
        n_pos_str  = "" if ev_type=="debate_only" else (f'{ev.get("n_pos","?")} positions' if ev.get("n_pos") else "")
        meta = " · ".join(filter(None,[n_pos_str,regime_str]))

        risks_html = "".join(f"<li>{_esc(r[:100])}</li>" for r in ev.get("risks",[])[:2])
        outcome = ev.get("outcome")
        oc_html = ""
        if outcome is not None:
            oc = outcome*100
            oc_html = f'<div class="outcome" style="color:{"#3fb950" if oc>=0 else "#f85149"}">14d: {"+" if oc>=0 else ""}{oc:.2f}%</div>'

        html += f"""
<div class="tl-item">
  <div class="tl-dot" style="background:{color};color:#0d1117">{icon}</div>
  <div class="tl-body">
    <div class="tl-meta"><span class="tl-date">{ev['date']}</span>
      <span class="tl-badge" style="border-color:{color};color:{color}">{badge}</span>
      {"<span class='tl-pos'>"+meta+"</span>" if meta else ""}
    </div>
    <div class="tl-label">{ev['label'].split('—')[0].strip()}</div>
    <div class="tl-reason">{_esc(ev.get('reason',''))}</div>
    {"<ul class='risks'>"+risks_html+"</ul>" if risks_html else ""}
    {oc_html}
  </div>
</div>"""
    return html


def _positions_html(positions: list) -> str:
    if not positions:
        return '<p class="empty">Position data unavailable.</p>'
    rows = ""
    for p in positions:
        c = "#3fb950" if p["unrealized_plpc"]>=0 else "#f85149"
        s = "+" if p["unrealized_plpc"]>=0 else ""
        rows += f"""<tr>
          <td class="sym">{p['symbol']}</td>
          <td>{p['weight']:.1f}%</td>
          <td>${p['market_value']:,.0f}</td>
          <td>${p['current_price']:.2f}</td>
          <td style="color:{c}">{s}{p['unrealized_plpc']:.2f}%</td>
        </tr>"""
    return f"""<table>
      <thead><tr><th>Symbol</th><th>Weight</th><th>Value</th><th>Price</th><th>Unrealized</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


# ── Full page builder ─────────────────────────────────────────────────────────

def build_html(
    dates: list, port: list, spy: list,
    verdicts: list, regime_bands: list,
    stats: dict, positions: list,
    generated_at: str,
    authority: dict = None,
    thesis: dict = None,
) -> str:
    authority = authority or {}
    thesis    = thesis or {}

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

    # Rebalance / milestone lines — only draw for actual executions and milestones
    for i, ev in enumerate(verdicts):
        if ev["date"] not in dates:
            continue
        if ev["type"] == "debate_only":
            continue  # don't clutter chart with debate-only sessions
        color = VERDICT_COLORS.get(ev["verdict"], "#58a6ff")
        if ev["type"] == "milestone":
            color = "#58a6ff"
            short = "Live Start"
        else:
            short = (ev["verdict"] or "executed").replace("_", " ").title()
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

    # ── Assemble section HTML ─────────────────────────────────────────────────
    nav       = stats.get("current_nav", 0)
    base      = stats.get("start_nav", 0)
    tr        = stats.get("total_return")
    spy_r     = stats.get("spy_return")
    alph      = stats.get("alpha")
    sharpe    = stats.get("sharpe", "N/A")
    sharpe_se = stats.get("sharpe_se")
    n_days    = stats.get("days_live", "?")

    # Regime from most recent verdict or regime_bands
    regime_label = "calm bull"
    if regime_bands:
        regime_label = regime_bands[-1]["label"].replace("_"," ")
    regime_class = regime_label.replace(" ","_").replace("-","_")

    pos_html      = _positions_html(positions)
    tl_html       = _timeline_html(verdicts)
    auth_html     = _earned_authority_html(authority)
    thesis_html   = _thesis_html(thesis)
    debate_html   = _debate_html(verdicts)

    sharpe_se_str = f"±{sharpe_se:.1f}" if sharpe_se else "±?"

    # ── Serialize for JS ──────────────────────────────────────────────────────
    dates_js       = json.dumps(dates)
    port_js        = json.dumps(port)
    spy_js         = json.dumps(spy)
    annotations_js = json.dumps(annotations, indent=2)
    start_nav_js   = json.dumps(round(base, 2))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ascent Capital — Live Performance</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.5}}
a{{color:#58a6ff;text-decoration:none}}
.wrap{{max-width:1300px;margin:0 auto;padding:0 24px 80px}}
.hero{{background:linear-gradient(135deg,#0d1117 0%,#161b22 60%,#0d1117 100%);border-bottom:1px solid #30363d;padding:48px 24px 40px;margin:0 -24px 36px}}
.hero-inner{{display:grid;grid-template-columns:1fr auto;gap:32px;align-items:center;max-width:1300px;margin:0 auto}}
.hero-brand h1{{font-size:34px;font-weight:800;letter-spacing:-1px;margin-bottom:4px}}
.hero-brand h1 em{{color:#58a6ff;font-style:normal}}
.hero-brand p{{color:#8b949e;font-size:14px;margin-bottom:14px}}
.live-pill{{display:inline-flex;align-items:center;gap:6px;background:#3fb95011;border:1px solid #3fb95033;color:#3fb950;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase}}
.live-pill::before{{content:'';width:6px;height:6px;border-radius:50%;background:#3fb950;animation:pulse 1.6s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.hero-numbers{{display:flex;gap:0;margin-top:28px;flex-wrap:wrap}}
.hn{{padding:0 28px 0 0;border-right:1px solid #30363d;margin-right:28px}}
.hn:last-child{{border-right:none;margin-right:0}}
.hn-label{{font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
.hn-val{{font-size:28px;font-weight:800;letter-spacing:-0.5px}}
.hn-val.main{{font-size:40px}}
.regime-ring{{text-align:center;background:#161b22;border:2px solid #30363d;border-radius:16px;padding:20px 28px;min-width:140px}}
.regime-ring.calm_bull{{border-color:#3fb95066;background:#3fb95008}}
.regime-ring.stressed{{border-color:#d2992266;background:#d2992208}}
.regime-ring.crisis{{border-color:#f8514966;background:#f8514908}}
.regime-ring.neutral{{border-color:#58a6ff66;background:#58a6ff08}}
.regime-name{{font-size:16px;font-weight:700;text-transform:uppercase;letter-spacing:1px}}
.regime-ring.calm_bull .regime-name{{color:#3fb950}}
.regime-ring.stressed  .regime-name{{color:#d29922}}
.regime-ring.crisis    .regime-name{{color:#f85149}}
.regime-ring.neutral   .regime-name{{color:#58a6ff}}
.regime-sub{{font-size:10px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:24px}}
.stat-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px}}
.stat-label{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.stat-value{{font-size:21px;font-weight:700}}
.stat-note{{font-size:10px;color:#6e7681;margin-top:3px}}
.chart-card,.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-bottom:24px}}
.chart-card h2,.card h2{{font-size:15px;font-weight:600;color:#8b949e;margin-bottom:14px}}
.chart-title-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}}
.chart-title-row h2{{margin-bottom:0}}
.regime-chip{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:3px 10px;border-radius:12px;border:1px solid}}
.regime-chip.calm_bull{{color:#3fb950;border-color:#3fb95044}}
.regime-chip.stressed{{color:#d29922;border-color:#d2992244}}
.regime-chip.neutral{{color:#58a6ff;border-color:#58a6ff44}}
.legend{{display:flex;gap:16px;margin-bottom:10px;flex-wrap:wrap}}
.leg{{display:flex;align-items:center;gap:6px;font-size:12px;color:#8b949e}}
.leg-dot{{width:9px;height:9px;border-radius:50%}}
.leg-dash{{width:18px;height:2px;background:repeating-linear-gradient(90deg,#6b7280 0,#6b7280 5px,transparent 5px,transparent 9px)}}
.chart-wrap{{position:relative;height:390px}}
.chart-wrap.short{{height:200px}}
.chart-note{{font-size:11px;color:#6e7681;margin-top:10px;line-height:1.5}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:24px}}
@media(max-width:900px){{.grid2{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
thead th{{text-align:left;color:#6e7681;font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.4px;padding:6px 0;border-bottom:1px solid #21262d}}
tbody td{{padding:9px 0;border-bottom:1px solid #1a1f27;vertical-align:middle}}
tbody tr:last-child td{{border-bottom:none}}
.sym{{font-weight:700;color:#58a6ff;font-family:'SF Mono',Menlo,monospace;font-size:13px}}
.timeline{{display:flex;flex-direction:column;max-height:520px;overflow-y:auto;padding-right:4px;scrollbar-width:thin;scrollbar-color:#30363d transparent}}
.tl-item{{display:flex;gap:12px;padding-bottom:18px;position:relative}}
.tl-item:not(:last-child)::after{{content:'';position:absolute;left:13px;top:28px;bottom:0;width:1px;background:#21262d}}
.tl-dot{{width:26px;height:26px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;flex-shrink:0}}
.tl-body{{flex:1;padding-top:2px}}
.tl-meta{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px}}
.tl-date{{font-size:11px;color:#8b949e;font-family:monospace}}
.tl-badge{{font-size:10px;font-weight:600;border:1px solid;border-radius:10px;padding:1px 7px}}
.tl-pos{{font-size:11px;color:#6e7681}}
.tl-label{{font-size:13px;font-weight:600;color:#c9d1d9;margin-bottom:3px}}
.tl-reason{{font-size:12px;color:#8b949e;line-height:1.55;margin-bottom:5px}}
.risks{{font-size:11px;color:#6e7681;padding-left:14px;margin-bottom:5px}}
.outcome,.oc-pill{{font-size:12px;font-weight:600}}
.ai-section{{background:linear-gradient(135deg,#161b22,#0d1117);border:1px solid #30363d;border-left:3px solid #a78bfa55;border-radius:12px;padding:28px;margin-bottom:24px}}
.ai-header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:12px}}
.ai-header-left{{display:flex;align-items:center;gap:16px}}
.ai-glyph{{font-size:28px;opacity:.8}}
.ai-header h2{{font-size:18px;font-weight:700;color:#e6edf3;margin-bottom:4px}}
.ai-header p{{font-size:13px;color:#8b949e}}
.ai-phase-chip{{background:#a78bfa11;border:1px solid #a78bfa44;color:#a78bfa;border-radius:20px;padding:5px 14px;font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase}}
.grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin-bottom:20px}}
@media(max-width:1100px){{.grid3{{grid-template-columns:1fr 1fr}}}}
@media(max-width:700px){{.grid3{{grid-template-columns:1fr}}}}
.ai-card{{background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:18px 20px}}
.ai-card h3{{font-size:13px;font-weight:600;color:#a78bfa;text-transform:uppercase;letter-spacing:.5px;margin-bottom:14px}}
.phase-steps{{display:flex;align-items:center;margin-bottom:12px}}
.ph-step{{text-align:center;flex-shrink:0}}
.ph-dot{{width:12px;height:12px;border-radius:50%;background:#21262d;border:2px solid #30363d;margin:0 auto 4px}}
.ph-step.active .ph-dot{{background:#a78bfa;border-color:#a78bfa;box-shadow:0 0 8px #a78bfa66}}
.ph-step.done .ph-dot{{background:#3fb950;border-color:#3fb950}}
.ph-name{{font-size:10px;color:#6e7681}}
.ph-step.active .ph-name{{color:#a78bfa;font-weight:600}}
.ph-line{{flex:1;height:2px;background:#21262d;margin:0 4px;margin-bottom:14px}}
.ph-line.done{{background:#3fb950}}
.phase-progress-bar{{height:6px;background:#21262d;border-radius:3px;margin-bottom:8px;overflow:hidden}}
.phase-fill{{height:100%;background:linear-gradient(90deg,#a78bfa,#818cf8);border-radius:3px}}
.phase-detail{{display:flex;justify-content:space-between;font-size:11px;color:#6e7681;margin-bottom:14px}}
.auth-stats{{display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;margin-bottom:10px}}
.auth-stat{{text-align:center}}
.as-val{{font-size:15px;font-weight:700;margin-bottom:2px}}
.as-lbl{{font-size:10px;color:#6e7681}}
.auth-note{{font-size:11px;color:#6e7681;line-height:1.5;border-top:1px solid #21262d;padding-top:10px;margin-top:4px}}
.alloc-chart-wrap{{position:relative;height:150px;margin-bottom:8px}}
.alloc-note{{font-size:11px;color:#6e7681;text-align:center}}
.thesis-meta{{font-size:11px;color:#6e7681;margin-bottom:8px}}
.thesis-mv{{font-size:12px;color:#8b949e;line-height:1.6;margin-bottom:10px}}
.override-section{{margin-bottom:10px}}
.ov-header{{font-size:11px;color:#a78bfa;font-weight:600;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}}
.override-row{{display:flex;flex-wrap:wrap;gap:4px;align-items:baseline;border-left:2px solid #a78bfa33;padding-left:8px;margin-bottom:6px}}
.ov-sym{{font-weight:700;color:#58a6ff;font-family:monospace;font-size:12px}}
.ov-action{{font-size:11px;color:#d29922}}
.ov-type{{font-size:10px;color:#6e7681;background:#21262d;padding:1px 5px;border-radius:4px}}
.ov-reason{{font-size:11px;color:#6e7681;width:100%;line-height:1.5}}
.thesis-risks{{font-size:11px;color:#6e7681;padding-left:14px;margin-bottom:8px}}
.thesis-wrong{{font-size:11px;color:#6e7681;border-top:1px solid #21262d;padding-top:8px;line-height:1.5}}
.debate-intro{{font-size:12px;color:#8b949e;margin-bottom:16px;line-height:1.5}}
.debate-item{{border:1px solid #21262d;border-radius:8px;margin-bottom:8px;overflow:hidden}}
.debate-summary{{display:flex;align-items:center;gap:10px;padding:12px 16px;cursor:pointer;list-style:none;user-select:none;flex-wrap:wrap}}
.debate-summary::-webkit-details-marker{{display:none}}
.debate-item[open] .di-caret{{transform:rotate(90deg)}}
.di-caret{{color:#6e7681;transition:transform .2s;font-size:10px;margin-left:auto}}
.di-date{{font-size:12px;color:#8b949e;font-family:monospace}}
.di-verdict{{font-size:11px;font-weight:600;border:1px solid;border-radius:10px;padding:1px 8px}}
.di-meta{{font-size:11px;color:#6e7681}}
.debate-body{{padding:0 16px 16px;display:grid;grid-template-columns:1fr 1fr;gap:10px}}
@media(max-width:700px){{.debate-body{{grid-template-columns:1fr}}}}
.agent-blk{{border-radius:6px;padding:12px;font-size:12px;line-height:1.6}}
.agent-blk .ab-label{{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}}
.agent-blk.bull{{background:#3fb95010;border:1px solid #3fb95022}}.agent-blk.bull .ab-label{{color:#3fb950}}
.agent-blk.bear{{background:#f8514910;border:1px solid #f8514922}}.agent-blk.bear .ab-label{{color:#f85149}}
.agent-blk.devil{{background:#d2992210;border:1px solid #d2992222}}.agent-blk.devil .ab-label{{color:#d29922}}
.agent-blk.judge{{background:#58a6ff10;border:1px solid #58a6ff22;grid-column:1/-1}}.agent-blk.judge .ab-label{{color:#58a6ff}}
.ab-text{{color:#8b949e}}
.debate-risks{{font-size:11px;color:#6e7681;padding-left:14px;margin-top:8px}}
.empty{{color:#6e7681;font-size:13px;padding:8px 0}}
footer{{text-align:center;color:#30363d;font-size:12px;margin-top:40px;padding-top:20px;border-top:1px solid #21262d}}
</style>
</head>
<body>
<div class="hero">
  <div class="hero-inner">
    <div>
      <div class="hero-brand">
        <h1>Ascent <em>Capital</em></h1>
        <p>AI-native quant fund · Multi-agent debate · Adversarial risk committee · Alpaca paper trading</p>
        <div class="live-pill">Paper trading live</div>
      </div>
      <div class="hero-numbers">
        <div class="hn"><div class="hn-label">Current NAV</div><div class="hn-val main" id="nav-ctr">${nav:,.0f}</div></div>
        <div class="hn"><div class="hn-label">Portfolio Return</div><div class="hn-val" id="ret-ctr" style="color:{_pct_color(tr)}">{_fmt_pct(tr)}</div></div>
        <div class="hn"><div class="hn-label">Alpha vs SPY</div><div class="hn-val" id="alpha-ctr" style="color:{_pct_color(alph)}">{_fmt_pct(alph)}</div></div>
        <div class="hn"><div class="hn-label">SPY Return</div><div class="hn-val" id="spy-ctr" style="color:{_pct_color(spy_r)}">{_fmt_pct(spy_r)}</div></div>
        <div class="hn"><div class="hn-label">Trading Days</div><div class="hn-val">{n_days}</div></div>
      </div>
    </div>
    <div class="regime-ring {regime_class}"><div class="regime-name">{regime_label}</div><div class="regime-sub">Current Regime</div></div>
  </div>
</div>
<div class="wrap">
<div class="stats">
  <div class="stat-card"><div class="stat-label">Sharpe (Ann.) ⚠</div><div class="stat-value">{sharpe}</div><div class="stat-note">{sharpe_se_str} SE · N={n_days} days — unreliable at this sample size</div></div>
  <div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value" style="color:{_pct_color(stats.get('max_drawdown'))}">{_fmt_pct(stats.get('max_drawdown'),False)}</div></div>
  <div class="stat-card"><div class="stat-label">Best Day</div><div class="stat-value" style="color:#3fb950">{_fmt_pct(stats.get('best_day'))}</div></div>
  <div class="stat-card"><div class="stat-label">Worst Day</div><div class="stat-value" style="color:#f85149">{_fmt_pct(stats.get('worst_day'),False)}</div></div>
  <div class="stat-card"><div class="stat-label">Start NAV</div><div class="stat-value">${base:,.0f}</div></div>
</div>
<div class="chart-card">
  <div class="chart-title-row">
    <h2>Equity Curve — Portfolio vs SPY</h2>
    <div class="regime-chip {regime_class}">{regime_label}</div>
  </div>
  <div class="legend">
    <div class="leg"><div class="leg-dot" style="background:#58a6ff"></div>Ascent Capital</div>
    <div class="leg"><div class="leg-dash"></div>SPY</div>
    <div class="leg"><div class="leg-dot" style="background:#3fb950"></div>Proceed</div>
    <div class="leg"><div class="leg-dot" style="background:#d29922"></div>Reduce Size</div>
    <div class="leg"><div class="leg-dot" style="background:#f85149"></div>Halt &amp; Review</div>
  </div>
  <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
  <p class="chart-note">⚠ Sharpe annualized from {n_days} trading days (SE {sharpe_se_str}) — not statistically significant. Walk-forward OOS Sharpe (0.518, 2020–2026) is the rigorous number. Live since April 1, 2026.</p>
</div>
<div class="grid2">
  <div class="chart-card" style="margin-bottom:0"><h2>Drawdown from Peak</h2><div class="chart-wrap short"><canvas id="ddChart"></canvas></div></div>
  <div class="chart-card" style="margin-bottom:0"><h2>Cumulative Alpha vs SPY</h2><div class="chart-wrap short"><canvas id="alphaChart"></canvas></div></div>
</div>
<div style="margin-bottom:24px"></div>
<div class="ai-section">
  <div class="ai-header">
    <div class="ai-header-left">
      <div class="ai-glyph">⬡</div>
      <div><h2>AI Intelligence Layer</h2><p>Multi-agent debate · Adversarial risk committee · Earned authority model</p></div>
    </div>
    <div class="ai-phase-chip">Phase 0 — Shadow Mode</div>
  </div>
  <div class="grid3">
    <div class="ai-card"><h3>Earned Authority</h3>{auth_html}</div>
    <div class="ai-card"><h3>Capital Allocation</h3><div class="alloc-chart-wrap"><canvas id="allocChart"></canvas></div><p class="alloc-note">Calm bull · AI PM 0% authority</p></div>
    <div class="ai-card"><h3>Latest AI PM Decision</h3>{thesis_html}</div>
  </div>
  <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:18px 20px">
    <h3 style="font-size:13px;font-weight:600;color:#a78bfa;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Rebalance Debate History</h3>
    <p class="debate-intro">Before every rebalance, four AI agents debate the proposed portfolio — bull, bear, devil's advocate, and a judge. Arguments are adversarial by design. Click any row to read the full debate.</p>
    {debate_html}
  </div>
</div>
<div class="grid2">
  <div class="card"><h2>Current Holdings</h2>{pos_html}</div>
  <div class="card"><h2>Event Timeline</h2><div class="timeline">{tl_html}</div></div>
</div>
<footer>Generated {generated_at} &nbsp;·&nbsp; Alpaca paper API + Yahoo Finance &nbsp;·&nbsp; <a href="https://github.com/ScottDongKhang/Ascent_Capital">Source on GitHub</a></footer>
</div>
<script>
Chart.register(window['chartjs-plugin-annotation']);
const dates={dates_js},portNAV={port_js},spyNAV={spy_js},startNAV={start_nav_js};
const ANNOTATIONS={annotations_js};
function animateNum(el,end,prefix,suffix,decimals){{
  const dur=1600,t0=performance.now();
  const go=t=>{{const p=Math.min((t-t0)/dur,1),e=1-Math.pow(1-p,3),v=end*e;
    el.textContent=prefix+(decimals?v.toFixed(decimals):Math.round(v).toLocaleString('en-US'))+suffix;
    if(p<1)requestAnimationFrame(go);}};
  requestAnimationFrame(go);
}}
window.addEventListener('load',()=>{{
  animateNum(document.getElementById('nav-ctr'),{nav},'$','',0);
  animateNum(document.getElementById('ret-ctr'),{abs(tr or 0):.4f},'{'+' if (tr or 0)>=0 else '-'}','%',2);
  animateNum(document.getElementById('alpha-ctr'),{abs(alph or 0):.4f},'{'+' if (alph or 0)>=0 else '-'}','%',2);
  animateNum(document.getElementById('spy-ctr'),{abs(spy_r or 0):.4f},'{'+' if (spy_r or 0)>=0 else '-'}','%',2);
}});
const TT={{backgroundColor:'#161b22',borderColor:'#30363d',borderWidth:1,titleColor:'#8b949e',bodyColor:'#e6edf3',padding:10}};
const SX={{ticks:{{color:'#6e7681',maxTicksLimit:9,maxRotation:0}},grid:{{color:'#21262d'}}}};
const SY={{ticks:{{color:'#6e7681'}},grid:{{color:'#21262d'}}}};
new Chart(document.getElementById('equityChart'),{{type:'line',data:{{labels:dates,datasets:[
  {{label:'Ascent Capital',data:portNAV,borderColor:'#58a6ff',backgroundColor:'rgba(88,166,255,0.07)',borderWidth:2.5,pointRadius:0,pointHoverRadius:5,fill:true,tension:0.3}},
  {{label:'SPY',data:spyNAV,borderColor:'#6b7280',backgroundColor:'transparent',borderWidth:1.5,pointRadius:0,pointHoverRadius:4,borderDash:[6,4],fill:false,tension:0.3}},
]}},options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},
  plugins:{{legend:{{display:false}},tooltip:{{...TT,callbacks:{{label:ctx=>{{
    const v=ctx.raw;if(!v)return'';const r=((v-startNAV)/startNAV*100),s=r>=0?'+':'';
    return ` ${{ctx.dataset.label}}: $${{v.toLocaleString('en-US',{{maximumFractionDigits:0}})}} (${{s}}${{r.toFixed(2)}}%)`;
  }}}}}},annotation:{{annotations:ANNOTATIONS}}}},
  scales:{{x:SX,y:{{...SY,ticks:{{...SY.ticks,callback:v=>'$'+Math.round(v).toLocaleString('en-US')}}}}}}
}}}});
const dd=[];let pk=portNAV[0];
portNAV.forEach(v=>{{pk=Math.max(pk,v);dd.push(+((v-pk)/pk*100).toFixed(3));}});
new Chart(document.getElementById('ddChart'),{{type:'bar',data:{{labels:dates,datasets:[{{
  label:'Drawdown',data:dd,backgroundColor:dd.map(v=>v<-2?'rgba(248,81,73,0.75)':'rgba(248,81,73,0.4)'),borderWidth:0,borderRadius:1,
}}]}},options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},
  plugins:{{legend:{{display:false}},tooltip:{{...TT,callbacks:{{label:ctx=>`Drawdown: ${{ctx.raw.toFixed(2)}}%`}}}}}},
  scales:{{x:{{...SX,ticks:{{...SX.ticks,maxTicksLimit:6}}}},y:{{...SY,ticks:{{...SY.ticks,callback:v=>v.toFixed(1)+'%'}}}}}}
}}}});
const ca=[];let cp=0,cs=0;
for(let i=0;i<portNAV.length;i++){{
  if(i>0){{cp+=(portNAV[i]-portNAV[i-1])/portNAV[i-1]*100;const sv=spyNAV[i],sp=spyNAV[i-1];if(sv&&sp)cs+=(sv-sp)/sp*100;}}
  ca.push(+(cp-cs).toFixed(3));
}}
new Chart(document.getElementById('alphaChart'),{{type:'bar',data:{{labels:dates,datasets:[{{
  label:'Cum. Alpha',data:ca,backgroundColor:ca.map(v=>v>=0?'rgba(63,185,80,0.7)':'rgba(248,81,73,0.6)'),borderWidth:0,borderRadius:1,
}}]}},options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},
  plugins:{{legend:{{display:false}},tooltip:{{...TT,callbacks:{{label:ctx=>`Cum. Alpha: ${{ctx.raw>=0?'+':''}}${{ctx.raw.toFixed(2)}}%`}}}},
    annotation:{{annotations:{{z:{{type:'line',yMin:0,yMax:0,borderColor:'#30363d',borderWidth:1}}}}}}}},
  scales:{{x:{{...SX,ticks:{{...SX.ticks,maxTicksLimit:6}}}},y:{{...SY,ticks:{{...SY.ticks,callback:v=>(v>=0?'+':'')+v.toFixed(1)+'%'}}}}}}
}}}});
new Chart(document.getElementById('allocChart'),{{type:'doughnut',data:{{
  labels:['US Equities','International','Macro','Alternatives'],
  datasets:[{{data:[70,12,10,8],backgroundColor:['#58a6ff','#3fb950','#a78bfa','#d29922'],borderWidth:0,hoverOffset:4}}]
}},options:{{responsive:true,maintainAspectRatio:false,
  plugins:{{legend:{{position:'right',labels:{{color:'#8b949e',font:{{size:11}},boxWidth:10,padding:8}}}},
    tooltip:{{...TT,callbacks:{{label:ctx=>`${{ctx.label}}: ${{ctx.raw}}%`}}}}}},cutout:'65%'
}}}});
</script></body></html>"""


# ── README stats updater ──────────────────────────────────────────────────────

def _update_readme_stats(stats: dict, positions: list, dates: list) -> None:
    readme = Path("README.md")
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    start_tag = "<!-- LIVE_STATS_START -->"
    end_tag   = "<!-- LIVE_STATS_END -->"
    if start_tag not in text or end_tag not in text:
        return

    nav     = stats.get("current_nav", 0)
    tr      = stats.get("total_return")
    alpha   = stats.get("alpha")
    sharpe  = stats.get("sharpe", "N/A")
    max_dd  = stats.get("max_drawdown")
    n_days  = stats.get("days_live", len(dates))
    n_pos   = len(positions)
    updated = date.today().isoformat()

    def fmt(v, show_sign=True):
        if v is None: return "N/A"
        sign = "+" if (v >= 0 and show_sign) else ""
        return f"{sign}{v:.2f}%"

    block = f"""{start_tag}
| Metric | Value |
|--------|-------|
| Current NAV | ${nav:,.0f} |
| Total Return | {fmt(tr)} |
| Alpha vs SPY | {fmt(alpha)} |
| Sharpe (Ann.) | {sharpe} |
| Max Drawdown | {fmt(max_dd, False)} |
| Days Live | {n_days} |
| Open Positions | {n_pos} |
| Last Updated | {updated} |
{end_tag}"""

    before = text[:text.index(start_tag)]
    after  = text[text.index(end_tag) + len(end_tag):]
    readme.write_text(before + block + after, encoding="utf-8")
    print(f"      Updated README.md live stats table")


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
    authority   = load_earned_authority()
    thesis      = load_latest_thesis()
    stats               = compute_stats(records, spy)
    dates, port, spy_v  = build_chart_data(records, spy)
    generated_at        = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    html                = build_html(dates, port, spy_v, verdicts, regime_bands,
                                     stats, positions, generated_at,
                                     authority=authority, thesis=thesis)

    DOCS_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(html, encoding="utf-8")
    print(f"      Wrote {OUTPUT_PATH} ({len(html):,} bytes)")

    _update_readme_stats(stats, positions, dates)

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
        subprocess.run(["git", "add", str(OUTPUT_PATH), "README.md"], check=True)
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
