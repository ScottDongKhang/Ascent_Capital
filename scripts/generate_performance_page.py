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
from datetime import date, datetime, timedelta, timezone
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
PRICES_LIVE_PATH = "data_cache/prices_live.parquet"


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

    # Alpaca publishes the 1D bar for a session only after ~00:00 UTC (17:00 PT),
    # i.e. after our daily run — so the series above always ends at YESTERDAY.
    # Append today's session from live account equity so the chart is current.
    today = date.today()
    if records and records[-1]["date"] < today.isoformat() and today.weekday() < 5:
        try:
            r = requests.get(f"{PAPER_BASE}/v2/account", headers=_alpaca_headers(), timeout=15)
            r.raise_for_status()
            live_eq = float(r.json().get("equity", 0) or 0)
            if live_eq > 0:
                prev = records[-1]["equity"]
                records.append({
                    "date":       today.isoformat(),
                    "equity":     round(live_eq, 2),
                    "day_return": round(live_eq / prev - 1, 6) if prev else 0.0,
                })
        except Exception:
            pass  # chart falls back to ending at the last published bar

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
    "calm_bull":  "rgba(63,185,80,0.05)",
    "neutral":    "rgba(88,166,255,0.05)",
    "stressed":   "rgba(210,153,34,0.08)",
    "crisis":     "rgba(248,81,73,0.10)",
    "uncertain":  "rgba(139,148,158,0.06)",
}

# Solid accents for regime boundary labels on the equity chart
REGIME_ACCENT = {
    "calm_bull":  "#3fb950",
    "neutral":    "#58a6ff",
    "stressed":   "#d29922",
    "crisis":     "#f85149",
    "uncertain":  "#8b949e",
}


# Dates that resulted in actual Alpaca order execution.
# Apr 4/5/6/12 were debate-only sessions (no orders placed).
# May 5 / May 19 / May 27 are confirmed rebalances (May 27 in rebalance_calendar.csv).
ACTUAL_REBALANCE_DATES = {"2026-04-15", "2026-05-05", "2026-05-19", "2026-05-27"}

# Sequential rebalance numbers (April 1 = initial deployment, not a rebalance)
REBALANCE_NUMBERS = {
    "2026-04-15": 1,
    "2026-05-05": 2,
    "2026-05-19": 3,
    "2026-05-27": 4,
}

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


def _execution_dates_from_eod_log() -> set[str]:
    """Dates where eod_runner actually submitted orders (rebalanced=True)."""
    dates: set[str] = set()
    try:
        with open("logs/eod_log.jsonl") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("rebalanced"):
                    d = rec.get("date") or rec.get("run_date")
                    if d:
                        dates.add(d)
    except FileNotFoundError:
        pass
    return dates


def load_verdicts() -> list[dict]:
    events = list(HARDCODED_EVENTS)  # start with known milestones

    # Execution days: hardcoded history + anything the EOD log proves was executed
    execution_dates = ACTUAL_REBALANCE_DATES | _execution_dates_from_eod_log()
    reb_numbers = dict(REBALANCE_NUMBERS)
    next_num = max(reb_numbers.values(), default=0) + 1
    for d in sorted(execution_dates):
        if d not in reb_numbers:
            reb_numbers[d] = next_num
            next_num += 1

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

            is_execution = ev_date in execution_dates
            if is_execution:
                reb_num = reb_numbers.get(ev_date, "")
                num_str = f" #{reb_num}" if reb_num else ""
                label = f"Rebalance{num_str} — Executed · {rec.replace('_', ' ').title()}"
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


def load_counterfactual() -> list:
    """Load Track A★/A/B/C/D daily returns from counterfactual_daily.jsonl."""
    path = Path("logs/counterfactual_daily.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def load_perf_feedback() -> dict:
    """Load daily learning brief from ai_pm_perf_feedback.json."""
    path = Path("data_cache/ai_pm_perf_feedback.json")
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_ai_pm_decisions() -> list:
    """Load per-rebalance override records from ai_pm_decision_log.jsonl."""
    path = Path("logs/ai_pm_decision_log.jsonl")
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return sorted(rows, key=lambda x: x.get("date", ""))


def load_latest_allocation() -> dict:
    """Returns {agent_id: pct} from the most recent rebalance verdict file."""
    files = sorted(glob.glob("outputs/debate_log/verdict_*.json"))
    for f in reversed(files):
        date_str = Path(f).stem.replace("verdict_", "")
        if date_str in ACTUAL_REBALANCE_DATES:
            try:
                with open(f) as fh:
                    d = json.load(fh)
                alloc = d.get("portfolio_state", {}).get("allocation", {})
                if alloc:
                    return {
                        "us_equities":   round(alloc.get("us_equities", 0) * 100),
                        "international": round(alloc.get("international", 0) * 100),
                        "macro":         round(alloc.get("macro", 0) * 100),
                        "alternatives":  round(alloc.get("alternatives", 0) * 100),
                    }
            except Exception:
                pass
    # Fallback to CLAUDE.md orchestrator defaults for calm_bull
    return {"us_equities": 60, "international": 15, "macro": 15, "alternatives": 10}


def load_latest_thesis() -> dict:
    files = sorted(glob.glob("outputs/ai_pm_theses/*.json"))
    if not files:
        return {}
    try:
        with open(files[-1]) as f:
            return json.load(f)
    except Exception:
        return {}


# ── Construction / book reasoning helpers ─────────────────────────────────────

def _sparkline_paths(symbols: list[str], n: int = 30) -> dict[str, dict]:
    """Last-n close series per symbol → 64x20 inline-SVG path string."""
    try:
        df = pd.read_parquet(PRICES_LIVE_PATH, columns=["symbol", "date", "close"])
    except Exception:
        return {}
    out: dict[str, dict] = {}
    W, H = 64, 20
    want = set(symbols)
    for sym, grp in df[df["symbol"].isin(want)].groupby("symbol"):
        ys = grp.sort_values("date")["close"].dropna().tail(n).tolist()
        if len(ys) < 2:
            continue
        lo, hi = min(ys), max(ys)
        rng = (hi - lo) or 1.0
        pts = []
        for i, y in enumerate(ys):
            x = round(i * (W - 2) / (len(ys) - 1) + 1, 1)
            yy = round(2 + (1 - (y - lo) / rng) * (H - 4), 1)
            pts.append(f"{x},{yy}")
        out[str(sym)] = {"d": "M" + " L".join(pts), "up": ys[-1] >= ys[0]}
    return out


_REDACTION_LABELS = [
    "Sleeve attribution", "Entry signal", "Risk weighting", "Cluster cap",
    "Regime threshold", "Correlation guard", "Exposure rule",
]


def _redaction_label(sym: str) -> str:
    return _REDACTION_LABELS[sum(ord(c) for c in sym) % len(_REDACTION_LABELS)]


def _position_reasoning(sym: str, verdict: dict, prethesis: dict) -> dict:
    """Qualitative-only reasoning per holding. Never emits model internals."""
    why = "Held on the composite cross-sectional ranking."
    for n in (prethesis or {}).get("high_conviction_names", []) or []:
        if isinstance(n, dict) and n.get("symbol") == sym and n.get("reason"):
            why = _esc(str(n["reason"]).split(". ")[0].rstrip(".") + ".")
            break
    risks = ((verdict or {}).get("verdict") or {}).get("key_risks") or []
    hit = next((r for r in risks if sym in str(r)), None)
    if hit:
        committee = f"<span class='whyflag'><b>Flagged.</b></span> {_esc(str(hit))}"
        flagged = True
    else:
        committee = "No adversarial flag this cycle."
        flagged = False
    return {"why": why, "committee": committee, "flagged": flagged}


# Seven-stage construction pipeline. Copy here is the confidentiality-audited
# source of truth — it shows WHAT each stage does and the public guardrails,
# and seals every tunable (sleeve identities/weights, params, thresholds, tilts).
_CONSTRUCTION_STAGES = [
    {"n": "01", "name": "Universe &amp; Data", "kind": "visible",
     "desc": "Live prices and macro across a multi-asset universe, joined <b>point-in-time</b> so no decision sees data it couldn't have had.",
     "tail": '<span class="openchip">Visible</span>',
     "chips": [("Yahoo · live", "now"), ("Macro · FRED", ""), ("Point-in-time joins", "")],
     "note": None,
     "more": "Survivorship-hardened: each rebalance rebuilds history using only the universe that existed on that date."},
    {"n": "02", "name": "Signals", "kind": "sealed",
     "desc": "A <b>multi-sleeve alpha model</b> scores every candidate on several independent return drivers, then renormalizes over whatever loaded cleanly.",
     "tail": '<span class="sealchip"><span class="lk">▤</span> Sealed</span>',
     "chips": [("5 sleeves", ""), ("cross-sectional", ""), ("walk-forward validated", "")],
     "note": "▤ Sleeve identities, weights &amp; parameters — sealed by design.",
     "more": "What the sleeves are, how they're weighted, and their parameters are the core of the edge."},
    {"n": "03", "name": "Conviction Ranking", "kind": "sealed",
     "desc": "Sleeve scores combine into a single cross-sectional ranking — the order in which names earn capital.",
     "tail": '<span class="thru">~120 scored</span><span class="sealchip"><span class="lk">▤</span> Blend sealed</span>',
     "chips": [],
     "note": "▤ Blend weights — sealed.",
     "more": "The combiner only blends sleeves that loaded successfully and renormalizes, so a missing data feed degrades gracefully rather than silently zeroing a name."},
    {"n": "04", "name": "Construction", "kind": "mixed",
     "desc": "Top names are selected under <b>sector and single-name caps</b>, then risk-balanced — down-weighting the volatile, capping correlated clusters.",
     "tail": '<span class="thru num">30 → {n_held}</span><span class="sealchip"><span class="lk">▤</span> Tilts sealed</span>',
     "chips": [("Max 10% / name", "cap"), ("1 per sector", "cap"), ("Cluster cap 20%", "cap"), ("inverse-vol tilt", "")],
     "note": "▤ Tilt strengths &amp; redistribution rule — sealed.",
     "more": "The hard caps are public guardrails. The tilt strengths that shape weights between them are not."},
    {"n": "05", "name": "Regime Overlay", "kind": "sealed",
     "desc": "A <b>hidden-state model</b> reads the market and scales gross exposure — leaning in when conditions are calm, cutting when they turn.",
     "tail": '<span class="thru">{regime} · {exposure}</span><span class="sealchip"><span class="lk">▤</span> Thresholds sealed</span>',
     "chips": [("Now · {regime}", "now"), ("3-state HMM", ""), ("vol-targeted", "")],
     "note": "▤ Regime thresholds &amp; exposure curve — sealed.",
     "more": "The current state and its effect on exposure are shown. The transition thresholds that trigger a cut are what make it work."},
    {"n": "06", "name": "Adversarial Gate", "kind": "visible",
     "desc": "Before any order moves, <b>four AI agents debate the proposed book</b> and a judge rules — proceed, reduce, or halt. Advisory by design; it never writes signals.",
     "tail": '<span class="thru">{verdict}</span><span class="openchip">Visible</span>',
     "chips": [],
     "note": None,
     "more": "This is the layer shown in full on the page — see <i>The latest verdict</i> for the exchange. The committee can trim or halt, but it cannot reach into the model."},
    {"n": "07", "name": "AI PM &amp; Execution", "kind": "visible",
     "desc": "An <b>AI PM</b> tilts the book within an authority budget it has earned on verified results; approved orders route to the broker, gated by caps and a kill switch.",
     "tail": '<span class="thru">{authority}</span><span class="openchip">Visible</span>',
     "chips": [("{auth_chip}", "now"), ("&gt;2% NAV needs approval", ""), ("kill switch 15%", "")],
     "note": None,
     "more": "The AI PM's influence is transparent and bounded — its current budget, its picks, and its track record are all on the page."},
]


def _construction_section_html(regime_label: str, verdict: dict, authority: dict,
                               n_held: int, universe_n: int) -> str:
    """Data-driven 'How the book is built' section. Never raises; seals all tunables."""
    regime = _esc(regime_label or "—")
    v = (verdict or {}).get("verdict") or {}
    rec = str(v.get("recommendation") or "—").replace("_", " ")
    conf = v.get("confidence")
    verdict_tail = f"{_esc(rec.title())}" + (f" · {conf:.2f}" if isinstance(conf, (int, float)) else "")
    lvl = (authority or {}).get("level", 0)
    title = _esc((authority or {}).get("title", "Shadow"))
    ai_pct = (authority or {}).get("ai_weight", 0.0) * 100
    authority_tail = f"{ai_pct:.0f}% authority" if lvl else "Shadow"
    auth_chip = f"Level {lvl} · {title} · {ai_pct:.0f}%" if lvl else "Shadow Mode"
    exposure = "full exposure" if regime_label in ("calm_bull", "euphoric") else "scaled exposure"

    uni = f"~{int(round(universe_n / 50.0) * 50)}" if universe_n else "—"
    funnel = [
        ("Universe", uni, 100), ("Scored", "~120", 62), ("Candidates", "30", 34),
        ("Constructed", str(n_held), 22), ("Held live", str(n_held), 22),
    ]
    fn_html = ""
    for i, (lbl, val, w) in enumerate(funnel):
        arrow = '<span class="fnarrow">›</span>' if i < len(funnel) - 1 else ""
        fn_html += (f'<div class="fn"><div class="fnum num">{val}</div>'
                    f'<div class="fnl">{lbl}</div>'
                    f'<div class="fnbar" data-w="{w}"></div>{arrow}</div>')

    subst = {"n_held": n_held, "regime": regime, "exposure": exposure,
             "verdict": verdict_tail, "authority": authority_tail, "auth_chip": auth_chip}
    stages_html = ""
    for s in _CONSTRUCTION_STAGES:
        cls = {"visible": "open-stage", "sealed": "sealed", "mixed": ""}[s["kind"]]
        tail = s["tail"].format(**subst)
        chips = "".join(
            f'<span class="chip {c.format(**subst) if "{" in c else c}">{t.format(**subst)}</span>'
            for (t, c) in s["chips"])
        chips_html = f'<div class="chips">{chips}</div>' if chips else ""
        note = f'<div class="sealednote">{s["note"]}</div>' if s["note"] else ""
        stages_html += (
            f'<div class="stage {cls}"><div class="spine"><div class="snum">{s["n"]}</div>'
            f'<div class="sdot"></div><div class="sline"></div></div>'
            f'<div class="body"><div class="srow"><span class="sname">{s["name"]} '
            f'<span class="scar">▶</span></span><span class="stail">{tail}</span></div>'
            f'<p class="sdesc">{s["desc"]}</p>'
            f'<div class="exp"><div class="ei"><div class="more">{chips_html}'
            f'<p>{s["more"]}</p>{note}</div></div></div></div></div>')

    return (
        '<div class="sec rev"><div class="sec-h"><h2>How the book is built</h2>'
        '<span class="sec-dateline">Data → live order · one daily loop</span></div>'
        '<p class="sec-lede">Every position travels the same chain — from a universe of '
        'hundreds to the names actually held. The visible layers are shown in full; the '
        'model at the core is sealed.</p>'
        f'<div class="funnel">{fn_html}</div>'
        f'<div class="pipe">{stages_html}</div>'
        '<div class="endcap">The quant engine proposes, the committee disposes, the AI PM '
        'tilts within its budget — and a name only reaches the book after clearing every '
        'stage. <b>What you can see is the whole process. What stays sealed is the model.</b>'
        '</div></div>')


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

    def _finite_spy(date_list, forward):
        seq = date_list if forward else list(reversed(date_list))
        for d in seq:
            v = spy.get(d)
            if v is not None and math.isfinite(v):
                return v
        return None

    spy_base = _finite_spy(dates, forward=True)
    spy_cur  = _finite_spy(dates, forward=False)
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

def _earned_authority_html(auth: dict, feedback: dict = None) -> str:
    if not auth:
        return '<p class="empty">data_cache/earned_authority.json not found</p>'
    feedback  = feedback or {}
    level     = auth.get("level", auth.get("phase", 0))
    title     = auth.get("title", ["Shadow","Analyst","Associate","Manager","Director","CEO"][min(level,5)])
    ai_weight = auth.get("ai_weight", 0.0)
    days      = auth.get("days_at_level", 0)
    cooldown  = auth.get("in_cooldown", False)
    reverts   = auth.get("auto_revert_count", 0)
    stuck     = feedback.get("stuck_alert", False)
    edge      = feedback.get("sortino_edge", 0.0)

    titles  = ["Shadow","Analyst","Associate","Manager","Director","CEO"]
    weights = ["0%","5%","15%","30%","50%","75%"]
    windows = ["-","21d","21d","42d","63d","-"]

    steps_html = ""
    for i, (t, w) in enumerate(zip(titles, weights)):
        cls = "ph-active" if i == level else ("ph-done" if i < level else "ph-future")
        steps_html += f'<div class="ph-step {cls}"><div class="ph-dot"></div><div class="ph-name">{t}<br><small>{w}</small></div></div>'
        if i < 5:
            steps_html += f'<div class="ph-line {"ph-done" if i < level else ""}"></div>'

    window   = windows[min(level, 5)]
    eval_win = 42 if level >= 3 else 21
    pct_done = min(100, round(days / max(eval_win, 1) * 100))

    alert_html = ""
    if stuck:
        alert_html += '<div style="background:#d29922;color:#0d1117;padding:6px 12px;border-radius:6px;margin-bottom:8px;font-size:12px">⚠ AI PM stuck at this level 63+ days — review promotion gates</div>'
    if cooldown:
        cd_rem = feedback.get("cooldown_days_remaining", 0)
        alert_html += f'<div style="background:#58a6ff33;color:#58a6ff;padding:6px 12px;border-radius:6px;margin-bottom:8px;font-size:12px">❄ Cooldown active — {cd_rem} trading days remaining</div>'

    edge_color = "#3fb950" if edge >= 0 else "#f85149"
    hit  = feedback.get("hit_rate_21d", 0)
    pf   = feedback.get("profit_factor", 0)
    n_ev = feedback.get("n_decisions_evaluated", 0)

    return f"""
{alert_html}
<div class="level-badge" style="font-size:14px;font-weight:600;margin-bottom:10px">{title} — {weights[min(level,5)]} authority — Day {days} of {window}</div>
<div class="phase-steps">{steps_html}</div>
<div class="phase-progress-bar"><div class="phase-fill" style="width:{pct_done}%"></div></div>
<div class="auth-stats">
  <div class="auth-stat"><div class="as-val" style="color:{edge_color}">{'+'if edge>=0 else ''}{edge:.3f}</div><div class="as-lbl">Sortino edge</div></div>
  <div class="auth-stat"><div class="as-val">{hit:.0%}</div><div class="as-lbl">Hit rate</div></div>
  <div class="auth-stat"><div class="as-val">{pf:.2f}x</div><div class="as-lbl">Profit factor</div></div>
  <div class="auth-stat"><div class="as-val">{n_ev}</div><div class="as-lbl">Decisions scored</div></div>
  <div class="auth-stat"><div class="as-val">{reverts}</div><div class="as-lbl">Demotions</div></div>
</div>
<p class="auth-note">Promotion: all 7 gates must clear simultaneously (Sortino edge, hit rate, profit factor, min decisions, fade rate, regime diversity, cooldown clear).</p>"""


def _promotion_gates_html(feedback: dict) -> str:
    if not feedback or "promotion_gates" not in feedback:
        return '<p class="empty">No promotion gate data yet — starts after first rebalance.</p>'
    gates = feedback["promotion_gates"]
    labels = {
        "sortino_edge": "Sortino edge", "hit_rate": "Hit rate",
        "profit_factor": "Profit factor", "min_decisions": "Min decisions",
        "fade_rate": "Fade rate", "regime_gate": "Regime diversity", "cooldown": "Cooldown clear",
    }
    rows = ""
    for key, label in labels.items():
        g = gates.get(key, {})
        passed = g.get("pass", False)
        val    = g.get("value", "—")
        thr    = g.get("threshold", "")
        icon   = "✓" if passed else "✗"
        color  = "#3fb950" if passed else "#f85149"
        thr_str = f" / need {thr}" if thr else ""
        rows += (f'<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #21262d">'
                 f'<span style="color:{color}">{icon} {label}</span>'
                 f'<span style="color:#8b949e;font-size:12px">{val}{thr_str}</span></div>')
    return f'<div style="font-size:13px">{rows}</div>'


def _counterfactual_chart_html(cfdata: list) -> str:
    if not cfdata:
        return '<p class="empty">No AI PM data yet — starts after next rebalance.</p>'

    # The raw jsonl can contain replayed or out-of-order rows (manual reruns) —
    # keep the last row per date and sort, or the line chart renders jagged loops.
    by_date: dict = {}
    for r in cfdata:
        d = r.get("date")
        if d:
            by_date[d] = r
    cfdata = [by_date[d] for d in sorted(by_date)]

    def cumulative(key):
        v, vals = 1.0, []
        for r in cfdata:
            v *= (1 + (r.get(key) or 0.0))
            vals.append(round((v - 1) * 100, 3))
        return vals

    dates  = [r["date"] for r in cfdata]
    astar  = cumulative("track_astar_return")
    actual = cumulative("track_b_return")
    spy    = cumulative("track_c_return")
    ai_pm  = cumulative("track_d_return")

    # Headline diffs MUST come from the common-window comparison, not from
    # subtracting two tracks each cumulated over its own (disjoint) window —
    # the latter is what produced the fictional "AI PM cost -11.6pp" (A★ data
    # ends 2026-06-04, Track B data only really starts in June, so the full-
    # window difference compares non-overlapping periods).
    from ascent.monitoring.ai_pm_counterfactual import get_cumulative_returns
    _cum = get_cumulative_returns() or {}
    sq          = _cum.get("ai_signal_d_vs_astar")
    impact      = _cum.get("ai_value_add_b_vs_astar")
    n_sq        = _cum.get("n_common_d_astar", 0)
    n_impact    = _cum.get("n_common_b_astar", 0)
    sq_color  = "#8b949e" if sq is None else ("#3fb950" if sq >= 0 else "#f85149")
    imp_color = "#8b949e" if impact is None else ("#3fb950" if impact >= 0 else "#f85149")
    _sq_txt   = "n/a" if sq is None else f"{'+' if sq>=0 else ''}{sq:.2f}pp ({n_sq}d)"
    _imp_txt  = "n/a" if impact is None or n_impact < 5 else f"{'+' if impact>=0 else ''}{impact:.2f}pp ({n_impact}d)"

    # Shadow period: leading stretch where the AI PM track produced no returns
    # (tracks logged, no capital). Annotate it so the flat segment reads as
    # intentional rather than broken.
    first_live = next(
        (i for i, r in enumerate(cfdata) if abs(r.get("track_d_return") or 0.0) > 1e-9),
        None,
    )
    shadow_js = "null"
    if first_live is not None and first_live > 0:
        shadow_js = json.dumps({"start": dates[0], "end": dates[first_live]})
    elif first_live is None and len(dates) > 1:
        shadow_js = json.dumps({"start": dates[0], "end": dates[-1]})

    return f"""
<div style="font-size:12px;color:#8b949e;margin-bottom:8px">
  AI signal quality (D−A★): <span style="color:{sq_color}">{_sq_txt}</span> common window &nbsp;|&nbsp;
  Actual portfolio impact (B−A★): <span style="color:{imp_color}">{_imp_txt}</span> common window
</div>
<div style="position:relative;height:230px"><canvas id="cfChart"></canvas></div>
<script>
(function(){{
  var el = document.getElementById('cfChart');
  if (!el || typeof Chart === 'undefined') return;
  var shadow = {shadow_js};
  var anns = {{}};
  if (shadow) {{
    anns.shadow = {{type:'box', xMin:shadow.start, xMax:shadow.end,
      backgroundColor:'rgba(139,148,158,0.06)', borderWidth:0,
      label:{{display:true, content:'SHADOW PERIOD — tracks scored, no capital',
        position:{{x:'center', y:'start'}}, color:'#6e7681',
        font:{{size:9, weight:'600'}}, backgroundColor:'rgba(13,17,23,0.85)',
        padding:{{x:5,y:2}}, borderRadius:4, yAdjust:4}}}};
  }}
  new Chart(el, {{
    type: 'line',
    data: {{
      labels: {json.dumps(dates)},
      datasets: [
        {{label:'Pure Quant (A★)', data:{json.dumps(astar)},  borderColor:'#8b949e', borderDash:[4,3], pointRadius:0, borderWidth:1.5, fill:false, tension:0.25}},
        {{label:'Actual (B)',       data:{json.dumps(actual)}, borderColor:'#3fb950', pointRadius:0,    borderWidth:2,   fill:false, tension:0.25}},
        {{label:'SPY (C)',          data:{json.dumps(spy)},    borderColor:'#58a6ff', borderDash:[4,3], pointRadius:0, borderWidth:1.5, fill:false, tension:0.25}},
        {{label:'Pure AI PM (D)',   data:{json.dumps(ai_pm)},  borderColor:'#d29922', pointRadius:0,    borderWidth:2,   fill:false, tension:0.25}},
      ]
    }},
    options:{{responsive:true, maintainAspectRatio:false, interaction:{{mode:'index',intersect:false}},
      plugins:{{
        legend:{{position:'bottom', labels:{{color:'#8b949e', boxWidth:10, boxHeight:2, font:{{size:10}}, padding:10}}}},
        tooltip:{{backgroundColor:'#1c2128', borderColor:'#30363d', borderWidth:1, titleColor:'#8b949e',
          bodyColor:'#e6edf3', cornerRadius:10, padding:12, boxPadding:4, usePointStyle:true,
          callbacks:{{label:function(c){{return ' '+c.dataset.label+': '+(c.raw>=0?'+':'')+c.raw.toFixed(2)+'%';}}}}}},
        annotation:{{annotations:anns}}
      }},
      scales:{{
        x:{{ticks:{{color:'#6e7681', maxTicksLimit:8, maxRotation:0, font:{{size:10}}}}, grid:{{color:'#1b212a'}}}},
        y:{{ticks:{{color:'#6e7681', font:{{size:10}}, callback:function(v){{return (v>=0?'+':'')+v.toFixed(1)+'%';}}}}, grid:{{color:'#1b212a'}}}}
      }}
    }}
  }});
}})();
</script>"""


def _override_scorecard_html(decisions: list, feedback: dict) -> str:
    last5 = (feedback.get("last_5_decisions") or [])[-5:]
    if not last5:
        return '<p class="empty">No scored overrides yet — outcomes available after 10 trading days.</p>'

    rows = ""
    for dec in last5:
        sym   = dec.get("symbol", "?")
        ov_t  = dec.get("type", "?")
        ai_w  = dec.get("ai_w", 0)
        qt_w  = dec.get("quant_w", 0)
        r5    = dec.get("outcome_5d")
        r10   = dec.get("outcome_10d")
        r21   = dec.get("outcome_21d")
        verd  = dec.get("verdict", "pending")
        vc    = {"win":"#3fb950","miss":"#f85149","fade":"#d29922","early":"#58a6ff"}.get(verd,"#8b949e")
        fmt   = lambda v: f"{v:+.2%}" if v is not None else "—"
        rows += (f'<tr><td>{dec.get("date","")[:10]}</td><td><b>{sym}</b></td>'
                 f'<td>{ov_t}</td><td>{ai_w:.1%}</td><td>{qt_w:.1%}</td>'
                 f'<td>{fmt(r5)}</td><td>{fmt(r10)}</td><td>{fmt(r21)}</td>'
                 f'<td style="color:{vc}">{verd.upper()}</td></tr>')

    win_rate = feedback.get("hit_rate_21d", 0)
    avg_alpha = feedback.get("amplify_avg_alpha_10d", 0)
    fade_rate = feedback.get("fade_rate", 0)

    return f"""
<div style="font-size:11px;color:#8b949e;margin-bottom:6px">
  Win rate: <b style="color:#e6edf3">{win_rate:.0%}</b> &nbsp;·&nbsp;
  Avg incremental α (10d): <b style="color:#e6edf3">{avg_alpha:+.3%}</b> &nbsp;·&nbsp;
  Fade rate: <b style="color:#e6edf3">{fade_rate:.0%}</b>
</div>
<div style="overflow-x:auto">
<table style="width:100%;font-size:11px;border-collapse:collapse">
  <thead><tr style="color:#8b949e">
    <th>Date</th><th>Symbol</th><th>Type</th><th>AI%</th><th>Quant%</th>
    <th>+5d</th><th>+10d</th><th>+21d</th><th>Result</th>
  </tr></thead>
  <tbody style="color:#e6edf3">{rows}</tbody>
</table></div>"""


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
    allocation: dict = None,
) -> str:
    authority  = authority or {}
    thesis     = thesis or {}
    allocation = allocation or {"us_equities": 60, "international": 15, "macro": 15, "alternatives": 10}

    # ── Chart.js annotation objects ──────────────────────────────────────────
    annotations = {}

    # Regime bands: a faint tint + a labeled boundary edge at the regime start.
    # The full-box label approach was unreadable; the left-edge label marks the
    # transition without painting over the curve.
    for i, band in enumerate(regime_bands):
        if band["start"] not in dates and band["end"] not in dates:
            continue
        tint   = REGIME_COLORS.get(band["label"], "rgba(100,100,100,0.04)")
        accent = REGIME_ACCENT.get(band["label"], "#8b949e")
        annotations[f"regime{i}"] = {
            "type":            "box",
            "xMin":            band["start"],
            "xMax":            band["end"],
            "backgroundColor": tint,
            "borderWidth":     0,
        }
        annotations[f"regimeEdge{i}"] = {
            "type":        "line",
            "xMin":        band["start"],
            "xMax":        band["start"],
            "borderColor": accent + "55",
            "borderWidth": 1,
            "borderDash":  [2, 3],
            "label": {
                "display":         True,
                "content":         band["label"].replace("_", " ").upper(),
                "position":        "start",
                "color":           accent,
                "font":            {"size": 9, "weight": "700"},
                "backgroundColor": "rgba(13,17,23,0.85)",
                "padding":         {"x": 5, "y": 2},
                "borderRadius":    4,
                "xAdjust":         30,
                "yAdjust":         2,
            },
        }

    # Rebalance / milestone lines — compact icon-only markers (legend explains them)
    verdict_icons = {"proceed": "✓", "reduce_size": "↓", "halt_and_review": "✗"}
    for i, ev in enumerate(verdicts):
        if ev["date"] not in dates:
            continue
        if ev["type"] == "debate_only":
            continue  # don't clutter chart with debate-only sessions
        color = VERDICT_COLORS.get(ev["verdict"], "#58a6ff")
        if ev["type"] == "milestone":
            color = "#58a6ff"
            icon  = "★"
        else:
            icon = verdict_icons.get(ev["verdict"], "↻")
        annotations[f"reb{i}"] = {
            "type":        "line",
            "xMin":        ev["date"],
            "xMax":        ev["date"],
            "borderColor": color + "99",
            "borderWidth": 1.5,
            "borderDash":  [5, 4],
            "label": {
                "display":         True,
                "content":         icon,
                "position":        "end",
                "backgroundColor": color + "26",
                "color":           color,
                "font":            {"size": 11, "weight": "700"},
                "padding":         {"x": 5, "y": 2},
                "borderRadius":    10,
                "yAdjust":         4,
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
    # Load new AI PM data
    _feedback      = load_perf_feedback()
    _cfdata        = load_counterfactual()
    _decisions     = load_ai_pm_decisions()
    auth_html      = _earned_authority_html(authority, feedback=_feedback)
    gates_html     = _promotion_gates_html(_feedback)
    cf_chart_html  = _counterfactual_chart_html(_cfdata)
    scorecard_html = _override_scorecard_html(_decisions, _feedback)
    thesis_html    = _thesis_html(thesis)
    debate_html    = _debate_html(verdicts)

    # AI PM phase chip + allocation note from live authority state
    _lvl    = authority.get("level", 0)
    _title  = authority.get("title", "Shadow")
    _ai_pct = authority.get("ai_weight", 0.0) * 100
    if _lvl > 0:
        phase_chip = f"Level {_lvl} — {_title} · {_ai_pct:.0f}% Authority"
    else:
        phase_chip = "Phase 0 — Shadow Mode"
    alloc_note = f"AI PM Level {_lvl} ({_title}) · {_ai_pct:.0f}% active-weight budget"

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
<title>Ascent Capital — Autonomous Multi-Agent Investment System</title>
<meta name="description" content="An autonomous, multi-agent investment system — regime-aware quant engine gated by an adversarial AI debate committee, with an AI PM earning allocative authority through verified performance. Live paper trading.">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;line-height:1.5}}
a{{color:#58a6ff;text-decoration:none}}
.num{{font-variant-numeric:tabular-nums}}
.wrap{{max-width:1300px;margin:0 auto;padding:0 24px 80px}}
.kicker{{font-size:11px;font-weight:700;color:#6e7681;text-transform:uppercase;letter-spacing:1.2px;margin-bottom:6px}}
.jsfx .fx{{opacity:0;transform:translateY(10px);transition:opacity .55s ease,transform .55s ease}}
.jsfx .fx.in{{opacity:1;transform:none}}
@media(prefers-reduced-motion:reduce){{.jsfx .fx{{opacity:1;transform:none;transition:none}}}}
.hero{{background:radial-gradient(1200px 420px at 18% -10%,#16202e 0%,#0d1117 55%),linear-gradient(135deg,#0d1117 0%,#11161d 60%,#0d1117 100%);border-bottom:1px solid #21262d;padding:52px 24px 44px;margin:0 0 36px}}
.hero-inner{{display:grid;grid-template-columns:1fr auto;gap:36px;align-items:start;max-width:1300px;margin:0 auto}}
@media(max-width:760px){{.hero-inner{{grid-template-columns:1fr;gap:20px}}.regime-ring{{justify-self:start;align-self:start}}}}
.hero-top{{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:14px}}
.hero-brand h1{{font-size:34px;font-weight:800;letter-spacing:-1px}}
.hero-brand h1 em{{color:#58a6ff;font-style:normal}}
.hero-pitch{{color:#c9d1d9;font-size:16px;line-height:1.55;max-width:760px;margin-bottom:10px;font-weight:500}}
.hero-sub{{color:#6e7681;font-size:12.5px;max-width:720px;margin-bottom:4px}}
.live-pill{{display:inline-flex;align-items:center;gap:6px;background:#3fb95011;border:1px solid #3fb95033;color:#3fb950;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700;letter-spacing:.5px;text-transform:uppercase;white-space:nowrap}}
.live-pill::before{{content:'';width:6px;height:6px;border-radius:50%;background:#3fb950;animation:pulse 1.6s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}
.hero-numbers{{display:flex;gap:0;margin-top:30px;flex-wrap:wrap;align-items:flex-end}}
.hn{{padding:0 28px 0 0;border-right:1px solid #21262d;margin-right:28px;margin-top:10px}}
.hn:last-child{{border-right:none;margin-right:0}}
.hn-label{{font-size:11px;color:#6e7681;text-transform:uppercase;letter-spacing:.6px;margin-bottom:4px;display:flex;align-items:center;gap:8px}}
.hn-val{{font-size:24px;font-weight:700;letter-spacing:-0.5px;font-variant-numeric:tabular-nums}}
.hn-val.main{{font-size:clamp(42px,5.5vw,56px);font-weight:800;letter-spacing:-1.5px;line-height:1.05;color:#e6edf3}}
.fresh{{font-size:10px;font-weight:600;color:#3fb950;background:#3fb9500d;border:1px solid #3fb95026;border-radius:10px;padding:1px 8px;letter-spacing:.4px;text-transform:none;white-space:nowrap}}
.regime-ring{{text-align:center;background:#161b22;border:2px solid #30363d;border-radius:16px;padding:20px 28px;min-width:150px;align-self:center}}
.regime-ring.calm_bull{{border-color:#3fb95066;background:#3fb95008;animation:ringGlow 3.2s ease-in-out infinite}}
.regime-ring.stressed{{border-color:#d2992266;background:#d2992208}}
.regime-ring.crisis{{border-color:#f8514966;background:#f8514908}}
.regime-ring.neutral{{border-color:#58a6ff66;background:#58a6ff08}}
@keyframes ringGlow{{0%,100%{{box-shadow:0 0 0 rgba(63,185,80,0)}}50%{{box-shadow:0 0 22px rgba(63,185,80,0.13)}}}}
@media(prefers-reduced-motion:reduce){{.regime-ring.calm_bull{{animation:none}}}}
.regime-name{{font-size:16px;font-weight:700;text-transform:uppercase;letter-spacing:1px}}
.regime-ring.calm_bull .regime-name{{color:#3fb950}}
.regime-ring.stressed  .regime-name{{color:#d29922}}
.regime-ring.crisis    .regime-name{{color:#f85149}}
.regime-ring.neutral   .regime-name{{color:#58a6ff}}
.regime-sub{{font-size:10px;color:#6e7681;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:24px}}
.stat-card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 16px}}
.stat-label{{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:6px}}
.stat-value{{font-size:21px;font-weight:700;font-variant-numeric:tabular-nums}}
.stat-note{{font-size:10px;color:#6e7681;margin-top:3px}}
.chart-card,.card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px 24px;margin-bottom:24px}}
.chart-card h2,.card h2{{font-size:15px;font-weight:600;color:#8b949e;margin-bottom:14px}}
.arch{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px 28px;margin-bottom:24px}}
.arch-head{{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:18px}}
.arch-head h2{{font-size:16px;font-weight:700;color:#e6edf3}}
.arch-head span{{font-size:12px;color:#6e7681}}
.arch-flow{{display:flex;align-items:stretch;gap:10px;flex-wrap:nowrap}}
.arch-node{{flex:1;min-width:0;background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:14px 14px 12px;text-align:center;position:relative;transition:border-color .25s}}
.arch-node:hover{{border-color:#58a6ff44}}
.arch-node.ai{{border-color:#a78bfa33}}
.arch-node.ai:hover{{border-color:#a78bfa66}}
.an-icon{{font-size:18px;margin-bottom:6px;opacity:.85}}
.an-name{{font-size:13px;font-weight:700;color:#e6edf3;margin-bottom:4px;white-space:nowrap}}
.an-sub{{font-size:10.5px;color:#6e7681;line-height:1.45}}
.an-tag{{position:absolute;top:-8px;right:8px;font-size:9px;font-weight:700;letter-spacing:.5px;color:#a78bfa;background:#0d1117;border:1px solid #a78bfa44;border-radius:8px;padding:0 6px;text-transform:uppercase}}
.arch-arrow{{display:flex;align-items:center;color:#30363d;font-size:16px;flex-shrink:0;user-select:none}}
.arch-note{{font-size:11.5px;color:#6e7681;margin-top:14px;line-height:1.55;border-top:1px solid #21262d;padding-top:12px}}
@media(max-width:900px){{.arch-flow{{flex-direction:column}}.arch-arrow{{justify-content:center;transform:rotate(90deg);height:14px}}.an-name{{white-space:normal}}}}
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
.ph-step.ph-active .ph-dot{{background:#a78bfa;border-color:#a78bfa;box-shadow:0 0 8px #a78bfa66}}
.ph-step.ph-done .ph-dot{{background:#3fb950;border-color:#3fb950}}
.ph-name{{font-size:10px;color:#6e7681}}
.ph-step.ph-active .ph-name{{color:#a78bfa;font-weight:600}}
.ph-line{{flex:1;height:2px;background:#21262d;margin:0 4px;margin-bottom:14px}}
.ph-line.ph-done{{background:#3fb950}}
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
        <div class="hero-top">
          <h1>Ascent <em>Capital</em></h1>
          <div class="live-pill">Paper Trading · Live</div>
        </div>
        <p class="hero-pitch">An autonomous, multi-agent investment system — a regime-aware quant engine gated by an adversarial AI debate committee, with an AI PM earning allocative authority through verified performance.</p>
        <p class="hero-sub">Runs daily on live market data and executes through the Alpaca paper API. Built and operated by one person. Not a backtest — and not live capital.</p>
      </div>
      <div class="hero-numbers">
        <div class="hn"><div class="hn-label">Net Asset Value <span class="fresh">Updated {generated_at}</span></div><div class="hn-val main" id="nav-ctr">${nav:,.0f}</div></div>
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
<div class="chart-card fx">
  <div class="chart-title-row">
    <div><div class="kicker">Performance</div><h2 style="margin-bottom:0">Equity Curve — Portfolio vs SPY</h2></div>
    <div class="regime-chip {regime_class}">{regime_label}</div>
  </div>
  <div class="legend">
    <div class="leg"><div class="leg-dot" style="background:#58a6ff"></div>Ascent Capital</div>
    <div class="leg"><div class="leg-dash"></div>SPY</div>
    <div class="leg"><div class="leg-dot" style="background:#3fb950"></div>✓ Proceed</div>
    <div class="leg"><div class="leg-dot" style="background:#d29922"></div>↓ Reduce Size</div>
    <div class="leg"><div class="leg-dot" style="background:#f85149"></div>✗ Halt &amp; Review</div>
  </div>
  <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
  <p class="chart-note">⚠ Sharpe annualized from {n_days} trading days (SE {sharpe_se_str}) — not statistically significant. Walk-forward OOS Sharpe (0.518, 2020–2026) is the rigorous number. Live since April 1, 2026.</p>
</div>
<div class="grid2 fx">
  <div class="chart-card" style="margin-bottom:0"><h2>Drawdown from Peak</h2><div class="chart-wrap short"><canvas id="ddChart"></canvas></div></div>
  <div class="chart-card" style="margin-bottom:0"><h2>Cumulative Alpha vs SPY</h2><div class="chart-wrap short"><canvas id="alphaChart"></canvas></div></div>
</div>
<div style="margin-bottom:24px"></div>
<div class="arch fx">
  <div class="arch-head">
    <div><div class="kicker">System Architecture</div><h2>One autonomous daily loop</h2></div>
    <span>every position passes the full chain before capital moves</span>
  </div>
  <div class="arch-flow">
    <div class="arch-node"><div class="an-icon">▦</div><div class="an-name">Quant Engine</div><div class="an-sub">4 specialist agents · multi-sleeve alpha · walk-forward validated</div></div>
    <div class="arch-arrow">›</div>
    <div class="arch-node"><div class="an-icon">◍</div><div class="an-name">Regime Model</div><div class="an-sub">hidden-state market detection · scales exposure to conditions</div></div>
    <div class="arch-arrow">›</div>
    <div class="arch-node ai"><div class="an-tag">LLM</div><div class="an-icon">⚖</div><div class="an-name">AI Debate</div><div class="an-sub">bull · bear · devil's advocate · judge — can trim or halt any trade</div></div>
    <div class="arch-arrow">›</div>
    <div class="arch-node ai"><div class="an-tag">LLM</div><div class="an-icon">⬡</div><div class="an-name">AI PM</div><div class="an-sub">independent thesis before quant runs · authority earned by results</div></div>
    <div class="arch-arrow">›</div>
    <div class="arch-node"><div class="an-icon">⇄</div><div class="an-name">Execution</div><div class="an-sub">position caps · kill switch · Alpaca paper API</div></div>
  </div>
  <p class="arch-note">The quant engine proposes; the debate committee disposes. The AI PM forms its own thesis before seeing quant output and can tilt the book only within an authority budget it has earned — verified on out-of-sample results, never granted.</p>
</div>
<div class="ai-section fx">
  <div class="ai-header">
    <div class="ai-header-left">
      <div class="ai-glyph">⚖</div>
      <div><h2>Decision Process — Adversarial Debate</h2><p>Before any order is placed, four AI agents argue the book — and the judge's verdict gates execution.</p></div>
    </div>
  </div>
  <div style="background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:18px 20px">
    <p class="debate-intro">Bull and bear argue from different evidence; the devil's advocate attacks portfolio convexity; a judge synthesizes a verdict — proceed, reduce size, or halt. Interventions are tracked and scored. Click any row to read the full debate.</p>
    {debate_html}
  </div>
</div>
<div class="ai-section fx">
  <div class="ai-header">
    <div class="ai-header-left">
      <div class="ai-glyph">⬡</div>
      <div><h2>AI Portfolio Manager — Earned Authority</h2><p>The AI PM's influence on the book is a budget it earns: promoted on verified out-of-sample results, demoted on misses.</p></div>
    </div>
    <div class="ai-phase-chip">{phase_chip}</div>
  </div>
  <div class="grid3">
    <div class="ai-card"><h3>Earned Authority</h3>{auth_html}</div>
    <div class="ai-card"><h3>Capital Allocation</h3><div class="alloc-chart-wrap"><canvas id="allocChart"></canvas></div><p class="alloc-note">{alloc_note}</p></div>
    <div class="ai-card"><h3>Latest AI PM Decision</h3>{thesis_html}</div>
  </div>
  <div class="grid3" style="margin-top:16px">
    <div class="ai-card"><h3>Promotion Gates</h3>{gates_html}</div>
    <div class="ai-card" style="grid-column:span 2"><h3>Four-Track Counterfactual (A★ / Actual / SPY / Pure AI PM)</h3>{cf_chart_html}</div>
  </div>
  <div class="ai-card" style="margin-top:16px"><h3>Override Scorecard</h3>{scorecard_html}</div>
</div>
<div class="grid2 fx">
  <div class="card"><h2>Current Holdings</h2>{pos_html}</div>
  <div class="card"><h2>Event Timeline</h2><div class="timeline">{tl_html}</div></div>
</div>
<footer>Updated {generated_at} &nbsp;·&nbsp; Alpaca paper API + Yahoo Finance &nbsp;·&nbsp; Paper trading — performance is not indicative of live-capital results &nbsp;·&nbsp; <a href="https://github.com/ScottDongKhang/Ascent_Capital">Source on GitHub</a></footer>
</div>
<script>
Chart.register(window['chartjs-plugin-annotation']);
const dates={dates_js},portNAV={port_js},spyNAV={spy_js},startNAV={start_nav_js};
const ANNOTATIONS={annotations_js};
const REDUCED=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
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
// Section fade-in on scroll. Scoped behind .jsfx so content is never hidden
// when JS or IntersectionObserver is unavailable; respects reduced motion.
if('IntersectionObserver' in window && !REDUCED){{
  document.documentElement.classList.add('jsfx');
  const io=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting){{e.target.classList.add('in');io.unobserve(e.target);}}}}),{{threshold:.07}});
  document.querySelectorAll('.fx').forEach(el=>io.observe(el));
}}
const TT={{backgroundColor:'#1c2128',borderColor:'#30363d',borderWidth:1,titleColor:'#8b949e',bodyColor:'#e6edf3',
  padding:12,cornerRadius:10,boxPadding:5,usePointStyle:true,titleFont:{{weight:'600',size:11}},bodyFont:{{size:12}},
  bodySpacing:5,caretSize:0,displayColors:true}};
const SX={{ticks:{{color:'#6e7681',maxTicksLimit:9,maxRotation:0,font:{{size:10.5}}}},grid:{{color:'#1b212a'}}}};
const SY={{ticks:{{color:'#6e7681',font:{{size:10.5}}}},grid:{{color:'#1b212a'}}}};
// Gradient fill under the portfolio line (rebuilt per layout to track chart area)
function navGradient(ctx){{
  const a=ctx.chart.chartArea;
  if(!a)return 'rgba(88,166,255,0.05)';
  const g=ctx.chart.ctx.createLinearGradient(0,a.top,0,a.bottom);
  g.addColorStop(0,'rgba(88,166,255,0.26)');g.addColorStop(.55,'rgba(88,166,255,0.07)');g.addColorStop(1,'rgba(88,166,255,0)');
  return g;
}}
// Soft glow dot on the live endpoint of the portfolio series
const endGlow={{id:'endGlow',afterDatasetsDraw(c){{
  const m=c.getDatasetMeta(0);const p=m.data[m.data.length-1];if(!p)return;
  const x=p.x,y=p.y;
  if(!isFinite(x)||!isFinite(y))return;  // mid draw-on animation
  const g=c.ctx;g.save();
  const r=g.createRadialGradient(x,y,0,x,y,13);
  r.addColorStop(0,'rgba(88,166,255,0.5)');r.addColorStop(1,'rgba(88,166,255,0)');
  g.fillStyle=r;g.beginPath();g.arc(x,y,13,0,Math.PI*2);g.fill();
  g.fillStyle='#58a6ff';g.beginPath();g.arc(x,y,3.5,0,Math.PI*2);g.fill();
  g.strokeStyle='#0d1117';g.lineWidth=1.5;g.stroke();g.restore();
}}}};
// Progressive left-to-right draw on load (~1.2s, Chart.js delayed-point pattern)
const STEP=1200/Math.max(dates.length,1);
const prevY=ctx=>{{
  if(ctx.index===0)return ctx.chart.scales.y.getPixelForValue(startNAV);
  const d=ctx.chart.getDatasetMeta(ctx.datasetIndex).data[ctx.index-1];
  return d?d.getProps(['y'],true).y:ctx.chart.scales.y.getPixelForValue(startNAV);
}};
const drawOn=REDUCED?undefined:{{
  x:{{type:'number',easing:'linear',duration:STEP,from:NaN,
     delay(ctx){{if(ctx.type!=='data'||ctx.xStarted)return 0;ctx.xStarted=true;return ctx.index*STEP;}}}},
  y:{{type:'number',easing:'linear',duration:STEP,from:prevY,
     delay(ctx){{if(ctx.type!=='data'||ctx.yStarted)return 0;ctx.yStarted=true;return ctx.index*STEP;}}}}
}};
new Chart(document.getElementById('equityChart'),{{type:'line',data:{{labels:dates,datasets:[
  {{label:'Ascent Capital',data:portNAV,borderColor:'#58a6ff',backgroundColor:navGradient,borderWidth:2.5,pointRadius:0,pointHoverRadius:5,pointHoverBackgroundColor:'#58a6ff',fill:true,tension:0.3}},
  {{label:'SPY',data:spyNAV,borderColor:'#6b7280',backgroundColor:'transparent',borderWidth:1.5,pointRadius:0,pointHoverRadius:4,pointHoverBackgroundColor:'#6b7280',borderDash:[6,4],fill:false,tension:0.3}},
]}},options:{{responsive:true,maintainAspectRatio:false,interaction:{{mode:'index',intersect:false}},
  animation:drawOn,
  plugins:{{legend:{{display:false}},tooltip:{{...TT,callbacks:{{label:ctx=>{{
    const v=ctx.raw;if(!v)return'';const r=((v-startNAV)/startNAV*100),s=r>=0?'+':'';
    return ` ${{ctx.dataset.label}}: $${{v.toLocaleString('en-US',{{maximumFractionDigits:0}})}} (${{s}}${{r.toFixed(2)}}%)`;
  }}}}}},annotation:{{annotations:ANNOTATIONS}}}},
  scales:{{x:SX,y:{{...SY,ticks:{{...SY.ticks,callback:v=>'$'+Math.round(v).toLocaleString('en-US')}}}}}}
}},plugins:[endGlow]}});
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
  datasets:[{{data:[{allocation['us_equities']},{allocation['international']},{allocation['macro']},{allocation['alternatives']}],backgroundColor:['#58a6ff','#3fb950','#a78bfa','#d29922'],borderWidth:0,hoverOffset:4}}]
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
    allocation  = load_latest_allocation()
    print(f"      Allocation: US={allocation['us_equities']}% Intl={allocation['international']}% "
          f"Macro={allocation['macro']}% Alt={allocation['alternatives']}%")
    stats               = compute_stats(records, spy)
    dates, port, spy_v  = build_chart_data(records, spy)
    generated_at        = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html                = build_html(dates, port, spy_v, verdicts, regime_bands,
                                     stats, positions, generated_at,
                                     authority=authority, thesis=thesis, allocation=allocation)

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
