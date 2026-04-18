"""
dashboard_20in20.py
====================
Generates a self-contained dark-terminal HTML intel report for 20in20.
Reads from outputs/20in20/ — run AFTER run_20in20.py.

Usage:
    python dashboard_20in20.py --asof 2026-03-21
    python dashboard_20in20.py --asof 2026-03-21 --open
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path, fallback=None):
    if not path.exists():
        return fallback
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text())
        if path.suffix == ".csv":
            return pd.read_csv(path)
    except Exception as e:
        print(f"[dashboard] Warning: could not load {path} — {e}")
    return fallback


def _fmt_pct(v, decimals=1):
    try:
        return f"{float(v)*100:+.{decimals}f}%"
    except Exception:
        return "—"


def _fmt_z(v):
    try:
        return f"{float(v):+.2f}"
    except Exception:
        return "—"


def _clean(s: str) -> str:
    """Remove underscores and title-case for display."""
    return str(s).replace("_", " ").title()


def _find_prev_asof(base_dir: Path, asof: str):
    """Find the most recent prior asof from existing regime_summary files."""
    import re
    pat   = re.compile(r"regime_summary_(\d{4}-\d{2}-\d{2})\.json")
    dates = [m.group(1) for f in base_dir.glob("regime_summary_*.json")
             if (m := pat.match(f.name)) and m.group(1) < asof]
    return max(dates) if dates else None


def _load_delta(base_dir: Path, asof: str) -> dict:
    """Compare current vs prior period. Returns delta dict."""
    prev = _find_prev_asof(base_dir, asof)
    if not prev:
        return {}

    curr_themes  = _load(base_dir / "tables" / f"themes_{asof}.csv",   pd.DataFrame())
    prev_themes  = _load(base_dir / "tables" / f"themes_{prev}.csv",   pd.DataFrame())
    curr_regime  = _load(base_dir / f"regime_summary_{asof}.json",     {})
    prev_regime  = _load(base_dir / f"regime_summary_{prev}.json",     {})

    out = {"prev_asof": prev}

    # Regime change flag
    cl = curr_regime.get("regime_label", "")
    pl = prev_regime.get("regime_label", "")
    if cl and pl:
        out["regime_changed"] = cl != pl
        out["prev_regime"]    = _clean(pl)
        out["curr_regime"]    = _clean(cl)

    # Theme score deltas
    if not curr_themes.empty and not prev_themes.empty and "theme" in curr_themes.columns:
        merged = curr_themes.merge(prev_themes, on="theme", suffixes=("_c","_p"), how="left")
        deltas = []
        for _, row in merged.iterrows():
            c = row.get("score_c")
            p = row.get("score_p")
            if pd.notna(c) and pd.notna(p):
                chg = float(c) - float(p)
                if abs(chg) > 0.01:
                    deltas.append({
                        "theme":     _clean(row["theme"]),
                        "change":    round(chg, 4),
                        "direction": "improved" if chg > 0 else "weakened",
                    })
        deltas.sort(key=lambda x: abs(x["change"]), reverse=True)
        out["theme_deltas"] = deltas[:4]

    return out


# ── HTML builder ──────────────────────────────────────────────────────────────

def build_html(asof: str, base_dir: Path) -> str:
    regime        = _load(base_dir / f"regime_summary_{asof}.json", {})
    themes        = _load(base_dir / "tables" / f"themes_{asof}.csv",            pd.DataFrame())
    rv            = _load(base_dir / "tables" / f"relative_value_{asof}.csv",    pd.DataFrame())
    sc            = _load(base_dir / "tables" / f"scenarios_{asof}.csv",          pd.DataFrame())
    memo          = _load(base_dir / "memos"  / f"market_memo_{asof}.json",       {})
    comps_summary = _load(base_dir / "tables" / f"comps_theme_summary_{asof}.csv", pd.DataFrame())
    delta         = _load_delta(base_dir, asof)

    posture      = regime.get("posture",         "unknown").upper()
    regime_label = _clean(regime.get("regime_label", "unknown"))
    confidence   = regime.get("confidence",      0)
    conf_pct     = f"{float(confidence)*100:.0f}%"
    risk_mult    = float(regime.get("risk_multiplier", 1.0))
    days_in      = regime.get("days_in_regime",  0)
    notes        = regime.get("notes",           "")
    drivers      = regime.get("drivers",         [])
    headline     = memo.get("headline", f"Posture: {posture}")
    takeaways    = memo.get("takeaways", [])
    disclaimer   = memo.get("disclaimer", "For internal discussion only. Not investment advice.")

    COLORS = {
        "CONSTRUCTIVE": "#22c55e",
        "SELECTIVE":    "#84cc16",
        "NEUTRAL":      "#94a3b8",
        "DEFENSIVE":    "#f59e0b",
        "CRISIS":       "#ef4444",
        "UNCERTAIN":    "#64748b",
    }
    pc = COLORS.get(posture, "#94a3b8")

    # ── Stat cards ────────────────────────────────────────────────────────────
    regime_cards = f"""
      <div class="stat-card">
        <div class="stat-label">REGIME</div>
        <div class="stat-value" style="color:{pc}">{regime_label}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">POSTURE</div>
        <div class="stat-value" style="color:{pc}">{posture}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">CONFIDENCE</div>
        <div class="stat-value">{conf_pct}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">RISK MULT</div>
        <div class="stat-value" style="color:{pc}">{risk_mult:.0%}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">DAYS IN REGIME</div>
        <div class="stat-value">{days_in}</div>
      </div>"""

    # ── Drivers ───────────────────────────────────────────────────────────────
    if drivers:
        items = "".join(
            f'<div class="driver-item">'
            f'<span class="dot" style="background:{pc}"></span>{d}'
            f'</div>'
            for d in drivers
        )
        drivers_section = f"""
        <div class="panel side-panel">
          <div class="panel-header">REGIME DRIVERS</div>
          <div class="flex-body">{items}</div>
        </div>"""
    else:
        drivers_section = ""

    # ── What changed ──────────────────────────────────────────────────────────
    if delta:
        items = []
        if delta.get("regime_changed"):
            items.append(
                f'<div class="delta-item warn">⚡ Regime shifted: '
                f'{delta["prev_regime"]} → {delta["curr_regime"]}</div>'
            )
        for td in delta.get("theme_deltas", []):
            col   = "#22c55e" if td["direction"] == "improved" else "#ef4444"
            arrow = "▲" if td["direction"] == "improved" else "▼"
            items.append(
                f'<div class="delta-item">'
                f'<span style="color:{col}">{arrow} {td["theme"]}</span> '
                f'{td["direction"]} {_fmt_pct(td["change"])}'
                f'</div>'
            )
        if not items:
            items = ['<div class="delta-item muted">No significant changes vs prior period.</div>']
        delta_section = f"""
        <div class="panel side-panel">
          <div class="panel-header">WHAT CHANGED vs {delta.get('prev_asof','')}</div>
          <div class="flex-body">{"".join(items)}</div>
        </div>"""
    else:
        delta_section = ""

    # Show drivers + delta side by side if both exist, else full-width
    if drivers_section and delta_section:
        meta_row = f'<div class="grid-2" style="margin-bottom:20px">{drivers_section}{delta_section}</div>'
    elif drivers_section:
        meta_row = f'<div style="margin-bottom:20px">{drivers_section}</div>'
    elif delta_section:
        meta_row = f'<div style="margin-bottom:20px">{delta_section}</div>'
    else:
        meta_row = ""

    # ── Themes ────────────────────────────────────────────────────────────────
    themes_rows = ""
    if not themes.empty and "theme" in themes.columns and "score" in themes.columns:
        for _, row in themes.iterrows():
            score = float(row["score"])
            col   = "#22c55e" if score >= 0 else "#ef4444"
            bar_w = min(abs(score) * 300, 100)
            arrow = "▲" if score >= 0 else "▼"
            n     = int(row.get("n_tickers", 0))
            themes_rows += f"""
            <tr>
              <td>{_clean(row['theme'])}</td>
              <td style="color:{col};font-weight:600">{arrow} {_fmt_pct(score)}</td>
              <td><div class="bar-wrap"><div class="bar" style="width:{bar_w:.0f}%;background:{col}"></div></div></td>
              <td class="muted">{n}</td>
            </tr>"""

    # ── RV ────────────────────────────────────────────────────────────────────
    def rv_rows(df, cond, col):
        out = ""
        if df.empty or "rv_z" not in df.columns:
            return out
        for _, row in df[cond(df["rv_z"])].iterrows():
            theme_val = row.get("theme", "")
            theme_disp = _clean(theme_val) if theme_val and str(theme_val) != "nan" else "—"
            out += f"""
            <tr>
              <td style="font-weight:600">{row.get('ticker','')}</td>
              <td class="muted">{theme_disp}</td>
              <td style="color:{col};font-weight:600">{_fmt_z(row.get('rv_z',0))}</td>
            </tr>"""
        return out

    ext_rows = rv_rows(rv, lambda z: z >  0.5, "#f59e0b") or "<tr><td colspan='3' class='muted'>—</td></tr>"
    dep_rows = rv_rows(rv, lambda z: z < -0.5, "#22c55e") or "<tr><td colspan='3' class='muted'>—</td></tr>"

    # ── Scenarios ─────────────────────────────────────────────────────────────
    sc_rows = ""
    if sc is not None and not sc.empty:
        for _, row in sc.iterrows():
            pnl      = float(row.get("pnl_est", 0))
            col      = "#ef4444" if pnl < 0 else "#22c55e"
            sev      = str(row.get("severity", ""))
            sev_col  = {"severe": "#ef4444", "moderate": "#f59e0b", "mild": "#64748b"}.get(sev, "#64748b")
            desc_raw = str(row.get("description", ""))
            desc     = desc_raw[:65] + "…" if len(desc_raw) > 65 else desc_raw
            exposed  = str(row.get("most_exposed", ""))
            sc_rows += f"""
            <tr>
              <td>
                <div style="font-weight:600">{row.get('scenario','')}</div>
                <div class="muted" style="font-size:11px;margin-top:2px">{desc}</div>
              </td>
              <td style="color:{col};font-weight:600">{_fmt_pct(pnl)}</td>
              <td><span class="badge" style="background:{sev_col}22;color:{sev_col}">{sev}</span></td>
              <td class="muted" style="font-size:11px">{exposed}</td>
            </tr>"""
    if not sc_rows:
        sc_rows = "<tr><td colspan='4' class='muted'>No scenario data</td></tr>"

    # ── Public comps ──────────────────────────────────────────────────────────
    if comps_summary is not None and not comps_summary.empty:
        c_rows = ""
        for _, row in comps_summary.iterrows():
            ts  = row.get("trend_score")
            ts_val = float(ts) if pd.notna(ts) else None
            tc  = "#22c55e" if ts_val and ts_val > 0 else "#ef4444" if ts_val and ts_val < 0 else "#94a3b8"
            flags     = str(row.get("risk_flags", "clean"))
            flag_col  = "#ef4444" if flags != "clean" else "#22c55e"
            ts_disp   = f"{ts_val:+.2f}" if ts_val is not None else "—"
            c_rows += f"""
            <tr>
              <td>{_clean(row.get('theme',''))}</td>
              <td style="color:{tc};font-weight:600">{ts_disp}</td>
              <td class="muted">{_fmt_pct(row.get('return_1m'))}</td>
              <td class="muted">{_fmt_pct(row.get('return_3m'))}</td>
              <td class="muted">{f"{float(row['vol_3m']):.0%}" if pd.notna(row.get('vol_3m')) else '—'}</td>
              <td class="muted">{_fmt_pct(row.get('drawdown_6m'))}</td>
              <td><span class="badge" style="color:{flag_col}">{flags}</span></td>
            </tr>"""
        comps_panel = f"""
        <div class="panel" style="margin-bottom:20px">
          <div class="panel-header">CATEGORY APPETITE — PUBLIC COMPS</div>
          <table>
            <thead><tr>
              <th>THEME</th><th>TREND</th><th>1M</th><th>3M</th>
              <th>VOL</th><th>MAX DD</th><th>FLAGS</th>
            </tr></thead>
            <tbody>{c_rows}</tbody>
          </table>
        </div>"""
    else:
        comps_panel = ""

    # ── Takeaways ─────────────────────────────────────────────────────────────
    ta_html = "".join(f"<li>{t}</li>" for t in takeaways) if takeaways else "<li>—</li>"

    # ── Notes ─────────────────────────────────────────────────────────────────
    notes_html = (
        f'<div class="notes-strip" style="border-left:3px solid {pc}">{notes}</div>'
        if notes else ""
    )

    # ══════════════════════════════════════════════════════════════════════════
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Ascent Intel | 20in20 | {asof}</title>
<style>
:root{{--bg:#0a0e1a;--surface:#111827;--border:#1e2a3a;--text:#e2e8f0;--muted:#64748b;--pc:{pc}}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'SF Mono','Fira Code','Consolas',monospace;font-size:13px;line-height:1.6;padding:24px}}
.header{{display:flex;justify-content:space-between;align-items:flex-end;padding-bottom:16px;border-bottom:1px solid var(--border);margin-bottom:24px}}
.brand{{font-size:22px;font-weight:700;letter-spacing:.08em}}.brand span{{color:var(--pc)}}
.subtitle{{font-size:11px;color:var(--muted);letter-spacing:.12em;margin-top:2px}}
.header-right{{text-align:right;font-size:11px;color:var(--muted)}}
.asof{{font-size:14px;font-weight:600;color:var(--text)}}
.headline-banner{{background:var(--surface);border:1px solid var(--pc);border-radius:6px;padding:14px 20px;margin-bottom:20px;font-size:14px;font-weight:600;color:var(--pc);letter-spacing:.04em}}
.stat-row{{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap}}
.stat-card{{flex:1;min-width:120px;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:14px 16px}}
.stat-label{{font-size:10px;letter-spacing:.12em;color:var(--muted);margin-bottom:6px}}
.stat-value{{font-size:20px;font-weight:700}}
.notes-strip{{background:var(--surface);padding:10px 16px;border-radius:4px;color:var(--muted);font-size:12px;margin-bottom:20px}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;margin-bottom:20px}}
.panel{{background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden}}
.side-panel{{background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden}}
.panel-header{{padding:10px 16px;border-bottom:1px solid var(--border);font-size:10px;letter-spacing:.14em;color:var(--muted);font-weight:600}}
.flex-body{{padding:12px 16px;display:flex;flex-direction:column;gap:8px}}
.driver-item{{display:flex;align-items:flex-start;gap:10px;font-size:12px;color:var(--text)}}
.dot{{width:7px;height:7px;border-radius:50%;margin-top:5px;flex-shrink:0}}
.delta-item{{font-size:12px;color:var(--text);padding:3px 0}}
.delta-item.warn{{color:#f59e0b;font-weight:600}}
table{{width:100%;border-collapse:collapse}}
th{{font-size:10px;letter-spacing:.1em;color:var(--muted);text-align:left;padding:8px 16px;border-bottom:1px solid var(--border);font-weight:600}}
td{{padding:9px 16px;border-bottom:1px solid var(--border);font-size:12px}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(255,255,255,.02)}}
.muted{{color:var(--muted)}}
.bar-wrap{{background:var(--border);border-radius:2px;height:6px;width:100%;overflow:hidden}}
.bar{{height:100%;border-radius:2px;min-width:2px}}
.badge{{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10px;letter-spacing:.06em;font-weight:600}}
.takeaways-list{{list-style:none;padding:14px 16px}}
.takeaways-list li{{padding:6px 0 6px 18px;position:relative;font-size:12px;border-bottom:1px solid var(--border)}}
.takeaways-list li:last-child{{border-bottom:none}}
.takeaways-list li::before{{content:'›';position:absolute;left:0;color:var(--pc);font-weight:700}}
.footer{{margin-top:32px;padding-top:16px;border-top:1px solid var(--border);display:flex;justify-content:space-between;font-size:10px;color:var(--muted);letter-spacing:.06em}}
@media print{{body{{background:#fff;color:#000;padding:16px}}.panel,.stat-card{{background:#f8f8f8;border-color:#ddd}}.brand,.stat-value{{color:#000}}th,.stat-label,.panel-header,.muted{{color:#555}}td{{color:#111}}}}
</style>
</head>
<body>

<div class="header">
  <div>
    <div class="brand">ASCENT <span>INTEL</span></div>
    <div class="subtitle">MARKET INTELLIGENCE FOR 20IN20 PARTNERS</div>
  </div>
  <div class="header-right">
    <div class="asof">{asof}</div>
    <div>AS-OF DATE</div>
  </div>
</div>

<div class="headline-banner">{headline}</div>
<div class="stat-row">{regime_cards}</div>
{notes_html}
{meta_row}

<div class="grid-3">
  <div class="panel">
    <div class="panel-header">THEME LEADERSHIP (3M)</div>
    <table>
      <thead><tr><th>THEME</th><th>RETURN</th><th>MOMENTUM</th><th>N</th></tr></thead>
      <tbody>{themes_rows or '<tr><td colspan="4" class="muted">No data</td></tr>'}</tbody>
    </table>
  </div>
  <div class="panel">
    <div class="panel-header">RELATIVE VALUE — EXTENDED</div>
    <table>
      <thead><tr><th>TICKER</th><th>THEME</th><th>Z-SCORE</th></tr></thead>
      <tbody>{ext_rows}</tbody>
    </table>
  </div>
  <div class="panel">
    <div class="panel-header">RELATIVE VALUE — DEPRESSED</div>
    <table>
      <thead><tr><th>TICKER</th><th>THEME</th><th>Z-SCORE</th></tr></thead>
      <tbody>{dep_rows}</tbody>
    </table>
  </div>
</div>

{comps_panel}

<div class="grid-2" style="margin-bottom:20px">
  <div class="panel">
    <div class="panel-header">SCENARIO WATCH</div>
    <table>
      <thead><tr><th>SCENARIO</th><th>EST. P&amp;L</th><th>SEVERITY</th><th>MOST EXPOSED</th></tr></thead>
      <tbody>{sc_rows}</tbody>
    </table>
  </div>
  <div class="panel">
    <div class="panel-header">KEY TAKEAWAYS</div>
    <ul class="takeaways-list">{ta_html}</ul>
  </div>
</div>

<div class="footer">
  <span>ASCENT CAPITAL — SYSTEMATIC RESEARCH PLATFORM</span>
  <span>{disclaimer}</span>
</div>

</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof",        required=True)
    parser.add_argument("--outputs_dir", default="outputs/20in20")
    parser.add_argument("--open",        action="store_true")
    args = parser.parse_args()

    base     = Path(args.outputs_dir)
    html     = build_html(args.asof, base)
    out_path = base / f"intel_{args.asof}.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"[dashboard] Written → {out_path.resolve()}")

    if args.open:
        import subprocess
        subprocess.run(["open", str(out_path)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
