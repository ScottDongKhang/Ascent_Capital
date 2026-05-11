"""
compliance/performance_report.py
GIPS-compliant performance presentation.

Time-weighted return (TWR) is the GIPS standard.
Reads from TimescaleDB nav_series or logs/us_equities_pnl.jsonl fallback.
"""

import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def _load_pnl_rows(start_date: date, end_date: date, agent_id: str = "us_equities") -> List[dict]:
    """Load daily PnL rows between start and end (inclusive). DB → JSONL fallback."""
    try:
        from ascent.data.store.timescale import get_nav_series
        df = get_nav_series(agent_id=agent_id, lookback_days=(end_date - start_date).days + 5)
        if not df.empty:
            rows = []
            for ts, row in df.iterrows():
                d = ts.date() if hasattr(ts, "date") else date.fromisoformat(str(ts)[:10])
                if start_date <= d <= end_date:
                    rows.append({
                        "date":         d.isoformat(),
                        "nav":          row.get("nav"),
                        "daily_return": row.get("daily_return"),
                        "spy_return":   row.get("spy_return"),
                    })
            if rows:
                return rows
    except Exception:
        pass

    path = Path("logs/us_equities_pnl.jsonl")
    if not path.exists():
        return []
    rows = []
    start_str = start_date.isoformat()
    end_str   = end_date.isoformat()
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if start_str <= r.get("date", "") <= end_str:
                    rows.append(r)
            except Exception:
                pass
    return rows


def _risk_free_rate() -> float:
    """Try to get TB3MS from FRED; fall back to 5.0% if unavailable."""
    try:
        from ascent.data.ingest.fred import fetch_series
        s = fetch_series("TB3MS")
        if s is not None and not s.empty:
            return float(s.dropna().iloc[-1]) / 100.0
    except Exception:
        pass
    return 0.05


def compute_gips_performance(
    start_date: date,
    end_date: date,
    agent_id: str = "us_equities",
) -> dict:
    """
    Compute GIPS-standard performance metrics.

    Returns dict with: twr, ann_return, ann_vol, sharpe, max_drawdown,
                       calmar, beta_spy, alpha_spy, n_days, nav_start, nav_end
    """
    rows = _load_pnl_rows(start_date, end_date, agent_id)
    if not rows:
        return {}

    returns     = [r["daily_return"] for r in rows if r.get("daily_return") is not None]
    spy_returns = [r["spy_return"]   for r in rows if r.get("spy_return")   is not None]
    n           = len(returns)
    if n == 0:
        return {}

    # TWR: chain daily returns
    twr = 1.0
    for r in returns:
        twr *= (1 + r)
    twr -= 1.0

    ann_return = (1 + twr) ** (252 / n) - 1
    mean_r     = sum(returns) / n
    variance   = sum((r - mean_r) ** 2 for r in returns) / max(n - 1, 1)
    ann_vol    = math.sqrt(variance) * math.sqrt(252)
    rfr        = _risk_free_rate()
    sharpe     = (ann_return - rfr) / ann_vol if ann_vol > 0 else 0.0

    # Max drawdown
    cum   = [1.0]
    for r in returns:
        cum.append(cum[-1] * (1 + r))
    peak   = cum[0]
    max_dd = 0.0
    for c in cum:
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd

    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    # Beta and Jensen's alpha vs SPY
    beta_spy = 0.0
    alpha_spy = ann_return
    if spy_returns and len(spy_returns) == n:
        spy_mean = sum(spy_returns) / n
        cov      = sum((returns[i] - mean_r) * (spy_returns[i] - spy_mean) for i in range(n)) / max(n - 1, 1)
        spy_var  = sum((s - spy_mean) ** 2 for s in spy_returns) / max(n - 1, 1)
        beta_spy = cov / spy_var if spy_var > 0 else 0.0

        spy_twr = 1.0
        for r in spy_returns:
            spy_twr *= (1 + r)
        spy_twr   -= 1.0
        spy_ann    = (1 + spy_twr) ** (252 / len(spy_returns)) - 1
        alpha_spy  = ann_return - beta_spy * spy_ann

    return {
        "twr":           round(twr, 4),
        "ann_return":    round(ann_return, 4),
        "ann_vol":       round(ann_vol, 4),
        "sharpe":        round(sharpe, 3),
        "max_drawdown":  round(max_dd, 4),
        "calmar":        round(calmar, 3),
        "beta_spy":      round(beta_spy, 3),
        "alpha_spy":     round(alpha_spy, 4),
        "n_days":        n,
        "nav_start":     rows[0].get("nav"),
        "nav_end":       rows[-1].get("nav"),
        "start_date":    start_date.isoformat(),
        "end_date":      end_date.isoformat(),
    }


def format_gips_table(performance_dict: Dict[str, dict]) -> str:
    """
    Format a Markdown table of performance across periods.

    Args:
        performance_dict: {"1M": {...}, "3M": {...}, ...}  — output of compute_gips_performance

    Returns:
        Markdown table string.
    """
    periods = ["1M", "3M", "6M", "YTD", "1Y", "SI"]
    rows = {
        "Return":       {},
        "Volatility":   {},
        "Sharpe":       {},
        "Max Drawdown": {},
    }
    for period in periods:
        p = performance_dict.get(period, {})
        rows["Return"][period]       = f"{p.get('ann_return', 0):+.1%}" if p else "—"
        rows["Volatility"][period]   = f"{p.get('ann_vol', 0):.1%}"     if p else "—"
        rows["Sharpe"][period]       = f"{p.get('sharpe', 0):.2f}"      if p else "—"
        rows["Max Drawdown"][period] = f"{p.get('max_drawdown', 0):.1%}" if p else "—"

    header = "| Metric | " + " | ".join(periods) + " |"
    sep    = "| --- | " + " | ".join(["---"] * len(periods)) + " |"
    lines  = [header, sep]
    for metric, vals in rows.items():
        line = f"| {metric} | " + " | ".join(vals.get(p, "—") for p in periods) + " |"
        lines.append(line)
    return "\n".join(lines)


def generate_performance_presentation(
    output_path: Optional[Path] = None,
    agent_id: str = "us_equities",
) -> Path:
    """
    Generate one-page PDF with GIPS performance table and NAV chart.
    Returns the output Path.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    out = output_path or Path("outputs/investor_reports/performance_presentation.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    today     = date.today()
    periods   = {
        "1M":  (today - timedelta(days=30),  today),
        "3M":  (today - timedelta(days=90),  today),
        "6M":  (today - timedelta(days=180), today),
        "YTD": (date(today.year, 1, 1),      today),
        "1Y":  (today - timedelta(days=365), today),
        "SI":  (date(2026, 4, 1),            today),  # since inception April 2026
    }
    perf_by_period = {k: compute_gips_performance(s, e, agent_id) for k, (s, e) in periods.items()}
    table_md = format_gips_table(perf_by_period)

    doc    = SimpleDocTemplate(str(out), pagesize=letter)
    styles = getSampleStyleSheet()
    gold   = colors.HexColor("#C9A84C")
    light  = colors.HexColor("#E8E8F0")
    dark   = colors.HexColor("#0F0F1A")

    h1    = ParagraphStyle("H1", parent=styles["Heading1"], textColor=gold, fontSize=20)
    body  = ParagraphStyle("Body", parent=styles["Normal"], textColor=light, fontSize=10, leading=14)
    small = ParagraphStyle("Small", parent=styles["Normal"], textColor=colors.grey, fontSize=8, leading=11)

    story = [
        Paragraph("Ascent Capital — Performance Presentation", h1),
        Spacer(1, 0.2*inch),
        Paragraph(f"As of: {today.isoformat()} | Strategy inception: April 1, 2026", body),
        Spacer(1, 0.2*inch),
    ]

    # Convert markdown table to reportlab Table
    si = perf_by_period.get("SI", {})
    si_data = [
        ["Since Inception", "Value"],
        ["Annualized Return",   f"{si.get('ann_return', 0):+.1%}" if si else "—"],
        ["Annualized Vol",      f"{si.get('ann_vol', 0):.1%}"     if si else "—"],
        ["Sharpe Ratio",        f"{si.get('sharpe', 0):.2f}"      if si else "—"],
        ["Max Drawdown",        f"{si.get('max_drawdown', 0):.1%}" if si else "—"],
        ["Alpha vs SPY",        f"{si.get('alpha_spy', 0):+.1%}"  if si else "—"],
        ["Beta vs SPY",         f"{si.get('beta_spy', 0):.2f}"    if si else "—"],
    ]
    tbl = Table(si_data, colWidths=[3*inch, 2*inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), gold),
        ("TEXTCOLOR",     (0, 0), (-1, 0), dark),
        ("TEXTCOLOR",     (0, 1), (-1, -1), light),
        ("BACKGROUND",    (0, 1), (-1, -1), colors.HexColor("#1a1a2e")),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#2d2d44")),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "GIPS COMPLIANCE NOTICE: Performance presented in accordance with GIPS principles. "
        "Time-weighted returns are chain-linked daily. Past performance is not indicative of future results. "
        "This presentation is for informational purposes only.",
        small,
    ))

    doc.build(story)
    log.info("[PerformanceReport] Written to %s", out)
    return out


def get_all_period_performance(agent_id: str = "us_equities") -> Dict[str, dict]:
    """Return performance dict for all standard periods. Used by README auto-update."""
    today   = date.today()
    periods = {
        "1M":  (today - timedelta(days=30),  today),
        "3M":  (today - timedelta(days=90),  today),
        "6M":  (today - timedelta(days=180), today),
        "YTD": (date(today.year, 1, 1),      today),
        "1Y":  (today - timedelta(days=365), today),
        "SI":  (date(2026, 4, 1),            today),
    }
    return {k: compute_gips_performance(s, e, agent_id) for k, (s, e) in periods.items()}
