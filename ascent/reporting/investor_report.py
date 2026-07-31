"""
ascent/reporting/investor_report.py
Monthly PDF investor report generator.

Uses reportlab for PDF generation (no system-library dependencies).
Output: outputs/investor_reports/YYYY-MM.pdf

Sections:
  1. Cover page
  2. Performance summary (GIPS-style)
  3. Attribution (factor-explained vs idiosyncratic)
  4. Risk summary (VaR, factor exposures, IS decomposition)
  5. Trade summary (turnover, slippage, event trades)
  6. Strategy commentary (Haiku-generated)
  7. Risk disclosures
"""

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/investor_reports")


def _pnl_for_month(year: int, month: int) -> list:
    from calendar import monthrange
    start = f"{year:04d}-{month:02d}-01"
    _, last_day = monthrange(year, month)
    end = f"{year:04d}-{month:02d}-{last_day:02d}"

    rows = []
    path = Path("logs/us_equities_pnl.jsonl")
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                d = r.get("date", "")
                if start <= d <= end:
                    rows.append(r)
            except Exception:
                pass
    return rows


def _slippage_for_month(year: int, month: int) -> list:
    start = f"{year:04d}-{month:02d}-01"
    from calendar import monthrange
    _, last_day = monthrange(year, month)
    end = f"{year:04d}-{month:02d}-{last_day:02d}"
    rows = []
    path = Path("logs/slippage_log.jsonl")
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if start <= r.get("run_date", "") <= end:
                    rows.append(r)
            except Exception:
                pass
    return rows


def _compute_performance(pnl_rows: list) -> dict:
    """Compute time-weighted return and related stats."""
    import math
    if not pnl_rows:
        return {}

    returns = [r["daily_return"] for r in pnl_rows if r.get("daily_return") is not None]
    spy_rets = [r["spy_return"] for r in pnl_rows if r.get("spy_return") is not None]
    if not returns:
        return {}

    n = len(returns)
    twr = 1.0
    for r in returns:
        twr *= (1 + r)
    twr -= 1.0

    ann_return = (1 + twr) ** (252 / max(n, 1)) - 1 if n > 0 else 0.0
    vol = (sum((r - sum(returns)/n)**2 for r in returns) / max(n-1, 1)) ** 0.5 * math.sqrt(252)
    sharpe = ann_return / vol if vol > 0 else 0.0

    # Max drawdown
    cum = [1.0]
    for r in returns:
        cum.append(cum[-1] * (1 + r))
    peak = cum[0]
    max_dd = 0.0
    for c in cum:
        if c > peak:
            peak = c
        dd = (peak - c) / peak
        if dd > max_dd:
            max_dd = dd

    spy_ann = 0.0
    if spy_rets:
        spy_twr = 1.0
        for r in spy_rets:
            spy_twr *= (1 + r)
        spy_twr -= 1.0
        spy_ann = (1 + spy_twr) ** (252 / max(len(spy_rets), 1)) - 1

    nav_start = pnl_rows[0].get("nav", 100_000)
    nav_end   = pnl_rows[-1].get("nav", nav_start)

    return {
        "twr":          round(twr, 4),
        "ann_return":   round(ann_return, 4),
        "ann_vol":      round(vol, 4),
        "sharpe":       round(sharpe, 3),
        "max_drawdown": round(max_dd, 4),
        "spy_ann":      round(spy_ann, 4),
        "alpha":        round(ann_return - spy_ann, 4),
        "nav_start":    round(nav_start, 2),
        "nav_end":      round(nav_end, 2),
        "n_days":       n,
    }


def _generate_commentary(perf: dict, year: int, month: int) -> str:
    """Generate 2-3 paragraph strategy commentary via Haiku."""
    try:
        from ascent.llm.client import generate, HAIKU_MODEL
        from ascent.utils.freshness import fresh_regime_label
        regime_result = fresh_regime_label()
        regime_display = regime_result["label"]
        if regime_result["stale"]:
            age = regime_result["age_days"]
            age_str = f"{age}d old" if age is not None else "age unknown"
            regime_display = f"{regime_display} (regime_signal.json stale, {age_str}; source: {regime_result['source']})"
        regime_data = {"regime": regime_display}

        prompt = f"""Write a 3-paragraph monthly strategy commentary for an institutional investor report.

Month: {year}-{month:02d}
Performance: {perf.get('twr', 0):.1%} monthly return, {perf.get('ann_return', 0):.1%} annualized
vs SPY: {perf.get('alpha', 0):.1%} alpha
Sharpe: {perf.get('sharpe', 0):.2f}
Max Drawdown: {perf.get('max_drawdown', 0):.1%}
Regime: {regime_data.get('regime', 'unknown')}

Paragraph 1: Regime environment and market conditions during the month.
Paragraph 2: Portfolio positioning and key drivers of performance.
Paragraph 3: Forward-looking positioning and risk posture.

Write professionally, concisely, and quantitatively. Do not use filler phrases."""

        return generate(prompt, model=HAIKU_MODEL, max_tokens=600)
    except Exception as exc:
        log.warning("[InvestorReport] Commentary generation failed: %s", exc)
        return (
            f"The strategy returned {perf.get('twr', 0):.1%} for the month "
            f"with a Sharpe ratio of {perf.get('sharpe', 0):.2f}. "
            "Market regime commentary unavailable — LLM client not configured."
        )


def _risk_disclosures() -> str:
    """Load or generate risk disclosures."""
    disc_path = Path("docs/risk_disclosures.md")
    if disc_path.exists():
        return disc_path.read_text()
    try:
        from compliance.risk_disclosure import generate_risk_disclosures
        return generate_risk_disclosures({})
    except Exception:
        return (
            "RISK DISCLOSURES\n\n"
            "Past performance is not indicative of future results. "
            "Systematic strategies may fail without warning when market dynamics shift. "
            "This strategy does not use leverage. "
            "Liquidity risk: positions may be illiquid in a market stress event. "
            "Model risk: all signals are derived from historical data which may not represent future conditions. "
            "Concentration risk: the portfolio holds a limited number of positions."
        )


def generate_monthly_report(year: int, month: int, output_path: Optional[Path] = None) -> Path:
    """
    Generate a PDF investor report for the given month.
    Saves to outputs/investor_reports/YYYY-MM.pdf by default.
    Returns the output Path.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or OUTPUT_DIR / f"{year:04d}-{month:02d}.pdf"

    pnl_rows   = _pnl_for_month(year, month)
    slip_rows  = _slippage_for_month(year, month)
    perf       = _compute_performance(pnl_rows)
    commentary = _generate_commentary(perf, year, month)
    disclosures = _risk_disclosures()

    doc = SimpleDocTemplate(str(out), pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    gold  = colors.HexColor("#C9A84C")
    dark  = colors.HexColor("#0F0F1A")
    light = colors.HexColor("#E8E8F0")

    h1 = ParagraphStyle("H1", parent=styles["Heading1"], textColor=gold, fontSize=22, spaceAfter=6)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=gold, fontSize=14, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"], textColor=light, fontSize=10, leading=14)
    small = ParagraphStyle("Small", parent=styles["Normal"], textColor=colors.grey, fontSize=8, leading=11)

    story = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("ASCENT CAPITAL", h1))
    story.append(Paragraph("Monthly Investor Report", ParagraphStyle("Sub", parent=h2, textColor=light, fontSize=16)))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Reporting Period: {year:04d}-{month:02d}", body))
    if perf.get("nav_end"):
        story.append(Paragraph(f"Strategy NAV (month-end): ${perf['nav_end']:,.0f}", body))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(
        "FOR INTERNAL USE ONLY — NOT FOR DISTRIBUTION. "
        "This document is confidential and intended solely for the named recipient.",
        small,
    ))
    story.append(PageBreak())

    # ── Performance summary ───────────────────────────────────────────────────
    story.append(Paragraph("Performance Summary", h2))
    if perf:
        perf_data = [
            ["Metric", "This Month", "Annualized"],
            ["Return", f"{perf['twr']:+.2%}", f"{perf['ann_return']:+.2%}"],
            ["SPY Return (est.)", "—", f"{perf['spy_ann']:+.2%}"],
            ["Alpha vs SPY", "—", f"{perf['alpha']:+.2%}"],
            ["Volatility (ann.)", "—", f"{perf['ann_vol']:.2%}"],
            ["Sharpe Ratio", "—", f"{perf['sharpe']:.2f}"],
            ["Max Drawdown", f"{perf['max_drawdown']:.2%}", "—"],
            ["Trading Days", str(perf["n_days"]), "—"],
        ]
        tbl = Table(perf_data, colWidths=[3*inch, 2*inch, 2*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), gold),
            ("TEXTCOLOR",    (0, 0), (-1, 0), dark),
            ("TEXTCOLOR",    (0, 1), (-1, -1), light),
            ("BACKGROUND",   (0, 1), (-1, -1), colors.HexColor("#1a1a2e")),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#1a1a2e"), colors.HexColor("#14142a")]),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 10),
            ("GRID",         (0, 0), (-1, -1), 0.3, colors.HexColor("#2d2d44")),
            ("ALIGN",        (1, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No performance data available for this period.", body))
    story.append(Spacer(1, 0.3*inch))

    # ── Attribution ───────────────────────────────────────────────────────────
    story.append(Paragraph("Attribution", h2))
    att_rows = []
    att_path = Path("logs/attribution_log.jsonl")
    if att_path.exists():
        with open(att_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    month_str = f"{year:04d}-{month:02d}"
                    if r.get("date", "").startswith(month_str):
                        att_rows.append(r)
                except Exception:
                    pass
    if att_rows:
        factor_pnl   = sum(r.get("factor_pnl", 0) or 0 for r in att_rows)
        specific_pnl = sum(r.get("idiosyncratic_pnl", r.get("total_pnl", 0)) or 0 for r in att_rows)
        story.append(Paragraph(
            f"Factor-explained P&L: ${factor_pnl:+,.0f} | "
            f"Idiosyncratic P&L: ${specific_pnl:+,.0f}",
            body,
        ))
    else:
        story.append(Paragraph("Attribution data not yet available.", body))
    story.append(Spacer(1, 0.3*inch))

    # ── Risk summary ──────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Summary", h2))
    if slip_rows:
        mean_slip = sum(r.get("slippage_bps", 0) for r in slip_rows) / len(slip_rows)
        story.append(Paragraph(
            f"Trades executed: {len(slip_rows)} | "
            f"Mean slippage: {mean_slip:.1f} bps",
            body,
        ))
    intra = []
    intra_path = Path("logs/intraday_adjustments.jsonl")
    if intra_path.exists():
        with open(intra_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    month_str = f"{year:04d}-{month:02d}"
                    if r.get("timestamp", "").startswith(month_str):
                        intra.append(r)
                except Exception:
                    pass
    story.append(Paragraph(f"Intraday trigger activations: {len(intra)}", body))
    story.append(Spacer(1, 0.3*inch))

    # ── Trade summary ─────────────────────────────────────────────────────────
    story.append(Paragraph("Trade Summary", h2))
    eod_rows = []
    eod_path = Path("logs/eod_log.jsonl")
    if eod_path.exists():
        with open(eod_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    month_str = f"{year:04d}-{month:02d}"
                    if r.get("date", "").startswith(month_str):
                        eod_rows.append(r)
                except Exception:
                    pass
    story.append(Paragraph(f"Rebalances this month: {len(eod_rows)}", body))
    story.append(Spacer(1, 0.3*inch))

    # ── Strategy commentary ───────────────────────────────────────────────────
    story.append(Paragraph("Strategy Commentary", h2))
    for para in commentary.split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), body))
            story.append(Spacer(1, 0.1*inch))
    story.append(Spacer(1, 0.3*inch))

    # ── Disclosures ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", color=gold))
    story.append(Spacer(1, 0.15*inch))
    story.append(Paragraph("Risk Disclosures", h2))
    for para in disclosures.replace("RISK DISCLOSURES\n\n", "").split("\n\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), small))
            story.append(Spacer(1, 0.05*inch))

    doc.build(story)
    log.info("[InvestorReport] Report written to %s", out)
    return out


def schedule_monthly_report() -> None:
    """Generate report for the prior month. Called on first Sunday of each month."""
    today = date.today()
    if today.month == 1:
        year, month = today.year - 1, 12
    else:
        year, month = today.year, today.month - 1

    try:
        path = generate_monthly_report(year, month)
        log.info("[InvestorReport] Monthly report complete: %s", path)
    except Exception as exc:
        log.error("[InvestorReport] Failed to generate monthly report: %s", exc)
