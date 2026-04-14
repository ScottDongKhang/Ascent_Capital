"""
ascent/reporting/catalyst_scanner.py
Scans for upcoming binary events (earnings, ex-div, FOMC) for held positions.

Called by debate_runner.py before agents run.
Returns a structured dict injected into portfolio_state["catalyst_context"].
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

log = logging.getLogger(__name__)

# 2026 FOMC meeting dates (announcement day, source: federalreserve.gov)
FOMC_DATES_2026: List[date] = [
    date(2026, 1, 29),
    date(2026, 3, 19),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


def _days_until(target: date, as_of: Optional[date] = None) -> int:
    """Days from as_of until target. Negative means target is in the past."""
    as_of = as_of or date.today()
    return (target - as_of).days


def _fetch_ticker_catalysts(symbol: str) -> dict:
    """
    Fetch earnings date and ex-dividend date for a symbol via yfinance.
    Returns {"earnings_date": date | None, "ex_div_date": date | None}.
    Never raises — returns None values on any failure.
    """
    import yfinance as yf

    result = {"earnings_date": None, "ex_div_date": None}

    try:
        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is not None and not cal.empty:
            earnings_ts = cal.columns[0]
            if hasattr(earnings_ts, "date"):
                result["earnings_date"] = earnings_ts.date()
            elif hasattr(earnings_ts, "year"):
                result["earnings_date"] = earnings_ts.to_pydatetime().date()
    except Exception:
        pass

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        ex_div_ts = info.get("exDividendDate")
        if ex_div_ts:
            result["ex_div_date"] = datetime.utcfromtimestamp(ex_div_ts).date()
    except Exception:
        pass

    return result


def scan_catalysts(
    symbols: List[str],
    as_of: Optional[date] = None,
    window_days: int = 21,
) -> dict:
    """
    Scan for upcoming catalysts for all symbols within window_days.

    Returns:
        {
          "upcoming_events": [
            {"symbol": str, "type": "earnings"|"ex_div"|"fomc", "date": date, "days_away": int},
            ...
          ],
          "catalyst_text": str,   # formatted summary for LLM prompt injection
        }
    Sorted by days_away ascending (soonest first).
    Never raises.
    """
    as_of = as_of or date.today()
    events: list = []

    # Per-symbol catalysts
    for sym in symbols:
        try:
            cats = _fetch_ticker_catalysts(sym)
        except Exception:
            cats = {"earnings_date": None, "ex_div_date": None}

        for event_type, dt in [("earnings", cats.get("earnings_date")),
                                ("ex_div", cats.get("ex_div_date"))]:
            if dt is None:
                continue
            days = _days_until(dt, as_of=as_of)
            if 0 <= days <= window_days:
                events.append({
                    "symbol": sym,
                    "type": event_type,
                    "date": dt,
                    "days_away": days,
                })

    # FOMC (not symbol-specific — affects all positions)
    for fomc_date in FOMC_DATES_2026:
        days = _days_until(fomc_date, as_of=as_of)
        if 0 <= days <= window_days:
            events.append({
                "symbol": "FOMC",
                "type": "fomc",
                "date": fomc_date,
                "days_away": days,
            })

    events.sort(key=lambda e: e["days_away"])

    catalyst_text = _format_catalyst_text(events, as_of=as_of, window_days=window_days)

    return {"upcoming_events": events, "catalyst_text": catalyst_text}


def _format_catalyst_text(events: list, as_of: date, window_days: int = 21) -> str:
    """Format catalyst events as a concise LLM-readable block."""
    if not events:
        return "No upcoming catalysts within the scan window."

    lines = [f"Upcoming catalysts (within {window_days} days of {as_of}):"]
    for ev in events:
        label = {
            "earnings": "Earnings",
            "ex_div":   "Ex-dividend",
            "fomc":     "FOMC meeting",
        }.get(ev["type"], ev["type"])

        sym_part = f"{ev['symbol']} " if ev["symbol"] != "FOMC" else ""
        lines.append(
            f"  {sym_part}{label}: {ev['date']} ({ev['days_away']} day(s) away)"
        )

    return "\n".join(lines)
