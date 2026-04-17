# Catalyst Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Before each rebalance debate, scan for upcoming earnings dates, ex-dividend dates, and FOMC meetings for all held positions and inject the findings into `portfolio_state` so debate agents can reason about near-term binary events.

**Architecture:** A new `ascent/reporting/catalyst_scanner.py` module fetches catalyst data using `yfinance` (already installed) for earnings/ex-div and a hardcoded 2026 FOMC calendar. `debate_runner.py` calls it after the scenario sim and before any agents run, injecting `"catalyst_context"` into `portfolio_state`. Each debate agent's `_build_context()` block already includes all `portfolio_state` keys, so no agent changes are needed — context propagates automatically.

**Tech Stack:** Python 3.12, yfinance (already installed), standard library datetime

---

## File Structure

- **Create:** `ascent/reporting/catalyst_scanner.py` — fetch and format catalyst data for a list of symbols
- **Modify:** `debate/debate_runner.py:131–153` — call `scan_catalysts()` after scenario sim, inject result into `portfolio_state`
- **Modify:** `debate/agents.py:21–33` — extend `_build_context()` to render `catalyst_context` if present
- **Create:** `tests/test_catalyst_scanner.py` — unit tests (mock yfinance, no network)

---

### Task 1: Build `catalyst_scanner.py` with earnings and ex-div detection

**Files:**
- Create: `ascent/reporting/catalyst_scanner.py`
- Test: `tests/test_catalyst_scanner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_catalyst_scanner.py
from unittest.mock import patch, MagicMock
from datetime import date
import pytest

from ascent.reporting.catalyst_scanner import (
    scan_catalysts,
    _days_until,
    FOMC_DATES_2026,
)


def test_days_until_future():
    d = date(2026, 4, 20)
    result = _days_until(d, as_of=date(2026, 4, 12))
    assert result == 8


def test_days_until_past():
    d = date(2026, 4, 10)
    result = _days_until(d, as_of=date(2026, 4, 12))
    assert result == -2


def test_days_until_today():
    d = date(2026, 4, 12)
    result = _days_until(d, as_of=date(2026, 4, 12))
    assert result == 0


def test_fomc_dates_present():
    # 2026 FOMC has at least 8 meetings
    assert len(FOMC_DATES_2026) >= 8
    assert all(isinstance(d, date) for d in FOMC_DATES_2026)


def test_scan_catalysts_returns_dict():
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": None, "ex_div_date": None}
        result = scan_catalysts(["AAPL", "MSFT"], as_of=date(2026, 4, 12), window_days=30)
    assert isinstance(result, dict)
    assert "upcoming_events" in result
    assert "catalyst_text" in result


def test_scan_catalysts_detects_earnings():
    earnings_date = date(2026, 4, 20)
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.side_effect = lambda sym: {
            "earnings_date": earnings_date if sym == "AAPL" else None,
            "ex_div_date": None,
        }
        result = scan_catalysts(["AAPL", "MSFT"], as_of=date(2026, 4, 12), window_days=30)
    assert any("AAPL" in ev["symbol"] and ev["type"] == "earnings" for ev in result["upcoming_events"])


def test_scan_catalysts_detects_ex_div():
    ex_div = date(2026, 4, 15)
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.side_effect = lambda sym: {
            "earnings_date": None,
            "ex_div_date": ex_div if sym == "MRK" else None,
        }
        result = scan_catalysts(["MRK", "WMT"], as_of=date(2026, 4, 12), window_days=30)
    assert any("MRK" in ev["symbol"] and ev["type"] == "ex_div" for ev in result["upcoming_events"])


def test_scan_catalysts_detects_fomc():
    # FOMC on 2026-05-06 — within 30 days of 2026-04-12
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": None, "ex_div_date": None}
        result = scan_catalysts(["AAPL"], as_of=date(2026, 4, 12), window_days=30)
    fomc_events = [ev for ev in result["upcoming_events"] if ev["type"] == "fomc"]
    assert len(fomc_events) >= 1


def test_scan_catalysts_filters_by_window():
    far_future = date(2026, 8, 1)
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": far_future, "ex_div_date": None}
        result = scan_catalysts(["AAPL"], as_of=date(2026, 4, 12), window_days=14)
    # 111 days away, outside the 14-day window
    assert not any(ev["type"] == "earnings" for ev in result["upcoming_events"])


def test_scan_catalysts_empty_symbols():
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.return_value = {"earnings_date": None, "ex_div_date": None}
        result = scan_catalysts([], as_of=date(2026, 4, 12))
    assert result["upcoming_events"] == []
    assert "no upcoming" in result["catalyst_text"].lower()


def test_yfinance_failure_does_not_crash():
    with patch("ascent.reporting.catalyst_scanner._fetch_ticker_catalysts") as mock_fetch:
        mock_fetch.side_effect = Exception("network error")
        # Should not raise — returns empty events
        result = scan_catalysts(["AAPL"], as_of=date(2026, 4, 12))
    assert result["upcoming_events"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_catalyst_scanner.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'ascent.reporting.catalyst_scanner'`

- [ ] **Step 3: Implement `catalyst_scanner.py`**

```python
# ascent/reporting/catalyst_scanner.py
"""
ascent/reporting/catalyst_scanner.py
Scans for upcoming binary events (earnings, ex-div, FOMC) for held positions.

Called by debate_runner.py before agents run.
Returns a structured dict injected into portfolio_state["catalyst_context"].
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

# 2026 FOMC meeting dates (announcement day, source: federalreserve.gov)
FOMC_DATES_2026 = [
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

        # Earnings date — from calendar
        cal = ticker.calendar
        if cal is not None and not cal.empty:
            # calendar is a DataFrame with dates as columns; first column is next earnings
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
        ex_div_ts = info.get("exDividendDate")  # unix timestamp or None
        if ex_div_ts:
            from datetime import datetime
            result["ex_div_date"] = datetime.utcfromtimestamp(ex_div_ts).date()
    except Exception:
        pass

    return result


def scan_catalysts(
    symbols: list[str],
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
    events: list[dict] = []

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

    catalyst_text = _format_catalyst_text(events, as_of=as_of)

    return {"upcoming_events": events, "catalyst_text": catalyst_text}


def _format_catalyst_text(events: list[dict], as_of: date) -> str:
    """Format catalyst events as a concise LLM-readable block."""
    if not events:
        return "No upcoming catalysts within the scan window."

    lines = [f"Upcoming catalysts (within {max(e['days_away'] for e in events)} days of {as_of}):"]
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
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_catalyst_scanner.py -v
```

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add ascent/reporting/catalyst_scanner.py tests/test_catalyst_scanner.py
git commit -m "feat: add catalyst_scanner — earnings, ex-div, FOMC detection"
```

---

### Task 2: Integrate catalyst scanner into the debate runner

**Files:**
- Modify: `debate/debate_runner.py:131–153` — add catalyst scan block after scenario sim
- Modify: `debate/agents.py:21–33` — extend `_build_context()` to include catalyst_context

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/test_catalyst_scanner.py

def test_debate_runner_injects_catalyst_context(monkeypatch):
    """debate_runner calls scan_catalysts and injects result into portfolio_state."""
    import importlib
    from unittest.mock import patch, MagicMock

    portfolio_state = {
        "date": "2026-04-12",
        "us_regime": "calm_bull",
        "macro_regime": "neutral",
        "n_positions": 2,
        "allocation": {},
        "weights": {"AAPL": 0.5, "MSFT": 0.5},
    }

    fake_catalyst = {
        "upcoming_events": [{"symbol": "AAPL", "type": "earnings", "date": "2026-04-20", "days_away": 8}],
        "catalyst_text": "AAPL Earnings: 2026-04-20 (8 days away)",
    }

    with patch("debate.debate_runner.scan_catalysts", return_value=fake_catalyst) as mock_scan, \
         patch("debate.debate_runner.score_pending_verdicts", return_value=0), \
         patch("debate.debate_runner.run_pending_debriefs", return_value=0), \
         patch("debate.debate_runner.detect_blind_spots"), \
         patch("debate.debate_runner.load_blind_spot_context", return_value=""), \
         patch("debate.debate_runner.run_all_scenarios", return_value=[]), \
         patch("debate.debate_runner.run_bull_agent", return_value="bull"), \
         patch("debate.debate_runner.run_bear_agent", return_value="bear"), \
         patch("debate.debate_runner.run_devils_advocate", return_value="devil"), \
         patch("debate.debate_runner.run_regime_specialist", return_value="regime"), \
         patch("debate.debate_runner.run_quant_sanity_check", return_value="quant"), \
         patch("debate.debate_runner.run_judge", return_value={
             "confidence": 0.8,
             "recommendation": "proceed",
             "key_risks": [],
             "reasoning": "ok",
         }):
        from debate.debate_runner import run_debate
        run_debate(portfolio_state=portfolio_state, run_date=date(2026, 4, 12))

    mock_scan.assert_called_once()
    assert "catalyst_context" in portfolio_state
    assert portfolio_state["catalyst_context"]["catalyst_text"] == fake_catalyst["catalyst_text"]
```

- [ ] **Step 2: Run failing test**

```bash
.venv/bin/pytest tests/test_catalyst_scanner.py::test_debate_runner_injects_catalyst_context -v
```

Expected: FAIL — `scan_catalysts` not imported in `debate_runner.py`

- [ ] **Step 3: Add catalyst scan block to `debate_runner.py`**

Add `from ascent.reporting.catalyst_scanner import scan_catalysts` at the top of `debate/debate_runner.py` with the other imports (line ~29).

Then insert this block **after** the scenario sim block (after line ~152, before the `# Bull agent` comment):

```python
    # Scan for upcoming catalysts (earnings, ex-div, FOMC)
    print("[Debate] Scanning for upcoming catalysts...")
    try:
        symbols = list(portfolio_state.get("weights", {}).keys())
        catalyst_result = scan_catalysts(symbols)
        portfolio_state["catalyst_context"] = catalyst_result
        if catalyst_result["upcoming_events"]:
            print(f"[Debate] Catalysts: {catalyst_result['catalyst_text'][:200]}")
        else:
            print("[Debate] No upcoming catalysts in scan window")
    except Exception as e:
        portfolio_state["catalyst_context"] = {"upcoming_events": [], "catalyst_text": ""}
        print(f"[Debate] Catalyst scan failed (non-fatal): {e}")
```

- [ ] **Step 4: Add catalyst rendering to `_build_context()` in `debate/agents.py`**

In `debate/agents.py`, extend `_build_context()` by adding this block after the weights loop (after line ~33, before `return "\n".join(lines)`):

```python
    catalyst_ctx = portfolio_state.get("catalyst_context", {})
    catalyst_text = catalyst_ctx.get("catalyst_text", "") if isinstance(catalyst_ctx, dict) else ""
    if catalyst_text and "no upcoming" not in catalyst_text.lower():
        lines.append("")
        lines.append("Upcoming catalysts:")
        lines.append(catalyst_text)
```

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/pytest tests/test_catalyst_scanner.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add debate/debate_runner.py debate/agents.py tests/test_catalyst_scanner.py
git commit -m "feat: wire catalyst scanner into debate runner — agents now see upcoming events"
```

---

## Self-Review

**Spec coverage:**
- Earnings detection: Task 1 (`_fetch_ticker_catalysts`, `scan_catalysts`) ✓
- Ex-dividend detection: Task 1 ✓
- FOMC detection: Task 1 (`FOMC_DATES_2026`) ✓
- Injection into `portfolio_state`: Task 2 (debate_runner) ✓
- Agents see catalyst context: Task 2 (`_build_context`) ✓
- Non-fatal failure handling: Task 1 (try/except in `scan_catalysts` + `_fetch_ticker_catalysts`) ✓

**Placeholder scan:** None found.

**Type consistency:**
- `scan_catalysts()` → `dict` with keys `upcoming_events` and `catalyst_text` — consistent in scanner, runner, and test.
- `_fetch_ticker_catalysts()` → `dict` with keys `earnings_date` and `ex_div_date` — used consistently in `scan_catalysts`.
