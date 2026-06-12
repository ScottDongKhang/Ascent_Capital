# TradingAgents Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-ticker AI PM outcome memory, instrument identity grounding, and StockTwits crowd sentiment to Ascent Capital.

**Architecture:** Three additive features: (1) `memory/ticker_memory.py` records every AI PM override per symbol, scores outcomes at 10d/21d via yfinance, and injects per-ticker history into the Phase 2 prompt; (2) `_build_data_grounding()` gains sector/industry identity from `profiles.parquet`; (3) `ascent/integrations/stocktwits.py` pre-fetches crowd sentiment and injects as a verified block into the pre-thesis prompt.

**Tech Stack:** Python 3.12, yfinance, pandas, existing `ascent.llm.client`, existing `run_all_agents.py` hooks.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `memory/ticker_memory.py` | Create | Record/score/retrieve per-ticker AI PM override history |
| `tests/memory/test_ticker_memory.py` | Create | Unit tests for ticker memory |
| `agents/ai_pm_agent.py` | Modify (2 places) | (1) Add sector/industry to `_build_data_grounding()`, (2) inject ticker context in `run_ai_pm()` |
| `run_all_agents.py` | Modify (2 places) | (1) Call `record_decision()` in `_write_decision_log()`, (2) call `score_outcomes()` after `compute_ai_feedback()` |
| `ascent/integrations/stocktwits.py` | Create | Fetch and parse StockTwits public sentiment |
| `tests/integrations/test_stocktwits.py` | Create | Unit tests for StockTwits integration |
| `run_all_agents.py` | Modify (1 more place) | Fetch StockTwits sentiment and inject into pre-thesis / Phase 2 |

---

## Task 1: `memory/ticker_memory.py` — Core Module

**Files:**
- Create: `memory/ticker_memory.py`
- Create: `tests/memory/__init__.py`
- Create: `tests/memory/test_ticker_memory.py`

### Step 1: Write failing tests

- [ ] Create `tests/memory/__init__.py` (empty)
- [ ] Create `tests/memory/test_ticker_memory.py`:

```python
import json
import pytest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock


def _make_ticker_memory(tmp_path):
    """Import ticker_memory with JSONL path redirected to tmp_path."""
    import importlib, sys
    # Temporarily patch the path constant after import
    import memory.ticker_memory as tm
    tm.TICKER_MEMORY_PATH = tmp_path / "ticker_memory.jsonl"
    return tm


def test_record_decision_appends_entry(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    tm.record_decision(
        symbol="CAT", date_str="2026-06-10", ai_w=0.10, quant_w=0.07,
        decision_type="amplify", rationale_snippet="Strong capex cycle momentum"
    )
    lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["symbol"] == "CAT"
    assert entry["ai_w"] == pytest.approx(0.10)
    assert entry["quant_w"] == pytest.approx(0.07)
    assert entry["scored"] is False


def test_record_decision_multiple_symbols(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    tm.record_decision("CAT", "2026-06-10", 0.10, 0.07, "amplify", "reason A")
    tm.record_decision("MRK", "2026-06-10", 0.06, 0.08, "reduce",  "reason B")
    lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_score_outcomes_skips_recent(tmp_path):
    """Entries less than 10 days old should not be scored."""
    tm = _make_ticker_memory(tmp_path)
    today_str = date.today().isoformat()
    tm.record_decision("CAT", today_str, 0.10, 0.07, "amplify", "recent")
    scored = tm.score_outcomes(date.today())
    assert scored == 0
    entry = json.loads((tmp_path / "ticker_memory.jsonl").read_text().splitlines()[0])
    assert entry["scored"] is False


def test_score_outcomes_scores_old_entry(tmp_path):
    """Entries 15 days old should be scored with mocked price data."""
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    tm.record_decision("CAT", old_date, 0.10, 0.07, "amplify", "old entry")

    with patch("memory.ticker_memory._fetch_return", side_effect=[0.05, 0.08]):
        scored = tm.score_outcomes(date.today())

    assert scored == 1
    entry = json.loads((tmp_path / "ticker_memory.jsonl").read_text().splitlines()[0])
    assert entry["scored"] is True
    assert entry["verdict"] in ("win", "miss", "fade", "early")
    # incremental_alpha = (ai_w - quant_w) * return = (0.10 - 0.07) * 0.05 = +0.0015
    assert entry["outcome_10d"] == pytest.approx((0.10 - 0.07) * 0.05, abs=1e-6)


def test_get_ticker_context_empty(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    ctx = tm.get_ticker_context("CAT")
    assert ctx == ""


def test_get_ticker_context_formats_history(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    tm.record_decision("CAT", old_date, 0.10, 0.07, "amplify", "capex thesis")
    # Manually mark as scored
    entry = json.loads((tmp_path / "ticker_memory.jsonl").read_text().splitlines()[0])
    entry.update(scored=True, outcome_10d=0.0015, outcome_21d=0.002, verdict="win")
    (tmp_path / "ticker_memory.jsonl").write_text(json.dumps(entry) + "\n")

    ctx = tm.get_ticker_context("CAT")
    assert "CAT" in ctx
    assert "amplify" in ctx
    assert "WIN" in ctx


def test_get_ticker_context_only_returns_that_symbol(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    tm.record_decision("CAT", old_date, 0.10, 0.07, "amplify", "reason")
    tm.record_decision("MRK", old_date, 0.06, 0.08, "reduce",  "reason")
    # score both
    for sym in ["CAT", "MRK"]:
        lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
        updated = []
        for line in lines:
            e = json.loads(line)
            if e["symbol"] == sym:
                e.update(scored=True, outcome_10d=0.001, outcome_21d=0.002, verdict="win")
            updated.append(json.dumps(e))
        (tmp_path / "ticker_memory.jsonl").write_text("\n".join(updated) + "\n")

    cat_ctx = tm.get_ticker_context("CAT")
    assert "CAT" in cat_ctx
    assert "MRK" not in cat_ctx


def test_get_cross_ticker_lessons_returns_recent(tmp_path):
    tm = _make_ticker_memory(tmp_path)
    old_date = (date.today() - timedelta(days=15)).isoformat()
    for sym in ["CAT", "MRK", "WMT"]:
        tm.record_decision(sym, old_date, 0.10, 0.07, "amplify", "reason")
        lines = (tmp_path / "ticker_memory.jsonl").read_text().splitlines()
        updated = []
        for line in lines:
            e = json.loads(line)
            if e["symbol"] == sym and not e["scored"]:
                e.update(scored=True, outcome_10d=0.001, outcome_21d=0.002, verdict="win")
            updated.append(json.dumps(e))
        (tmp_path / "ticker_memory.jsonl").write_text("\n".join(updated) + "\n")

    ctx = tm.get_cross_ticker_lessons(n=2)
    assert ctx != ""
    # Should include at most 2 entries
    assert ctx.count("amplify") <= 2
```

- [ ] Run tests to confirm all fail:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/memory/test_ticker_memory.py -v 2>&1 | tail -20
```

Expected: `ERROR` or `ModuleNotFoundError` — `memory.ticker_memory` does not exist yet.

---

### Step 2: Implement `memory/ticker_memory.py`

- [ ] Create `memory/ticker_memory.py`:

```python
"""
memory/ticker_memory.py

Per-ticker AI PM outcome memory.

Records every AI PM override per symbol with rationale snippet.
Scores outcomes at 10d/21d via yfinance (incremental alpha = (ai_w - quant_w) * return).
Injects per-ticker history into Phase 2 prompt when a symbol is being considered.

Zero LLM cost — pure Python + yfinance.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent
TICKER_MEMORY_PATH = _REPO / "memory" / "ticker_memory.jsonl"


# ── Write ──────────────────────────────────────────────────────────────────────

def record_decision(
    symbol: str,
    date_str: str,
    ai_w: float,
    quant_w: float,
    decision_type: str,
    rationale_snippet: str,
) -> None:
    """Append one override decision to ticker_memory.jsonl. Never raises."""
    try:
        entry = {
            "symbol":            symbol.upper(),
            "date":              date_str,
            "ai_w":              round(float(ai_w), 6),
            "quant_w":           round(float(quant_w), 6),
            "type":              decision_type,
            "rationale":         str(rationale_snippet)[:200],
            "scored":            False,
            "outcome_10d":       None,
            "outcome_21d":       None,
            "verdict":           None,
        }
        TICKER_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(TICKER_MEMORY_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.debug("[TickerMemory] record_decision %s failed: %s", symbol, exc)


# ── Score ──────────────────────────────────────────────────────────────────────

def _fetch_return(symbol: str, from_date: str, horizon: int) -> Optional[float]:
    """Fetch stock return from from_date + horizon trading days. Returns None on failure."""
    try:
        import yfinance as yf
        start = date.fromisoformat(from_date)
        end   = (start + timedelta(days=horizon + 15)).isoformat()
        df    = yf.download(symbol, start=from_date, end=end,
                            auto_adjust=True, progress=False)
        if df.empty or len(df) < 2:
            return None
        closes = df["Close"].squeeze().dropna()
        idx = min(horizon, len(closes) - 1)
        return float((closes.iloc[idx] - closes.iloc[0]) / closes.iloc[0])
    except Exception as exc:
        log.debug("[TickerMemory] _fetch_return %s: %s", symbol, exc)
        return None


def _classify_verdict(r10: Optional[float], r21: Optional[float]) -> Optional[str]:
    if r10 is None:
        return None
    if r10 > 0 and r21 is not None and r21 < 0:
        return "fade"
    if r10 < 0 and r21 is not None and r21 > 0:
        return "early"
    return "win" if r10 >= 0 else "miss"


def score_outcomes(today: date) -> int:
    """
    For each unscored entry 10+ days old: fetch returns, compute incremental alpha,
    classify verdict, rewrite the file. Returns count of newly scored entries.
    """
    if not TICKER_MEMORY_PATH.exists():
        return 0

    entries: List[dict] = []
    try:
        for line in TICKER_MEMORY_PATH.read_text().splitlines():
            if line.strip():
                entries.append(json.loads(line))
    except Exception as exc:
        log.warning("[TickerMemory] Could not read: %s", exc)
        return 0

    scored_count = 0
    for entry in entries:
        if entry.get("scored"):
            continue
        try:
            dec_date   = date.fromisoformat(entry["date"])
        except Exception:
            continue
        days_since = (today - dec_date).days
        if days_since < 10:
            continue

        sym   = entry["symbol"]
        ai_w  = entry["ai_w"]
        qw    = entry["quant_w"]
        r10 = r21 = None

        raw10 = _fetch_return(sym, entry["date"], 10)
        if raw10 is not None:
            r10 = round((ai_w - qw) * raw10, 6)

        if days_since >= 21:
            raw21 = _fetch_return(sym, entry["date"], 21)
            if raw21 is not None:
                r21 = round((ai_w - qw) * raw21, 6)

        entry["outcome_10d"] = r10
        entry["outcome_21d"] = r21
        entry["verdict"]     = _classify_verdict(r10, r21)
        entry["scored"]      = True
        scored_count        += 1
        log.info("[TickerMemory] Scored %s @ %s: 10d=%s 21d=%s → %s",
                 sym, entry["date"], r10, r21, entry["verdict"])

    if scored_count:
        TICKER_MEMORY_PATH.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n"
        )

    return scored_count


# ── Read / Inject ──────────────────────────────────────────────────────────────

def _load_entries() -> List[dict]:
    if not TICKER_MEMORY_PATH.exists():
        return []
    rows = []
    try:
        for line in TICKER_MEMORY_PATH.read_text().splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        pass
    return rows


def get_ticker_context(symbol: str, n: int = 3) -> str:
    """
    Return formatted string of last N AI PM decisions on this ticker with outcomes.
    Empty string if no history or all entries are unscored.
    """
    sym    = symbol.upper()
    rows   = [e for e in _load_entries() if e.get("symbol") == sym]
    scored = [e for e in rows if e.get("scored")]
    recent = sorted(scored, key=lambda e: e.get("date", ""), reverse=True)[:n]

    if not recent:
        return ""

    lines = [f"AI PM HISTORY — {sym} (last {len(recent)} call(s)):"]
    for e in reversed(recent):  # chronological order
        r10_str  = f"{e['outcome_10d']:+.3%}" if e.get("outcome_10d") is not None else "pending"
        r21_str  = f"{e['outcome_21d']:+.3%}" if e.get("outcome_21d") is not None else "pending"
        verdict  = (e.get("verdict") or "?").upper()
        lines.append(
            f"  {e['date']} {e['type']:8s} ai={e['ai_w']:.1%} vs q={e['quant_w']:.1%}"
            f" → 10d={r10_str} 21d={r21_str} [{verdict}]"
            f"\n    rationale: {e.get('rationale','')[:120]}"
        )
    return "\n".join(lines)


def get_cross_ticker_lessons(n: int = 3) -> str:
    """
    Return last N scored decisions (any ticker) as a cross-asset learning block.
    Empty string if no scored history.
    """
    scored = [e for e in _load_entries() if e.get("scored")]
    recent = sorted(scored, key=lambda e: e.get("date", ""), reverse=True)[:n]
    if not recent:
        return ""
    lines = [f"CROSS-TICKER AI PM LESSONS (last {len(recent)} scored calls):"]
    for e in recent:
        verdict = (e.get("verdict") or "?").upper()
        r10     = f"{e['outcome_10d']:+.3%}" if e.get("outcome_10d") is not None else "?"
        lines.append(
            f"  {e['date']} {e['symbol']:5s} {e['type']:8s}"
            f" 10d={r10} [{verdict}]: {e.get('rationale','')[:100]}"
        )
    return "\n".join(lines)
```

- [ ] Run tests to confirm they pass:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/memory/test_ticker_memory.py -v 2>&1 | tail -20
```

Expected: all 8 tests PASS.

- [ ] Commit:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add memory/ticker_memory.py tests/memory/__init__.py tests/memory/test_ticker_memory.py && git commit -m "feat: per-ticker AI PM outcome memory module"
```

---

## Task 2: Hook `record_decision` into `_write_decision_log()`

**Files:**
- Modify: `run_all_agents.py:137-155`

- [ ] Open `run_all_agents.py`. In `_write_decision_log()`, after the `with open(AI_PM_DECISION_LOG, "a") as f:` block (after line ~155), add ticker memory recording for each override.

Replace the closing lines of `_write_decision_log()` from:
```python
        with open(AI_PM_DECISION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Runner] AI PM decision logged (Level {entry['level']}, {len(overrides)} overrides)")
    except Exception as e:
        print(f"[Runner] Decision log skipped: {e}")
```

With:
```python
        with open(AI_PM_DECISION_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[Runner] AI PM decision logged (Level {entry['level']}, {len(overrides)} overrides)")

        # Record per-ticker decisions for ticker memory
        try:
            from memory.ticker_memory import record_decision as _record_ticker
            ai_portfolio = (ai_pm_result.portfolio
                            if ai_pm_result and not ai_pm_result.fallback else {})
            for ov in overrides:
                sym = ov.get("symbol", "")
                if not sym:
                    continue
                _record_ticker(
                    symbol=sym,
                    date_str=today.isoformat(),
                    ai_w=ai_portfolio.get(sym, quant_weights.get(sym, 0.0)),
                    quant_w=quant_weights.get(sym, 0.0),
                    decision_type=ov.get("ai_action", ov.get("override_type", "unknown")),
                    rationale_snippet=ov.get("reason", "")[:200],
                )
        except Exception as _tm_e:
            print(f"[Runner] Ticker memory record skipped: {_tm_e}")
    except Exception as e:
        print(f"[Runner] Decision log skipped: {e}")
```

- [ ] Verify syntax:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] Commit:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add run_all_agents.py && git commit -m "feat: record AI PM overrides to ticker memory on each rebalance"
```

---

## Task 3: Score outcomes daily in `run_all_agents.py`

**Files:**
- Modify: `run_all_agents.py` around line 1823

The daily feedback block ends at approximately:
```python
        except Exception as _fbe:
            print(f"[Runner] Feedback/authority update skipped: {_fbe}")
```

- [ ] Add `score_outcomes` call immediately after that `except` block:

```python
        except Exception as _fbe:
            print(f"[Runner] Feedback/authority update skipped: {_fbe}")

        # Score any ticker memory entries now old enough (10d+)
        try:
            from memory.ticker_memory import score_outcomes as _score_ticker
            _n_scored = _score_ticker(today)
            if _n_scored:
                print(f"[Runner] Ticker memory: scored {_n_scored} outcome(s)")
        except Exception as _ste:
            print(f"[Runner] Ticker memory scoring skipped: {_ste}")
```

- [ ] Verify syntax:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -c "import ast; ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

- [ ] Commit:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add run_all_agents.py && git commit -m "feat: score ticker memory outcomes in daily learning run"
```

---

## Task 4: Inject ticker context into AI PM Phase 2 prompt

**Files:**
- Modify: `agents/ai_pm_agent.py` around line 2031

The existing pattern is:
```python
    _p2_grounding = _build_data_grounding(_p2_symbols)
    _system = _p2_grounding + _system  # prepend grounding before all other context
```

- [ ] Replace those two lines with:

```python
    _p2_grounding = _build_data_grounding(_p2_symbols)

    # Per-ticker AI PM history: inject what worked / failed last time for each symbol
    _ticker_ctx = ""
    try:
        from memory.ticker_memory import get_ticker_context, get_cross_ticker_lessons
        ticker_blocks = [get_ticker_context(s) for s in _p2_symbols if get_ticker_context(s)]
        cross_lessons = get_cross_ticker_lessons(n=3)
        if ticker_blocks or cross_lessons:
            _ticker_ctx = "\n\n══ AI PM TICKER TRACK RECORD ══\n"
            if ticker_blocks:
                _ticker_ctx += "\n\n".join(ticker_blocks)
            if cross_lessons:
                _ticker_ctx += "\n\n" + cross_lessons
            _ticker_ctx += "\n══════════════════════════════\n"
    except Exception as _tc_exc:
        log.debug("[AIPMAgent] Ticker context failed: %s", _tc_exc)

    _system = _p2_grounding + _ticker_ctx + _system  # prepend grounding + history
```

- [ ] Verify syntax:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -c "import ast; ast.parse(open('agents/ai_pm_agent.py').read()); print('OK')"
```

Expected: `OK`

- [ ] Commit:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add agents/ai_pm_agent.py && git commit -m "feat: inject per-ticker AI PM history into Phase 2 system prompt"
```

---

## Task 5: Instrument identity in `_build_data_grounding()`

**Files:**
- Modify: `agents/ai_pm_agent.py:35-100` (`_build_data_grounding`)

The current function builds lines like:
```
  CAT: 21d=+4.2% | 63d=+11.3% | 252d=+22.1% | alpha_score=0.721
```

- [ ] In `_build_data_grounding()`, after loading `prices_path` and before the `for sym in symbols` loop, add a profiles lookup:

Add this block after the `alpha_scores` loading block (after line ~70):
```python
        # Load sector/industry identity from profiles.parquet
        _identity: dict = {}
        try:
            _prof_path = _REPO_ROOT / "data_cache" / "profiles.parquet"
            if _prof_path.exists():
                _prof = pd.read_parquet(_prof_path)
                if {"symbol", "sector", "industry"}.issubset(_prof.columns):
                    for _, row in _prof.iterrows():
                        s = str(row["symbol"])
                        sec = str(row.get("sector", "")) or ""
                        ind = str(row.get("industry", "")) or ""
                        if sec and sec != "Unknown":
                            _identity[s] = f"{sec} | {ind}" if ind and ind != "Unknown" else sec
        except Exception:
            pass
```

Then in the symbol loop, change the line that builds `parts` from:
```python
            parts = [f"{sym}:"]
```
To:
```python
            identity_tag = _identity.get(sym, "")
            parts = [f"{sym}:" + (f" [{identity_tag}]" if identity_tag else "")]
```

- [ ] Verify syntax:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -c "import ast; ast.parse(open('agents/ai_pm_agent.py').read()); print('OK')"
```

Expected: `OK`

- [ ] Smoke test the function directly:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -c "
from agents.ai_pm_agent import _build_data_grounding
result = _build_data_grounding(['CAT', 'MRK', 'WMT'])
print(result[:600])
"
```

Expected: lines containing `[Industrials | ...]` or similar identity tags alongside price data.

- [ ] Commit:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add agents/ai_pm_agent.py && git commit -m "feat: add sector/industry identity to AI PM data grounding"
```

---

## Task 6: `ascent/integrations/stocktwits.py` — Crowd Sentiment

**Files:**
- Create: `ascent/integrations/stocktwits.py`
- Create: `tests/integrations/__init__.py`
- Create: `tests/integrations/test_stocktwits.py`

### Step 1: Write failing tests

- [ ] Create `tests/integrations/__init__.py` (empty)
- [ ] Create `tests/integrations/test_stocktwits.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import json


MOCK_TWITS_RESPONSE = {
    "messages": [
        {"body": "CAT breaking out!", "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "selling CAT",       "entities": {"sentiment": {"basic": "Bearish"}}},
        {"body": "CAT hold",          "entities": {}},  # no sentiment tag
        {"body": "CAT looks good",    "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "CAT looks bad",     "entities": {"sentiment": {"basic": "Bearish"}}},
        {"body": "CAT looking ok",    "entities": {"sentiment": {"basic": "Bullish"}}},
    ]
}


def _mock_get(url, timeout):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = MOCK_TWITS_RESPONSE
    return resp


def test_get_sentiment_returns_dict():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    assert "CAT" in result
    assert isinstance(result["CAT"]["bullish"], int)
    assert isinstance(result["CAT"]["bearish"], int)


def test_get_sentiment_counts_labels():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    # 3 bullish, 2 bearish in mock data
    assert result["CAT"]["bullish"] == 3
    assert result["CAT"]["bearish"] == 2
    assert result["CAT"]["n_labeled"] == 5
    assert result["CAT"]["n_total"] == 6


def test_get_sentiment_computes_ratio():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    assert result["CAT"]["ratio"] == pytest.approx(3 / 5, abs=0.01)


def test_get_sentiment_band_bullish():
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_get):
        result = get_sentiment(["CAT"])
    assert result["CAT"]["band"] == "bullish"  # ratio 0.60 → bullish


def test_get_sentiment_stale_when_few_labels():
    """Less than 5 labeled messages → stale=True."""
    sparse = {"messages": [
        {"body": "x", "entities": {"sentiment": {"basic": "Bullish"}}},
        {"body": "y", "entities": {}},
    ]}
    def _mock_sparse(url, timeout):
        r = MagicMock(); r.status_code = 200; r.json.return_value = sparse; return r
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_sparse):
        result = get_sentiment(["CAT"])
    assert result["CAT"]["stale"] is True


def test_get_sentiment_handles_api_error():
    """Network error → returns stale entry, does not raise."""
    def _mock_err(url, timeout):
        raise ConnectionError("network down")
    from ascent.integrations.stocktwits import get_sentiment
    with patch("ascent.integrations.stocktwits.requests.get", side_effect=_mock_err):
        result = get_sentiment(["CAT"])
    assert "CAT" in result
    assert result["CAT"]["stale"] is True


def test_format_sentiment_block_shows_band():
    from ascent.integrations.stocktwits import format_sentiment_block
    data = {
        "CAT": {"bullish": 18, "bearish": 5, "n_labeled": 23, "n_total": 30,
                "ratio": 0.78, "band": "bullish", "stale": False},
        "MRK": {"bullish": 3,  "bearish": 14, "n_labeled": 17, "n_total": 30,
                "ratio": 0.18, "band": "strongly_bearish", "stale": False},
    }
    block = format_sentiment_block(data)
    assert "CAT" in block
    assert "bullish" in block.lower()
    assert "MRK" in block
    assert "strongly_bearish" in block.lower()


def test_format_sentiment_block_skips_stale():
    from ascent.integrations.stocktwits import format_sentiment_block
    data = {
        "CAT": {"bullish": 1, "bearish": 0, "n_labeled": 1, "n_total": 30,
                "ratio": 1.0, "band": "strongly_bullish", "stale": True},
    }
    block = format_sentiment_block(data)
    assert "CAT" not in block or "stale" in block.lower()
```

- [ ] Run to confirm failures:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/integrations/test_stocktwits.py -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError` or similar — module does not exist yet.

### Step 2: Implement `ascent/integrations/stocktwits.py`

- [ ] Create `ascent/integrations/stocktwits.py`:

```python
"""
ascent/integrations/stocktwits.py

StockTwits public sentiment — pre-fetched, zero hallucination.

Fetches user-labeled Bullish/Bearish tag counts from StockTwits public API
(no auth required). Returns structured sentiment per ticker.

Rate limit: ~200 req/hour free tier. With 15-symbol universe = 15 req/run.
Add 0.2s delay between requests to stay polite.
"""
from __future__ import annotations

import logging
import time
from typing import Dict

import requests

log = logging.getLogger(__name__)

_BASE_URL     = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
_TIMEOUT      = 5   # seconds
_DELAY        = 0.2  # seconds between requests
_MIN_LABELED  = 5   # below this → stale signal


def _fetch_symbol(symbol: str) -> dict:
    """Fetch last 30 messages for symbol. Returns raw API dict or {}."""
    url = _BASE_URL.format(symbol=symbol.upper())
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
        log.debug("[StockTwits] %s returned HTTP %d", symbol, resp.status_code)
    except Exception as exc:
        log.debug("[StockTwits] %s fetch failed: %s", symbol, exc)
    return {}


def _parse_messages(messages: list) -> tuple[int, int, int]:
    """Return (bullish_count, bearish_count, total_count)."""
    bullish = bearish = 0
    for msg in messages:
        sentiment = (msg.get("entities") or {}).get("sentiment") or {}
        basic = sentiment.get("basic", "")
        if basic == "Bullish":
            bullish += 1
        elif basic == "Bearish":
            bearish += 1
    return bullish, bearish, len(messages)


def _band(ratio: float) -> str:
    if ratio >= 0.75:  return "strongly_bullish"
    if ratio >= 0.55:  return "bullish"
    if ratio >= 0.45:  return "neutral"
    if ratio >= 0.25:  return "bearish"
    return "strongly_bearish"


def _empty_entry(stale: bool = True) -> dict:
    return {"bullish": 0, "bearish": 0, "n_labeled": 0, "n_total": 0,
            "ratio": 0.5, "band": "neutral", "stale": stale}


def get_sentiment(symbols: list[str], max_messages: int = 30) -> Dict[str, dict]:
    """
    Fetch Bullish/Bearish label counts from StockTwits for each symbol.

    Returns:
        {
          "CAT": {"bullish": 18, "bearish": 5, "n_labeled": 23, "n_total": 30,
                  "ratio": 0.78, "band": "bullish", "stale": False},
          ...
        }
    stale=True when n_labeled < 5 (low signal — do not rely on it).
    """
    results: Dict[str, dict] = {}
    for i, sym in enumerate(symbols):
        if i > 0:
            time.sleep(_DELAY)
        sym_upper = sym.upper()
        raw = _fetch_symbol(sym_upper)
        messages = raw.get("messages", [])
        bullish, bearish, total = _parse_messages(messages)
        n_labeled = bullish + bearish

        if n_labeled < _MIN_LABELED:
            results[sym_upper] = _empty_entry(stale=True)
            results[sym_upper].update(bullish=bullish, bearish=bearish,
                                      n_labeled=n_labeled, n_total=total)
            log.debug("[StockTwits] %s: stale (n_labeled=%d)", sym_upper, n_labeled)
            continue

        ratio = bullish / n_labeled
        results[sym_upper] = {
            "bullish":   bullish,
            "bearish":   bearish,
            "n_labeled": n_labeled,
            "n_total":   total,
            "ratio":     round(ratio, 3),
            "band":      _band(ratio),
            "stale":     False,
        }
        log.debug("[StockTwits] %s: %.0f%% bullish (%d/%d labeled)",
                  sym_upper, ratio * 100, bullish, n_labeled)

    return results


def format_sentiment_block(sentiment: Dict[str, dict]) -> str:
    """
    Format sentiment dict as a concise verified block for LLM prompt injection.
    Skips stale entries. Returns empty string if nothing to show.
    """
    lines = []
    for sym, data in sorted(sentiment.items()):
        if data.get("stale"):
            continue
        n    = data["n_labeled"]
        tot  = data["n_total"]
        band = data["band"]
        pct  = round(data["ratio"] * 100)
        lines.append(f"  {sym}: {pct}% bullish ({n} labeled / {tot} total) — {band}")

    if not lines:
        return ""

    return (
        "══ CROWD SENTIMENT (StockTwits, user-labeled) ══\n"
        + "\n".join(lines)
        + "\n══════════════════════════════════════════════\n"
    )
```

- [ ] Run tests:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/integrations/test_stocktwits.py -v 2>&1 | tail -20
```

Expected: all 8 tests PASS.

- [ ] Commit:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add ascent/integrations/stocktwits.py tests/integrations/__init__.py tests/integrations/test_stocktwits.py && git commit -m "feat: StockTwits grounded sentiment integration"
```

---

## Task 7: Wire StockTwits into pre-thesis and Phase 2 prompts

**Files:**
- Modify: `run_all_agents.py:1069` — add sentiment fetch before pre-thesis call
- Modify: `agents/ai_pm_agent.py:1783` — add param to `run_ai_pm_prethesis()`
- Modify: `agents/ai_pm_agent.py:1905` — add param to `run_ai_pm()`

### Step 1: Add `sentiment_block` parameter to `run_ai_pm_prethesis()` (line 1783)

- [ ] In `agents/ai_pm_agent.py`, change line 1783 from:

```python
def run_ai_pm_prethesis() -> Optional[AIPreThesis]:
```

To:

```python
def run_ai_pm_prethesis(sentiment_block: str = "") -> Optional[AIPreThesis]:
```

- [ ] Inside `run_ai_pm_prethesis()`, inject sentiment into the Phase 1 system prompt. The system prompt is assembled at line 1830:

```python
            system_prompt=_build_temporal_context(feedback=_p1_feedback) + _p1_grounding + _PRE_THESIS_PROMPT,
```

Change it to:

```python
            system_prompt=_build_temporal_context(feedback=_p1_feedback) + _p1_grounding + sentiment_block + _PRE_THESIS_PROMPT,
```

### Step 2: Add `sentiment_block` parameter to `run_ai_pm()` (line 1905)

- [ ] In `agents/ai_pm_agent.py`, add `sentiment_block: str = ""` to `run_ai_pm()`'s signature:

```python
def run_ai_pm(
    quant_outputs: Optional[list] = None,
    merged_weights: Optional[Dict[str, float]] = None,
    prethesis: Optional[AIPreThesis] = None,
    causal_track_record: Optional[dict] = None,
    model_override: Optional[str] = None,
    sentiment_block: str = "",
) -> AIPMResult:
```

- [ ] Inside `run_ai_pm()`, find the block added in Task 4:

```python
    _p2_grounding = _build_data_grounding(_p2_symbols)

    # Per-ticker AI PM history: inject what worked / failed last time for each symbol
    _ticker_ctx = ""
```

Add sentiment injection immediately after `_p2_grounding = _build_data_grounding(_p2_symbols)`:

```python
    _p2_grounding = _build_data_grounding(_p2_symbols)
    if sentiment_block:
        _p2_grounding = sentiment_block + _p2_grounding

    # Per-ticker AI PM history: inject what worked / failed last time for each symbol
    _ticker_ctx = ""
```

### Step 3: Fetch sentiment in `run_all_agents.py` and pass to both calls

- [ ] In `run_all_agents.py`, find line 1069 (just before `run_ai_pm_prethesis()` call). Add the sentiment fetch block before the `try:` that calls `run_ai_pm_prethesis()`:

```python
        # Fetch StockTwits crowd sentiment for the current universe (pre-fetch before pre-thesis)
        _sentiment_block = ""
        try:
            from ascent.integrations.stocktwits import get_sentiment, format_sentiment_block
            _universe_syms = list((merged_weights or {}).keys())[:20]
            _st_data = get_sentiment(_universe_syms)
            _sentiment_block = format_sentiment_block(_st_data)
            if _sentiment_block:
                _n_live = len([v for v in _st_data.values() if not v["stale"]])
                print(f"[Runner] StockTwits: {_n_live} non-stale signals fetched")
            _st_ic_path = Path("logs/stocktwits_ic.jsonl")
            _st_ic_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_st_ic_path, "a") as _icf:
                for _sym, _sd in _st_data.items():
                    if not _sd["stale"]:
                        _icf.write(json.dumps({
                            "date": today.isoformat(),
                            "symbol": _sym,
                            "sentiment_ratio": _sd["ratio"],
                            "band": _sd["band"],
                        }) + "\n")
        except Exception as _ste:
            print(f"[Runner] StockTwits fetch skipped: {_ste}")

        try:
            print("[Runner] AI PM Phase 1 — forming original thesis before quant runs...")
            _ai_prethesis = run_ai_pm_prethesis(sentiment_block=_sentiment_block)
```

Note: this replaces the existing line `_ai_prethesis = run_ai_pm_prethesis()` with `_ai_prethesis = run_ai_pm_prethesis(sentiment_block=_sentiment_block)`.

- [ ] Update the `run_ai_pm()` call at line 1196 to pass `sentiment_block`:

```python
            ai_pm_result = run_ai_pm(
                quant_outputs=agent_outputs,
                merged_weights=merged_weights,
                prethesis=_ai_prethesis,
                causal_track_record=_causal_track_record,
                sentiment_block=_sentiment_block,
            )
```

- [ ] Verify syntax on both files:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -c "import ast; ast.parse(open('agents/ai_pm_agent.py').read()); ast.parse(open('run_all_agents.py').read()); print('OK')"
```

Expected: `OK`

### Step 4: Run full test suite

- [ ] Run existing tests to confirm no regressions:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && .venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```

Expected: same pass count as before (~777), no new failures.

- [ ] Commit:

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1" && git add agents/ai_pm_agent.py run_all_agents.py && git commit -m "feat: wire StockTwits sentiment into pre-thesis and Phase 2 prompts"
```

---

## Self-Review Notes

**Spec coverage check:**
- ✅ Feature 1 (ticker memory): Tasks 1–4 cover `record_decision`, `score_outcomes`, `get_ticker_context`, `get_cross_ticker_lessons`, integration in `_write_decision_log()`, daily scoring, Phase 2 injection.
- ✅ Feature 2 (instrument identity): Task 5 adds sector/industry to `_build_data_grounding()`.
- ✅ Feature 3 (StockTwits): Tasks 6–7 implement `get_sentiment`, `format_sentiment_block`, IC logging, pre-thesis + Phase 2 injection.
- ✅ No Reddit (permanently disabled per project memory).
- ✅ No alpha sleeve until IC validated.

**Type consistency:**
- `record_decision(symbol, date_str, ai_w, quant_w, decision_type, rationale_snippet)` — consistent across Task 1 tests, Task 1 implementation, Task 2 call site.
- `get_sentiment(symbols) → Dict[str, dict]` — consistent across Task 6 tests and Task 7 usage.
- `format_sentiment_block(data: Dict[str, dict]) → str` — consistent across Task 6 tests and Task 7 injection.
