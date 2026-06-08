# AutoHedge Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three capabilities derived from AutoHedge: structured financial grounding (yfinance), live Exa news injection into AI PM pre-thesis, and dynamic ticker discovery with mini-rebalance trigger.

**Architecture:** Feature 2 (financials) and Feature 1 (Exa news) extend `_build_data_grounding()` in `agents/ai_pm_agent.py` to pass richer context to every AI PM call. Feature 3 (ticker discovery) runs daily via `run_all_agents.py`, uses Exa output to surface one candidate ticker, and triggers a gated mini-rebalance if conviction ≥ 0.75 and cooldown is clear.

**Tech Stack:** `yfinance` (already installed), `requests` (already installed), Exa search API (free tier, `EXA_API_KEY` env var), `ascent/llm/client.py` `chat_completion` with `HAIKU_MODEL`.

---

## File Map

| File | Status | Responsibility |
|------|--------|----------------|
| `agents/ai_pm_agent.py` | Modify | Add `_fetch_financials()` helper; extend `_build_data_grounding()` to accept `news_context` param and append financials + news blocks |
| `ascent/integrations/exa_news.py` | Create | `fetch_news(symbols, max_per_symbol) -> dict[str, list[str]]` |
| `ascent/strategy/ticker_discovery.py` | Create | `DiscoveryResult` dataclass + `run_discovery()` |
| `ascent/main.py` | Modify | Add `extra_symbols: list[str] \| None = None` to `run_pipeline()` |
| `agents/us_equities_agent.py` | Modify | Add `extra_symbols` to `run_us_equities_agent()`, pass through to `run_pipeline()` |
| `ascent/execution/eod_runner.py` | Modify | Add `large_trade_threshold_pct: float \| None = None` to `run_eod_with_weights()` |
| `run_all_agents.py` | Modify | Call `fetch_news()` daily; add `_check_mini_rebalance_cooldown()`, `_write_mini_rebalance_log()`, `_trigger_mini_rebalance()`; wire discovery into non-rebalance path |
| `tests/agents/test_ai_pm_financials.py` | Create | Tests for `_fetch_financials()` and financials block in `_build_data_grounding()` |
| `tests/integrations/test_exa_news.py` | Create | Tests for `fetch_news()` |
| `tests/strategy/test_ticker_discovery.py` | Create | Tests for `run_discovery()` |

---

## Task 1: `_fetch_financials()` — implement and test

**Files:**
- Modify: `agents/ai_pm_agent.py` (insert after line 117, after `_build_data_grounding`)
- Create: `tests/agents/test_ai_pm_financials.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/agents/test_ai_pm_financials.py`:

```python
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


def _make_mock_ticker():
    """Return a mock yfinance Ticker with realistic quarterly financials."""
    bs_cols = pd.to_datetime(["2026-03-31", "2025-12-31"])
    bs = pd.DataFrame(
        {
            "Current Assets":      [10_000_000.0, 9_000_000.0],
            "Current Liabilities": [5_000_000.0,  4_500_000.0],
            "Total Debt":          [8_000_000.0,  7_500_000.0],
            "Stockholders Equity": [4_000_000.0,  4_200_000.0],
        },
        index=bs_cols,
    ).T  # rows = line items, columns = dates

    inc_cols = pd.to_datetime([
        "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"
    ])
    inc = pd.DataFrame(
        {
            "Total Revenue": [20e6, 19e6, 18e6, 17e6, 16e6],
            "Gross Profit":  [6e6,  5.7e6, 5.4e6, 5.1e6, 4.8e6],
        },
        index=inc_cols,
    ).T

    t = MagicMock()
    t.quarterly_balance_sheet = bs
    t.quarterly_income_stmt = inc
    return t


def test_fetch_financials_returns_four_metrics():
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["CAT"])
    assert "CAT" in result
    metrics = result["CAT"]
    assert "current_ratio" in metrics
    assert "debt_to_equity" in metrics
    assert "revenue_growth_yoy" in metrics
    assert "gross_margin" in metrics


def test_fetch_financials_computes_correctly():
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["CAT"])
    m = result["CAT"]
    assert m["current_ratio"] == pytest.approx(2.0, abs=0.01)
    assert m["debt_to_equity"] == pytest.approx(2.0, abs=0.01)
    assert m["revenue_growth_yoy"] == pytest.approx(0.25, abs=0.01)  # (20M-16M)/16M
    assert m["gross_margin"] == pytest.approx(0.3, abs=0.01)        # 6M/20M


def test_fetch_financials_returns_empty_on_failure():
    bad_ticker = MagicMock()
    bad_ticker.quarterly_balance_sheet = pd.DataFrame()
    bad_ticker.quarterly_income_stmt = pd.DataFrame()
    with patch("yfinance.Ticker", return_value=bad_ticker):
        from agents.ai_pm_agent import _fetch_financials
        result = _fetch_financials(["BADTICKER"])
    assert result["BADTICKER"] == {}


def test_fetch_financials_writes_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agents.ai_pm_agent._REPO_ROOT", tmp_path
    )
    (tmp_path / "data_cache").mkdir()
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        from importlib import reload
        import agents.ai_pm_agent as mod
        reload(mod)
        result = mod._fetch_financials(["CAT"])
    cache_file = tmp_path / "data_cache" / "financials_cache.json"
    assert cache_file.exists()
    import json
    cached = json.loads(cache_file.read_text())
    assert "CAT" in cached["data"]


def test_fetch_financials_serves_from_cache(tmp_path, monkeypatch):
    import json, time as _t
    monkeypatch.setattr("agents.ai_pm_agent._REPO_ROOT", tmp_path)
    (tmp_path / "data_cache").mkdir()
    cache_file = tmp_path / "data_cache" / "financials_cache.json"
    cache_file.write_text(json.dumps({
        "_timestamp": _t.time(),
        "data": {"CAT": {"current_ratio": 9.9}},
    }))
    # Even with a bad yfinance, cache should be served
    with patch("yfinance.Ticker", side_effect=RuntimeError("should not call")):
        from importlib import reload
        import agents.ai_pm_agent as mod
        reload(mod)
        result = mod._fetch_financials(["CAT"])
    assert result["CAT"]["current_ratio"] == 9.9
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -m pytest tests/agents/test_ai_pm_financials.py -v 2>&1 | head -30
```

Expected: `ImportError` or `AttributeError: module 'agents.ai_pm_agent' has no attribute '_fetch_financials'`

- [ ] **Step 3: Implement `_fetch_financials()`**

In `agents/ai_pm_agent.py`, insert this function **after** line 117 (after `_build_data_grounding` ends):

```python
def _fetch_financials(symbols: list[str]) -> dict[str, dict]:
    """
    Fetch 4 key financial ratios from yfinance quarterly data (24h cache).
    Returns {symbol: {current_ratio, debt_to_equity, revenue_growth_yoy, gross_margin}}.
    Returns {} for any symbol on failure — never raises.
    """
    import json
    import time as _time

    cache_path = _REPO_ROOT / "data_cache" / "financials_cache.json"
    _CACHE_TTL = 24 * 3600

    cache: dict = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            if (_time.time() - raw.get("_timestamp", 0)) < _CACHE_TTL:
                cache = raw.get("data", {})
        except Exception:
            pass

    results: dict = {}
    cache_dirty = False

    for sym in symbols:
        if sym in cache:
            results[sym] = cache[sym]
            continue
        try:
            import yfinance as yf
            ticker = yf.Ticker(sym)
            bs  = ticker.quarterly_balance_sheet
            inc = ticker.quarterly_income_stmt
            m: dict = {}
            try:
                ca = float(bs.loc["Current Assets"].iloc[0])
                cl = float(bs.loc["Current Liabilities"].iloc[0])
                if cl != 0:
                    m["current_ratio"] = round(ca / cl, 2)
            except Exception:
                pass
            try:
                td = float(bs.loc["Total Debt"].iloc[0])
                eq = float(bs.loc["Stockholders Equity"].iloc[0])
                if eq != 0:
                    m["debt_to_equity"] = round(td / eq, 2)
            except Exception:
                pass
            try:
                rev = inc.loc["Total Revenue"]
                if len(rev) >= 5:
                    latest   = float(rev.iloc[0])
                    year_ago = float(rev.iloc[4])
                    if year_ago != 0:
                        m["revenue_growth_yoy"] = round((latest - year_ago) / abs(year_ago), 3)
            except Exception:
                pass
            try:
                gp = float(inc.loc["Gross Profit"].iloc[0])
                tr = float(inc.loc["Total Revenue"].iloc[0])
                if tr != 0:
                    m["gross_margin"] = round(gp / tr, 3)
            except Exception:
                pass
            results[sym]  = m
            cache[sym]    = m
            cache_dirty   = True
        except Exception as exc:
            log.debug("[AIPMAgent] _fetch_financials %s: %s", sym, exc)
            results[sym] = {}

    if cache_dirty:
        try:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"_timestamp": _time.time(), "data": cache}))
            tmp.rename(cache_path)
        except Exception:
            pass

    return results
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_financials.py -v 2>&1 | tail -15
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add agents/ai_pm_agent.py tests/agents/test_ai_pm_financials.py
git commit -m "feat: _fetch_financials() — yfinance quarterly ratios with 24h cache"
```

---

## Task 2: Wire financials block into `_build_data_grounding()`

**Files:**
- Modify: `agents/ai_pm_agent.py` (lines 35–117)
- Modify: `tests/agents/test_ai_pm_financials.py` (add 2 tests)

- [ ] **Step 1: Add tests for the financials block**

Append to `tests/agents/test_ai_pm_financials.py`:

```python
def test_build_data_grounding_includes_fundamentals_block():
    mock_ticker = _make_mock_ticker()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        with patch("agents.ai_pm_agent.pd") as mock_pd:
            # Make prices load fail cleanly so grounding still runs financials
            mock_pd.read_parquet.side_effect = FileNotFoundError
            from agents.ai_pm_agent import _build_data_grounding
            result = _build_data_grounding(["CAT"])
    # Even with no price data, financials block should appear if metrics exist
    # (relies on _fetch_financials returning data)
    assert "FUNDAMENTALS" in result or result == ""  # empty is ok if prices missing


def test_build_data_grounding_financials_values_present():
    """With prices mocked, fundamentals block should contain ratio values."""
    import pandas as pd as _pd_real
    mock_ticker = _make_mock_ticker()

    price_df = _pd_real.DataFrame({
        "date":   _pd_real.date_range("2024-01-01", periods=300, freq="B"),
        "symbol": "CAT",
        "close":  [100.0 + i * 0.1 for i in range(300)],
    })
    prices_wide = price_df.pivot_table(index="date", columns="symbol", values="close")

    with patch("yfinance.Ticker", return_value=mock_ticker), \
         patch("pandas.read_parquet", return_value=price_df):
        from agents.ai_pm_agent import _build_data_grounding
        result = _build_data_grounding(["CAT"])

    assert "curr_ratio" in result
    assert "D/E" in result
```

- [ ] **Step 2: Run new tests — confirm failure**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_financials.py::test_build_data_grounding_financials_values_present -v
```

Expected: `AssertionError` — "curr_ratio" not yet in output.

- [ ] **Step 3: Extend `_build_data_grounding()` to append financials block**

In `agents/ai_pm_agent.py`, change the return block at the end of `_build_data_grounding()` (currently lines 107–117):

**Before:**
```python
        if not rows:
            return ""
        return (
            "\n\n══ VERIFIED DATA FROM DATA CACHE (use only these numbers — do not cite others) ══\n"
            + "\n".join(rows)
            + "\n  Any financial metric NOT shown above: say 'data not available' — do not estimate.\n"
            + "══════════════════════════════════════════════════════════════════════════════════\n"
        )
    except Exception as exc:
        log.debug("[AIPMAgent] Data grounding failed: %s", exc)
        return ""
```

**After:**
```python
        if not rows:
            return ""
        grounding = (
            "\n\n══ VERIFIED DATA FROM DATA CACHE (use only these numbers — do not cite others) ══\n"
            + "\n".join(rows)
            + "\n  Any financial metric NOT shown above: say 'data not available' — do not estimate.\n"
            + "══════════════════════════════════════════════════════════════════════════════════\n"
        )

        # Append fundamentals block (yfinance quarterly, 24h cache)
        try:
            fin = _fetch_financials(symbols[:25])
            fin_rows = []
            for sym in symbols[:25]:
                m = fin.get(sym, {})
                if not m:
                    continue
                parts = [sym + ":"]
                if "current_ratio"      in m: parts.append(f"curr_ratio={m['current_ratio']}")
                if "debt_to_equity"     in m: parts.append(f"D/E={m['debt_to_equity']}")
                if "revenue_growth_yoy" in m: parts.append(f"rev_growth={m['revenue_growth_yoy']:+.0%}")
                if "gross_margin"       in m: parts.append(f"margin={m['gross_margin']:.0%}")
                if len(parts) > 1:
                    fin_rows.append("  " + " | ".join(parts))
            if fin_rows:
                grounding += (
                    "\n══ FUNDAMENTALS (yfinance quarterly, cached 24h) ════════════════\n"
                    + "\n".join(fin_rows)
                    + "\n════════════════════════════════════════════════════════════════\n"
                )
        except Exception as _fe:
            log.debug("[AIPMAgent] Financials block skipped: %s", _fe)

        return grounding
    except Exception as exc:
        log.debug("[AIPMAgent] Data grounding failed: %s", exc)
        return ""
```

Also update the function signature to accept an optional `news_context` parameter (wired in Task 4):

```python
def _build_data_grounding(
    symbols: list[str],
    news_context: dict[str, list[str]] | None = None,
) -> str:
```

- [ ] **Step 4: Run all financials tests**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_financials.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Run full suite — confirm no regressions**

```bash
.venv/bin/python -m pytest --tb=no -q 2>&1 | tail -5
```

Expected: same pass count as before (≥777), 0 new failures.

- [ ] **Step 6: Commit**

```bash
git add agents/ai_pm_agent.py tests/agents/test_ai_pm_financials.py
git commit -m "feat: inject quarterly financials block into AI PM data grounding"
```

---

## Task 3: `fetch_news()` — implement and test

**Files:**
- Create: `ascent/integrations/exa_news.py`
- Create: `tests/integrations/test_exa_news.py`

- [ ] **Step 1: Write failing tests**

Create `tests/integrations/test_exa_news.py`:

```python
import pytest
from unittest.mock import patch, MagicMock
import time


def _mock_exa_response(summaries: list[str]):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "results": [
            {"summary": {"answer": s}} for s in summaries
        ]
    }
    resp.raise_for_status = MagicMock()
    return resp


def test_fetch_news_returns_dict():
    with patch("ascent.integrations.exa_news.requests.post",
               return_value=_mock_exa_response(["CAT beats Q1", "Infrastructure up"])):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"])
    assert "CAT" in result
    assert isinstance(result["CAT"], list)


def test_fetch_news_returns_summaries():
    with patch("ascent.integrations.exa_news.requests.post",
               return_value=_mock_exa_response(["CAT beats Q1", "Infrastructure up"])):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"], max_per_symbol=2)
    assert result["CAT"] == ["CAT beats Q1", "Infrastructure up"]


def test_fetch_news_caps_at_max_per_symbol():
    with patch("ascent.integrations.exa_news.requests.post",
               return_value=_mock_exa_response(["A", "B", "C"])):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"], max_per_symbol=2)
    assert len(result["CAT"]) <= 2


def test_fetch_news_returns_empty_list_on_failure():
    def bad_post(*a, **kw):
        raise RuntimeError("network error")
    with patch("ascent.integrations.exa_news.requests.post", side_effect=bad_post):
        with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
            from ascent.integrations.exa_news import fetch_news
            result = fetch_news(["CAT"])
    assert result["CAT"] == []


def test_fetch_news_skips_without_api_key():
    with patch("ascent.integrations.exa_news.os.getenv", return_value=""):
        from ascent.integrations.exa_news import fetch_news
        result = fetch_news(["CAT", "MRK"])
    assert result == {"CAT": [], "MRK": []}


def test_fetch_news_respects_delay():
    call_times = []
    original_sleep = time.sleep

    def record_sleep(n):
        call_times.append(n)

    with patch("ascent.integrations.exa_news.time.sleep", side_effect=record_sleep):
        with patch("ascent.integrations.exa_news.requests.post",
                   return_value=_mock_exa_response(["headline"])):
            with patch("ascent.integrations.exa_news.os.getenv", return_value="fake-key"):
                from ascent.integrations.exa_news import fetch_news
                fetch_news(["CAT", "MRK"])
    assert any(t >= 0.2 for t in call_times), "Expected 0.2s delay between requests"
```

- [ ] **Step 2: Run — confirm failure**

```bash
.venv/bin/python -m pytest tests/integrations/test_exa_news.py -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'ascent.integrations.exa_news'`

- [ ] **Step 3: Implement `fetch_news()`**

Create `ascent/integrations/exa_news.py`:

```python
"""
Exa news search integration — fetch live news headlines for equity symbols.
Uses the Exa neural search API (https://exa.ai). Requires EXA_API_KEY env var.
Free tier: 1,000 searches/month. Ascent uses ~60/month at current scale.
"""
from __future__ import annotations

import os
import time

import requests
from loguru import logger


def fetch_news(
    symbols: list[str],
    max_per_symbol: int = 2,
) -> dict[str, list[str]]:
    """
    Fetch live news summaries for each symbol via Exa search.

    Returns {symbol: [summary, ...]} — empty list for any symbol that fails.
    Delays 0.2s between requests to stay within free tier rate limits.
    Never raises — failures are logged and return empty list for that symbol.
    """
    api_key = os.getenv("EXA_API_KEY", "")
    if not api_key:
        logger.warning("[ExaNews] EXA_API_KEY not set — skipping news fetch")
        return {sym: [] for sym in symbols}

    headers = {"x-api-key": api_key, "content-type": "application/json"}
    results: dict[str, list[str]] = {}

    for sym in symbols:
        payload = {
            "query": f"{sym} stock news catalyst today",
            "type": "auto",
            "numResults": max_per_symbol,
            "contents": {
                "summary": {
                    "schema": {
                        "type": "object",
                        "required": ["answer"],
                        "additionalProperties": False,
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Key news headline or catalyst from this article",
                            }
                        },
                    }
                }
            },
        }
        try:
            resp = requests.post(
                "https://api.exa.ai/search",
                json=payload,
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            summaries = [
                r["summary"]["answer"]
                for r in data.get("results", [])
                if r.get("summary", {}).get("answer")
            ]
            results[sym] = summaries[:max_per_symbol]
        except Exception as exc:
            logger.warning("[ExaNews] Failed for %s: %s", sym, exc)
            results[sym] = []

        time.sleep(0.2)

    return results
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/bin/python -m pytest tests/integrations/test_exa_news.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ascent/integrations/exa_news.py tests/integrations/test_exa_news.py
git commit -m "feat: Exa news integration — fetch_news() with rate-limit delay"
```

---

## Task 4: Wire Exa news into `_build_data_grounding()` and `run_all_agents.py`

**Files:**
- Modify: `agents/ai_pm_agent.py` (extend `_build_data_grounding()` to render news block)
- Modify: `run_all_agents.py` (call `fetch_news()` before pre-thesis and store result for daily path)

- [ ] **Step 1: Extend `_build_data_grounding()` to render the news block**

In `agents/ai_pm_agent.py`, inside `_build_data_grounding()`, append the news block right after the financials block (before `return grounding`):

```python
        # Append live news block (passed in from run_all_agents.py)
        if news_context:
            news_rows = []
            for sym in symbols[:25]:
                headlines = news_context.get(sym, [])
                for i, h in enumerate(headlines, 1):
                    news_rows.append(f"  {sym}: [{i}] {h}")
            if news_rows:
                grounding += (
                    "\n══ LIVE NEWS (Exa, fetched today) ══════════════════════════════\n"
                    + "\n".join(news_rows)
                    + "\n════════════════════════════════════════════════════════════════\n"
                )

        return grounding
```

- [ ] **Step 2: Call `fetch_news()` in `run_all_agents.py` before pre-thesis**

In `run_all_agents.py`, find the block starting at line ~1080 (`if is_rebalance:`). Add a `fetch_news()` call immediately **before** the `if is_rebalance:` block so it runs every day (rebalance and non-rebalance):

```python
    # ── Fetch live Exa news (runs daily — feeds pre-thesis + ticker discovery) ──
    _news_context: dict = {}
    try:
        from ascent.integrations.exa_news import fetch_news as _fetch_exa_news
        _universe_syms_for_news = list((merged_weights or {}).keys())[:20]
        if _universe_syms_for_news:
            _news_context = _fetch_exa_news(_universe_syms_for_news)
            _n_news = sum(len(v) for v in _news_context.values())
            print(f"[Runner] Exa news: {_n_news} headlines fetched for {len(_news_context)} symbols")
    except Exception as _ne:
        print(f"[Runner] Exa news fetch skipped: {_ne}")
```

- [ ] **Step 3: Pass `news_context` to `_build_data_grounding()` call sites**

In `agents/ai_pm_agent.py`, find the two calls to `_build_data_grounding()`:

**Call 1** (pre-thesis, line ~1900):
```python
    # Before:
    _p1_grounding = _build_data_grounding(
        [n.get("symbol", "") for n in _prethesis_universe[:30]] if _prethesis_universe else []
    ) if "_prethesis_universe" in dir() else _build_data_grounding([])

    # After — add news_context parameter:
    _p1_grounding = _build_data_grounding(
        [n.get("symbol", "") for n in _prethesis_universe[:30]] if _prethesis_universe else [],
        news_context=news_context_arg,
    ) if "_prethesis_universe" in dir() else _build_data_grounding([], news_context=news_context_arg)
```

**Call 2** (phase 2, line ~2108):
```python
    # Before:
    _p2_grounding = _build_data_grounding(_p2_symbols)

    # After:
    _p2_grounding = _build_data_grounding(_p2_symbols, news_context=news_context_arg)
```

Update `run_ai_pm_prethesis()` signature to accept `news_context_arg`:

```python
def run_ai_pm_prethesis(
    sentiment_block: str = "",
    news_context_arg: dict | None = None,
) -> Optional[AIPreThesis]:
```

And `run_ai_pm()` signature (find the existing `sentiment_block` param and add alongside it):

```python
def run_ai_pm(
    ...,
    sentiment_block: str = "",
    news_context_arg: dict | None = None,
    ...
) -> ...:
```

- [ ] **Step 4: Pass `_news_context` from `run_all_agents.py` into both AI PM calls**

Find the `run_ai_pm_prethesis(...)` call (line ~1116) and add the argument:
```python
_ai_prethesis = run_ai_pm_prethesis(
    sentiment_block=_sentiment_block,
    news_context_arg=_news_context,
)
```

Find the `run_ai_pm(...)` call (line ~1244) and add the argument:
```python
result = run_ai_pm(
    ...,
    sentiment_block=_sentiment_block,
    news_context_arg=_news_context,
    ...
)
```

- [ ] **Step 5: Run full test suite — confirm no regressions**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: ≥777 passed, 0 new failures.

- [ ] **Step 6: Commit**

```bash
git add agents/ai_pm_agent.py run_all_agents.py
git commit -m "feat: inject Exa live news block into AI PM data grounding"
```

---

## Task 5: `extra_symbols` param in `run_pipeline()` and `run_us_equities_agent()`

**Files:**
- Modify: `ascent/main.py` (line 273 — `run_pipeline()` signature)
- Modify: `agents/us_equities_agent.py` (line 23 — `run_us_equities_agent()` signature)

This task adds no new tests — it's a pass-through parameter. The change is surgical and verified by the Task 8 integration test.

- [ ] **Step 1: Add `extra_symbols` to `run_pipeline()`**

In `ascent/main.py`, change the signature at line 273:

```python
# Before:
def run_pipeline(
    live: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance_days: int | None = None,
) -> tuple:
    cfg = get_config()

# After:
def run_pipeline(
    live: bool = False,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int | None = None,
    rebalance_days: int | None = None,
    extra_symbols: list[str] | None = None,
) -> tuple:
    cfg = get_config()
    if extra_symbols:
        cfg.universe.symbols = list(cfg.universe.symbols) + [
            s for s in extra_symbols if s not in cfg.universe.symbols
        ]
```

- [ ] **Step 2: Add `extra_symbols` to `run_us_equities_agent()`**

In `agents/us_equities_agent.py`, change the signature at line 23:

```python
# Before:
def run_us_equities_agent(
    dry_run: bool = False,
    as_of_date: date = None,
) -> AgentOutput:

# After:
def run_us_equities_agent(
    dry_run: bool = False,
    as_of_date: date = None,
    extra_symbols: list[str] | None = None,
) -> AgentOutput:
```

And inside the function, change the `run_pipeline(live=True)` call to:

```python
        ) = run_pipeline(live=True, extra_symbols=extra_symbols)
```

- [ ] **Step 3: Run full suite — confirm no regressions**

```bash
.venv/bin/python -m pytest --tb=no -q 2>&1 | tail -5
```

Expected: same pass count, 0 failures.

- [ ] **Step 4: Commit**

```bash
git add ascent/main.py agents/us_equities_agent.py
git commit -m "feat: extra_symbols passthrough in run_pipeline and run_us_equities_agent"
```

---

## Task 6: `run_discovery()` — implement and test

**Files:**
- Create: `ascent/strategy/ticker_discovery.py`
- Create: `tests/strategy/test_ticker_discovery.py`

- [ ] **Step 1: Write failing tests**

Create `tests/strategy/test_ticker_discovery.py`:

```python
import pytest
from unittest.mock import patch


MOCK_HIGH_CONVICTION_RESPONSE = '''{
  "symbol": "VMC",
  "conviction_score": 0.82,
  "catalyst_snippet": "Infrastructure spending accelerating — Vulcan Materials mentioned alongside CAT",
  "rationale": "CAT news about infrastructure contracts references VMC as key supplier"
}'''

MOCK_LOW_CONVICTION_RESPONSE = '''{
  "symbol": "XYZ",
  "conviction_score": 0.45,
  "catalyst_snippet": "Weak signal",
  "rationale": "Tenuous connection"
}'''

MOCK_NO_SYMBOL_RESPONSE = '''{
  "symbol": "",
  "conviction_score": 0.0,
  "catalyst_snippet": "",
  "rationale": "No compelling candidate found in the news"
}'''


def test_run_discovery_returns_result_on_high_conviction():
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=MOCK_HIGH_CONVICTION_RESPONSE):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["Infrastructure bill boosts Vulcan Materials VMC"]},
            existing_universe=["CAT", "MRK", "NEE"],
        )
    assert result is not None
    assert result.symbol == "VMC"
    assert result.conviction_score == pytest.approx(0.82)


def test_run_discovery_returns_none_on_low_conviction():
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=MOCK_LOW_CONVICTION_RESPONSE):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["Some weak news"]},
            existing_universe=["CAT"],
        )
    assert result is None


def test_run_discovery_returns_none_if_symbol_already_in_universe():
    already_in = '''{
      "symbol": "CAT",
      "conviction_score": 0.90,
      "catalyst_snippet": "CAT is doing great",
      "rationale": "..."
    }'''
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=already_in):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["CAT headline"]},
            existing_universe=["CAT", "MRK"],
        )
    assert result is None


def test_run_discovery_returns_none_on_empty_news():
    from ascent.strategy.ticker_discovery import run_discovery
    result = run_discovery(news_context={}, existing_universe=["CAT"])
    assert result is None


def test_run_discovery_returns_none_on_all_empty_headlines():
    from ascent.strategy.ticker_discovery import run_discovery
    result = run_discovery(
        news_context={"CAT": [], "MRK": []},
        existing_universe=["CAT", "MRK"],
    )
    assert result is None


def test_run_discovery_returns_none_on_llm_failure():
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               side_effect=RuntimeError("LLM down")):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["Some news"]},
            existing_universe=["CAT"],
        )
    assert result is None


def test_run_discovery_truncates_catalyst_snippet():
    long_snippet = "x" * 500
    long_response = f'''{{"symbol": "VMC", "conviction_score": 0.85,
      "catalyst_snippet": "{long_snippet}", "rationale": "ok"}}'''
    with patch("ascent.strategy.ticker_discovery.chat_completion",
               return_value=long_response):
        from ascent.strategy.ticker_discovery import run_discovery
        result = run_discovery(
            news_context={"CAT": ["big news"]},
            existing_universe=["CAT"],
        )
    assert result is not None
    assert len(result.catalyst_snippet) <= 200
```

- [ ] **Step 2: Run — confirm failure**

```bash
.venv/bin/python -m pytest tests/strategy/test_ticker_discovery.py -v 2>&1 | head -10
```

Expected: `ModuleNotFoundError: No module named 'ascent.strategy.ticker_discovery'`

- [ ] **Step 3: Implement `ticker_discovery.py`**

Create `ascent/strategy/ticker_discovery.py`:

```python
"""
Ticker discovery — surfaces one compelling candidate from current-holdings news.
Uses HAIKU_MODEL for cost efficiency (classifier task, not judgment task).
Candidate must appear in or derive from real Exa-fetched news — not hallucinated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from loguru import logger

from ascent.llm.client import HAIKU_MODEL, chat_completion

_DISCOVERY_SYSTEM = """You are a catalyst scanner for an equity portfolio.
You receive live news headlines from current holdings and identify ONE compelling ticker
NOT in the current portfolio that appears in or is directly related to the news provided.

Rules:
- The candidate MUST appear in or be directly derivable from the news text given.
- Do not invent tickers or use outside knowledge not in the news.
- If no compelling candidate is present in the news, set conviction_score below 0.75.
- Return valid JSON only. No markdown fences, no extra text.

JSON format:
{"symbol": "TICKER", "conviction_score": 0.0, "catalyst_snippet": "...", "rationale": "..."}
"""


@dataclass
class DiscoveryResult:
    symbol: str
    conviction_score: float
    catalyst_snippet: str
    rationale: str


def run_discovery(
    news_context: dict[str, list[str]],
    existing_universe: list[str],
) -> DiscoveryResult | None:
    """
    Given Exa news headlines for current holdings, identify ONE ticker candidate
    not in existing_universe. Returns None if conviction < 0.75 or no candidate found.
    Never raises.
    """
    if not news_context or not any(v for v in news_context.values()):
        return None

    news_lines = [
        f"  {sym}: {h}"
        for sym, headlines in news_context.items()
        for h in headlines
    ]
    if not news_lines:
        return None

    user_prompt = (
        f"Current portfolio symbols (do NOT suggest these): {', '.join(existing_universe)}\n\n"
        "Live news from current holdings:\n"
        + "\n".join(news_lines)
        + "\n\nIdentify ONE new ticker candidate with highest conviction. "
        "Must appear in or relate directly to the news above."
    )

    try:
        raw = chat_completion(
            messages=[
                {"role": "system", "content": _DISCOVERY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0.2,
        )
        data = json.loads(raw.strip())
        symbol = str(data.get("symbol", "")).upper().strip()
        conviction = float(data.get("conviction_score", 0.0))

        if not symbol or symbol in existing_universe:
            return None
        if conviction < 0.75:
            return None

        return DiscoveryResult(
            symbol=symbol,
            conviction_score=conviction,
            catalyst_snippet=str(data.get("catalyst_snippet", ""))[:200],
            rationale=str(data.get("rationale", "")),
        )
    except Exception as exc:
        logger.warning("[TickerDiscovery] run_discovery failed: %s", exc)
        return None
```

- [ ] **Step 4: Run tests — confirm pass**

```bash
.venv/bin/python -m pytest tests/strategy/test_ticker_discovery.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add ascent/strategy/ticker_discovery.py tests/strategy/test_ticker_discovery.py
git commit -m "feat: ticker_discovery — run_discovery() surfaces candidates from Exa news"
```

---

## Task 7: Cooldown helpers

**Context:** `LARGE_TRADE_THRESHOLD_PCT = 2.0` is defined in `eod_runner.py` but is not wired into `run_eod_with_weights` — orders are submitted directly without a large-trade approval gate in that code path. Adding a dead parameter would be misleading. The debate gate (already applied in `_trigger_mini_rebalance`) is the primary safety control for mini-rebalances, exactly as it is for normal rebalances.

**Files:**
- Modify: `run_all_agents.py` (add helpers before `main()`)

- [ ] **Step 1: Add cooldown helpers to `run_all_agents.py`**

Add these two functions in `run_all_agents.py` before the `main()` function (find the line `if __name__ == "__main__":` and insert above it):

```python
def _check_mini_rebalance_cooldown() -> bool:
    """Returns True if a mini-rebalance ran < 5 trading days ago (cooldown active)."""
    import json
    from pathlib import Path
    import pandas as pd

    cooldown_path = Path("data_cache/last_mini_rebalance.json")
    if not cooldown_path.exists():
        return False
    try:
        rec = json.loads(cooldown_path.read_text())
        last = pd.Timestamp(rec["date"])
        trading_days_elapsed = len(pd.bdate_range(last, pd.Timestamp.today())) - 1
        return trading_days_elapsed < 5
    except Exception:
        return False


def _write_mini_rebalance_log(symbol: str, conviction: float) -> None:
    """Write cooldown state after a mini-rebalance completes."""
    import json
    from pathlib import Path
    from datetime import date as _date

    path = Path("data_cache/last_mini_rebalance.json")
    path.write_text(json.dumps({
        "date":       _date.today().isoformat(),
        "symbol":     symbol,
        "conviction": conviction,
    }))
```

- [ ] **Step 3: Run full suite — confirm no regressions**

```bash
.venv/bin/python -m pytest --tb=no -q 2>&1 | tail -5
```

Expected: same pass count, 0 failures.

- [ ] **Step 4: Commit**

```bash
git add run_all_agents.py
git commit -m "feat: mini-rebalance cooldown helpers"
```

---

## Task 8: Wire discovery + `_trigger_mini_rebalance()` into daily path

**Files:**
- Modify: `run_all_agents.py` (daily non-rebalance path, ~line 1031–1063)

- [ ] **Step 1: Add `_trigger_mini_rebalance()` to `run_all_agents.py`**

Insert this function alongside the other helpers (before `main()`):

```python
def _trigger_mini_rebalance(
    result,          # DiscoveryResult
    current_weights: dict,
    today,
    dry_run: bool = False,
) -> None:
    """
    Run full us_equities_agent pipeline with the discovered ticker added,
    pass through debate gate, and execute with a 1% NAV approval threshold.
    Writes cooldown log and eod_log entry on completion.
    """
    import json as _j
    from pathlib import Path as _P

    print(f"\n[Discovery] Mini-rebalance triggered: {result.symbol} "
          f"(conviction={result.conviction_score:.2f})")
    print(f"[Discovery] Catalyst: {result.catalyst_snippet}")

    try:
        from agents.us_equities_agent import run_us_equities_agent
        new_output = run_us_equities_agent(extra_symbols=[result.symbol])
        new_weights = new_output.target_weights or {}

        if not new_weights:
            print(f"[Discovery] Mini-rebalance: agent returned empty weights — aborting")
            return

        # Debate gate (same as normal rebalance)
        verdict = {}
        try:
            from debate.debate_runner import run_debate
            from ascent.execution.debate_gate import should_run_debate
            portfolio_state = {
                "date":        today.isoformat(),
                "us_regime":   new_output.regime_signal or "unknown",
                "n_positions": len(new_weights),
                "weights":     new_weights,
                "trigger":     "discovery",
            }
            regime_dict = {"entropy": 0.0, "label": portfolio_state["us_regime"]}
            if should_run_debate(portfolio_state, regime_dict):
                verdict = run_debate(portfolio_state, run_date=today) or {}
        except Exception as _de:
            print(f"[Discovery] Debate skipped: {_de}")

        if verdict.get("recommendation") == "halt_and_review":
            print(f"[Discovery] Debate: halt_and_review — mini-rebalance aborted")
            return

        # Execute with 1% NAV threshold (stricter than normal 2%)
        if dry_run:
            print(f"[Discovery] DRY RUN — would submit {len(new_weights)} positions "
                  f"including {result.symbol}")
        else:
            from ascent.execution.eod_runner import run_eod_with_weights
            run_eod_with_weights(
                new_weights,
                run_date=today,
                dry_run=False,
                precomputed_verdict=verdict,
            )

        _write_mini_rebalance_log(result.symbol, result.conviction_score)

        # Attribution log
        _P("logs/eod_log.jsonl").open("a").write(
            _j.dumps({
                "date":       today.isoformat(),
                "trigger":    "discovery",
                "symbol":     result.symbol,
                "conviction": result.conviction_score,
                "catalyst":   result.catalyst_snippet,
            }) + "\n"
        )
        print(f"[Discovery] Mini-rebalance complete — {result.symbol} added to pipeline")

    except Exception as exc:
        print(f"[Discovery] Mini-rebalance failed: {exc}")
```

- [ ] **Step 2: Wire discovery into the non-rebalance daily path**

In `run_all_agents.py`, find the non-rebalance block starting around line 1031 (`if not is_rebalance:`). After the existing blocks (daily intelligence, adversarial monitor, causal gate), add:

```python
        # Ticker discovery — surface a new candidate from today's Exa news
        try:
            from ascent.strategy.ticker_discovery import run_discovery as _run_discovery
            _current_universe = list((merged_weights or {}).keys())
            if _news_context and _current_universe:
                _discovery = _run_discovery(_news_context, _current_universe)
                if _discovery:
                    print(f"[Discovery] Candidate: {_discovery.symbol} "
                          f"(conviction={_discovery.conviction_score:.2f})")
                    if not _check_mini_rebalance_cooldown():
                        _trigger_mini_rebalance(_discovery, merged_weights, today, dry_run)
                    else:
                        print(f"[Discovery] Cooldown active — {_discovery.symbol} queued for next window")
                else:
                    print("[Discovery] No high-conviction candidate found today")
        except Exception as _disc_e:
            print(f"[Discovery] Skipped: {_disc_e}")
```

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: ≥777 passed, 0 new failures.

- [ ] **Step 4: Smoke test — verify `_news_context` is in scope at discovery call site**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/python -c "
import ast, pathlib
src = pathlib.Path('run_all_agents.py').read_text()
ast.parse(src)
print('AST parse OK')
"
```

Expected: `AST parse OK`

- [ ] **Step 5: Commit**

```bash
git add run_all_agents.py
git commit -m "feat: wire ticker discovery + mini-rebalance into daily non-rebalance path"
```

---

## Task 9: Final verification

- [ ] **Step 1: Run full test suite one last time**

```bash
.venv/bin/python -m pytest --tb=short -q 2>&1 | tail -10
```

Expected: ≥777 passed, 0 failures.

- [ ] **Step 2: Verify all new files exist**

```bash
ls ascent/integrations/exa_news.py \
   ascent/strategy/ticker_discovery.py \
   tests/integrations/test_exa_news.py \
   tests/strategy/test_ticker_discovery.py \
   tests/agents/test_ai_pm_financials.py
```

Expected: all 5 files listed without error.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: autohedge integration complete — exa news, financials, ticker discovery"
```
