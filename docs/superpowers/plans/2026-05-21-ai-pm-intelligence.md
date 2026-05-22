# AI PM Intelligence Upgrades Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the AI PM three new capabilities — live news, analyst consensus estimates, and calibration-aware confidence — so it makes better-informed decisions when it earns real weight in ~September 2026.

**Architecture:** Two new tools (`get_live_news`, `get_analyst_estimates`) added to `agents/ai_pm_agent.py` using yfinance (already a dependency, no new API keys). A calibration gate auto-injects a warning into the system prompt when the AI PM's conviction IC is below 0.05 so it knows to discount its own confidence. Tool count goes from 17 → 19.

**Tech Stack:** Python 3.12, yfinance (existing dep), `ascent/llm/client.py`, `ascent/strategy/calibration_tracker.py`.

---

### Task 1: `get_live_news` Tool

**Files:**
- Modify: `agents/ai_pm_agent.py` (add tool schema + executor + update tool list)
- Test: `tests/agents/test_ai_pm_news_tool.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_ai_pm_news_tool.py
import pytest
from unittest.mock import patch, MagicMock
import time


def _make_news_item(title: str, age_hours: float = 12.0) -> dict:
    return {
        "title": title,
        "providerPublishTime": int(time.time() - age_hours * 3600),
        "publisher": "Reuters",
    }


def test_get_live_news_returns_headlines(monkeypatch):
    from agents.ai_pm_agent import _tool_get_live_news

    mock_ticker = MagicMock()
    mock_ticker.news = [
        _make_news_item("AAPL beats earnings estimates by 12%", 10),
        _make_news_item("Apple expands AI features to iPhone 16", 20),
        _make_news_item("Old news", 100),  # > 72h — should be filtered
    ]
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = _tool_get_live_news({"symbol": "AAPL"})

    assert "AAPL beats earnings" in result
    assert "Old news" not in result


def test_get_live_news_no_symbol():
    from agents.ai_pm_agent import _tool_get_live_news
    result = _tool_get_live_news({})
    assert "Error" in result or "symbol" in result.lower()


def test_get_live_news_handles_yf_failure(monkeypatch):
    from agents.ai_pm_agent import _tool_get_live_news
    with patch("yfinance.Ticker", side_effect=Exception("network error")):
        result = _tool_get_live_news({"symbol": "AAPL"})
    assert "failed" in result.lower() or "error" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_news_tool.py -v
```
Expected: `ImportError: cannot import name '_tool_get_live_news'`

- [ ] **Step 3: Add `_tool_get_live_news` to `agents/ai_pm_agent.py`**

Find the existing tool executor functions (they follow the `_tool_*` naming pattern). Add after `_tool_get_rebalance_brief`:

```python
def _tool_get_live_news(inputs: dict) -> str:
    """Fetch last 72h news headlines for a symbol via yfinance."""
    import time as _time
    from datetime import datetime as _dt
    symbol = inputs.get("symbol", "").upper().strip()
    if not symbol:
        return "Error: symbol required"
    try:
        import yfinance as yf
        news = yf.Ticker(symbol).news or []
        cutoff = _time.time() - 72 * 3600
        recent = [n for n in news if n.get("providerPublishTime", 0) > cutoff][:5]
        if not recent:
            return f"No news in last 72h for {symbol}."
        lines = [f"{symbol} news (last 72h):"]
        for n in recent:
            ts = _dt.fromtimestamp(n["providerPublishTime"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [{ts}] {n.get('title', 'No title')} — {n.get('publisher', '')}")
        return "\n".join(lines)
    except Exception as e:
        return f"News fetch failed for {symbol}: {e}"
```

- [ ] **Step 4: Register the tool schema in `AI_PM_TOOLS`**

Find `AI_PM_TOOLS` in `agents/ai_pm_agent.py`. Add the new tool entry (keep alphabetical-ish ordering, add after `get_rebalance_brief`):

```python
{
    "name": "get_live_news",
    "description": (
        "Fetch last 72 hours of news headlines for a specific ticker symbol. "
        "Use this to check for recent earnings, guidance changes, M&A, management changes, "
        "or macro events that could affect a position thesis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
        },
        "required": ["symbol"],
    },
},
```

- [ ] **Step 5: Register in `_make_executor`**

Find the `_make_executor` function (returns a dict mapping tool names to callables). Add:

```python
"get_live_news": _tool_get_live_news,
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_news_tool.py -v
```
Expected: `3 passed`

- [ ] **Step 7: Commit**

```bash
git add agents/ai_pm_agent.py tests/agents/test_ai_pm_news_tool.py
git commit -m "feat: add get_live_news tool to AI PM (last 72h headlines via yfinance)"
```

---

### Task 2: `get_analyst_estimates` Tool

**Files:**
- Modify: `agents/ai_pm_agent.py` (add tool schema + executor)
- Test: `tests/agents/test_ai_pm_analyst_tool.py`

**What this gives the AI PM:** Forward P/E, analyst target price, number of analysts covering, recommendation mean (1=Strong Buy, 5=Strong Sell). Lets it compare its view against Wall Street consensus before making an override.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_ai_pm_analyst_tool.py
import pytest
from unittest.mock import patch, MagicMock


def _mock_info():
    return {
        "forwardPE": 28.5,
        "targetMeanPrice": 225.0,
        "numberOfAnalystOpinions": 42,
        "recommendationMean": 1.8,
        "earningsGrowth": 0.15,
        "revenueGrowth": 0.08,
    }


def test_analyst_estimates_returns_data(monkeypatch):
    from agents.ai_pm_agent import _tool_get_analyst_estimates

    mock_ticker = MagicMock()
    mock_ticker.info = _mock_info()
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = _tool_get_analyst_estimates({"symbol": "AAPL"})

    assert "28.5" in result           # forward PE
    assert "42" in result             # analyst count
    assert "225.0" in result          # target price


def test_analyst_estimates_no_symbol():
    from agents.ai_pm_agent import _tool_get_analyst_estimates
    result = _tool_get_analyst_estimates({})
    assert "Error" in result or "symbol" in result.lower()


def test_analyst_estimates_handles_failure(monkeypatch):
    from agents.ai_pm_agent import _tool_get_analyst_estimates
    with patch("yfinance.Ticker", side_effect=Exception("timeout")):
        result = _tool_get_analyst_estimates({"symbol": "AAPL"})
    assert "failed" in result.lower() or "error" in result.lower()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_analyst_tool.py -v
```
Expected: `ImportError: cannot import name '_tool_get_analyst_estimates'`

- [ ] **Step 3: Add `_tool_get_analyst_estimates` to `agents/ai_pm_agent.py`**

Add after `_tool_get_live_news`:

```python
def _tool_get_analyst_estimates(inputs: dict) -> str:
    """Fetch forward valuation and analyst consensus for a symbol via yfinance."""
    symbol = inputs.get("symbol", "").upper().strip()
    if not symbol:
        return "Error: symbol required"
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        fields = [
            ("forwardPE",               "Forward P/E"),
            ("priceToBook",             "Price/Book"),
            ("targetMeanPrice",         "Analyst target (mean)"),
            ("targetLowPrice",          "Analyst target (low)"),
            ("targetHighPrice",         "Analyst target (high)"),
            ("numberOfAnalystOpinions", "# analysts covering"),
            ("recommendationMean",      "Rec mean (1=Strong Buy, 5=Strong Sell)"),
            ("earningsGrowth",          "Earnings growth (YoY)"),
            ("revenueGrowth",           "Revenue growth (YoY)"),
        ]
        lines = [f"{symbol} analyst consensus:"]
        for key, label in fields:
            val = info.get(key)
            if val is not None:
                if isinstance(val, float) and key.endswith("Growth"):
                    lines.append(f"  {label}: {val*100:.1f}%")
                else:
                    lines.append(f"  {label}: {val}")
        if len(lines) == 1:
            return f"No analyst data available for {symbol}."
        return "\n".join(lines)
    except Exception as e:
        return f"Analyst data failed for {symbol}: {e}"
```

- [ ] **Step 4: Register tool schema in `AI_PM_TOOLS`**

Add:

```python
{
    "name": "get_analyst_estimates",
    "description": (
        "Fetch analyst consensus for a ticker: forward P/E, target price range, "
        "number of analysts, recommendation mean (1=Strong Buy), earnings and revenue growth. "
        "Use before making a high-conviction override of the quant signal."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
        },
        "required": ["symbol"],
    },
},
```

- [ ] **Step 5: Register in `_make_executor`**

```python
"get_analyst_estimates": _tool_get_analyst_estimates,
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_analyst_tool.py -v
```
Expected: `3 passed`

- [ ] **Step 7: Commit**

```bash
git add agents/ai_pm_agent.py tests/agents/test_ai_pm_analyst_tool.py
git commit -m "feat: add get_analyst_estimates tool to AI PM (forward PE, target, rec mean)"
```

---

### Task 3: Calibration-Aware System Prompt

**Files:**
- Modify: `agents/ai_pm_agent.py` (`run_ai_pm` function — inject calibration warning into system prompt when IC < 0.05)
- Test: `tests/agents/test_ai_pm_calibration_gate.py`

**What this does:** When the AI PM's historical conviction IC is below 0.05 (Uncalibrated), a one-sentence warning is prepended to the system prompt before the tool loop runs. The AI PM will naturally hedge its high-conviction calls. No extra LLM call — purely prompt engineering.

- [ ] **Step 1: Write the failing test**

```python
# tests/agents/test_ai_pm_calibration_gate.py
import pytest
from unittest.mock import patch, MagicMock


def _make_calibration_report(ic: float) -> dict:
    status = "Calibrated" if ic >= 0.20 else ("Weak" if ic >= 0.05 else "Uncalibrated")
    return {"spearman_ic": ic, "status": status, "n_rebalances": 5}


def test_uncalibrated_injects_warning(monkeypatch):
    """When IC < 0.05, system prompt contains calibration warning."""
    from agents import ai_pm_agent

    captured_prompts = []

    def mock_tool_completion(system_prompt, user_prompt, tools, tool_executor,
                              model, max_tokens, max_tool_calls, use_cache=False):
        captured_prompts.append(system_prompt)
        return '{"action": "propose_portfolio", "weights": {"SPY": 1.0}, "thesis": {}}'

    monkeypatch.setattr(ai_pm_agent, "_get_calibration_report_safe",
                        lambda n: _make_calibration_report(0.02))
    monkeypatch.setattr("ascent.llm.client.tool_completion", mock_tool_completion)

    # run_ai_pm will call tool_completion — check the system prompt
    # (use a minimal mock to avoid full pipeline)
    prompt = ai_pm_agent._build_system_prompt(ic=0.02)
    assert "CALIBRATION WARNING" in prompt or "uncalibrated" in prompt.lower()


def test_calibrated_no_warning():
    from agents import ai_pm_agent
    prompt = ai_pm_agent._build_system_prompt(ic=0.25)
    assert "CALIBRATION WARNING" not in prompt
    assert "uncalibrated" not in prompt.lower()


def test_build_system_prompt_none_ic():
    """None IC (no data yet) — no warning injected."""
    from agents import ai_pm_agent
    prompt = ai_pm_agent._build_system_prompt(ic=None)
    assert isinstance(prompt, str)
    assert len(prompt) > 100
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_calibration_gate.py -v
```
Expected: `ImportError: cannot import name '_build_system_prompt'`

- [ ] **Step 3: Extract `_build_system_prompt` from `run_ai_pm` in `agents/ai_pm_agent.py`**

Find `_SYSTEM_PROMPT` (the big string constant at top of the file). Add a new function that returns it with optional calibration warning prepended:

```python
def _build_system_prompt(ic: float | None = None) -> str:
    """Return the AI PM system prompt, prepending a calibration warning if IC < 0.05."""
    base = _SYSTEM_PROMPT
    if ic is not None and ic < 0.05:
        warning = (
            "\n\n⚠️  CALIBRATION WARNING: Your recent conviction-vs-outcome IC is "
            f"{ic:.3f} (Uncalibrated, threshold 0.05). Your high-conviction overrides "
            "have not been predictive. Be conservative — prefer the quant baseline "
            "unless you have a clearly non-quantitative thesis (news, events, regime shift). "
            "Do not override quant on momentum or valuation alone this session.\n"
        )
        return warning + base
    return base
```

Also add a safe calibration report getter:

```python
def _get_calibration_report_safe(n_rebalances: int = 10) -> dict | None:
    """Load calibration report. Returns None on any failure."""
    try:
        from ascent.strategy.calibration_tracker import get_calibration_report
        return get_calibration_report(n_rebalances=n_rebalances)
    except Exception:
        return None
```

- [ ] **Step 4: Wire into `run_ai_pm`**

Find the `tool_completion` call inside `run_ai_pm`. Before that call, replace the hardcoded `_SYSTEM_PROMPT` with:

```python
    # Build system prompt with calibration gate
    _cal_report = _get_calibration_report_safe(n_rebalances=10)
    _ic = _cal_report.get("spearman_ic") if _cal_report else None
    _system = _build_system_prompt(ic=_ic)
```

Then pass `_system` instead of `_SYSTEM_PROMPT` into both `tool_completion` calls (main loop and revision pass).

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/agents/test_ai_pm_calibration_gate.py -v
```
Expected: `3 passed`

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/python -m pytest -q
```
Expected: 506+ passed, 1 skipped

- [ ] **Step 7: Commit**

```bash
git add agents/ai_pm_agent.py tests/agents/test_ai_pm_calibration_gate.py
git commit -m "feat: calibration-aware AI PM prompt — warns when conviction IC < 0.05"
```

---

## Self-Review

**Spec coverage check:**

1. ✅ `get_live_news` tool (yfinance, 72h window) → Task 1
2. ✅ `get_analyst_estimates` tool (forward PE, target, rec mean) → Task 2
3. ✅ Calibration-aware system prompt (IC gate) → Task 3

**Placeholder scan:** None found.

**Type consistency:**
- `_tool_get_live_news(inputs: dict) → str` — consistent across implementation and tests
- `_tool_get_analyst_estimates(inputs: dict) → str` — consistent
- `_build_system_prompt(ic: float | None) → str` — consistent
- `_get_calibration_report_safe(n_rebalances: int) → dict | None` — consistent
