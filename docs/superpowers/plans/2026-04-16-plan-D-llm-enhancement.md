.# Plan D — LLM Enhancement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the quant+AI gap. Right now debate agents make claims about correlation, VaR, and factor exposure but have no access to compute them. After this plan: agents receive pre-computed structured quant data, the judge uses extended thinking for harder synthesis, and repeated prompt prefixes are cached.

**Architecture:** Three additions: (1) a `QuantContext` builder that pre-computes factor exposures, VaR estimate, and attribution from the day's data before debate starts; (2) extended thinking for the judge; (3) prompt caching headers for the 4 system prompts that are identical on every rebalance day.

**Tech Stack:** `ascent/llm/client.py`, `debate/agents.py`, `debate/judge.py`, `debate/debate_runner.py`. New module: `ascent/monitoring/quant_context.py`.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ascent/monitoring/quant_context.py` | Pre-compute VaR, factor exposure, and attribution for debate context |
| Modify | `debate/agents.py:_build_context` | Inject structured quant data into every agent prompt |
| Modify | `debate/judge.py` | Add extended thinking for judge synthesis |
| Modify | `ascent/llm/client.py` | Add prompt caching support |
| Modify | `debate/debate_runner.py` | Build `QuantContext` before debate, pass to portfolio_state |

---

## Task D1: Pre-compute quantitative context for debate agents

**Problem:** Bull agent says "correlation risk is manageable." Bear says "VaR is elevated." Neither one has actually computed these numbers. The quant layer produces correlation matrices and VaR estimates during the run but never pipes them to the debate layer in structured form.

**The fix:** New `ascent/monitoring/quant_context.py` with `build_quant_context(weights, prices)` that returns factor exposures, portfolio VaR (historical simulation), and sector concentration — as a structured dict that becomes part of `portfolio_state`.

**Files:**
- Create: `ascent/monitoring/quant_context.py`
- Modify: `debate/debate_runner.py` — call `build_quant_context` before debate starts
- Modify: `debate/agents.py:_build_context` — render quant context in prompts

- [ ] **Step 1: Write failing test**

```python
# tests/test_plan_d.py
import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

def _make_prices(symbols, n_days=90, seed=42):
    np.random.seed(seed)
    idx = pd.date_range(end=date.today(), periods=n_days, freq="B")
    returns = np.random.normal(0.0003, 0.012, size=(n_days, len(symbols)))
    prices = 100 * np.cumprod(1 + returns, axis=0)
    return pd.DataFrame(prices, index=idx, columns=symbols)


def test_quant_context_keys():
    """build_quant_context must return all required keys."""
    from ascent.monitoring.quant_context import build_quant_context

    weights = {"AAPL": 0.10, "MSFT": 0.10, "EEM": 0.08, "GLD": 0.07,
               "TLT": 0.06, "SPY": 0.05, "JPM": 0.09, "XOM": 0.05,
               "NEE": 0.06, "MRK": 0.06, "WMT": 0.05, "AMZN": 0.07,
               "NVDA": 0.08, "V": 0.06, "MA": 0.02}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)

    required = ["portfolio_var_95", "portfolio_var_99", "factor_exposures",
                "sector_concentration", "top_correlated_pairs", "summary_text"]
    for key in required:
        assert key in ctx, f"quant_context missing key: {key}"


def test_quant_context_var_is_negative():
    """VaR should be a negative number (worst-case loss)."""
    from ascent.monitoring.quant_context import build_quant_context

    weights = {"AAPL": 0.30, "MSFT": 0.30, "AMZN": 0.20, "NVDA": 0.20}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)

    assert ctx["portfolio_var_95"] < 0, "VaR 95 should be negative (a loss)"
    assert ctx["portfolio_var_99"] < 0, "VaR 99 should be negative (a loss)"
    assert ctx["portfolio_var_99"] <= ctx["portfolio_var_95"], "99th pctile VaR must be >= 95th"


def test_quant_context_factor_exposures():
    """Factor exposures must sum to <= 1.0 and include em_equity if EEM present."""
    from ascent.monitoring.quant_context import build_quant_context

    weights = {"EEM": 0.12, "GLD": 0.08, "TLT": 0.10, "AAPL": 0.30, "JPM": 0.40}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)

    fe = ctx["factor_exposures"]
    assert "em_equity" in fe, "EEM must register as em_equity exposure"
    assert fe["em_equity"] > 0
    total_exposure = sum(fe.values())
    assert total_exposure <= 1.01, f"Factor exposures sum {total_exposure} > 1.0"


def test_quant_context_summary_text_is_string():
    """summary_text must be a non-empty string ready for agent injection."""
    from ascent.monitoring.quant_context import build_quant_context

    weights = {"AAPL": 0.25, "MSFT": 0.25, "EEM": 0.25, "GLD": 0.25}
    prices = _make_prices(list(weights.keys()))
    ctx = build_quant_context(weights, prices)

    assert isinstance(ctx["summary_text"], str)
    assert len(ctx["summary_text"]) > 50
    assert "VaR" in ctx["summary_text"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"
.venv/bin/pytest tests/test_plan_d.py::test_quant_context_keys tests/test_plan_d.py::test_quant_context_var_is_negative tests/test_plan_d.py::test_quant_context_factor_exposures tests/test_plan_d.py::test_quant_context_summary_text_is_string -v
```
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Create `ascent/monitoring/quant_context.py`**

```python
"""
ascent/monitoring/quant_context.py
Pre-compute quantitative context for LLM debate agents.

Computes factor exposures, historical VaR, and sector concentration
from portfolio weights and recent prices. Output is structured for
injection into debate agent prompts so agents argue with numbers,
not vague claims.

Called by debate_runner.py before the debate starts.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Factor buckets (mirrors orchestrator/central_intelligence.py)
FACTOR_BUCKETS = {
    "rates_long":   {"TLT", "IEF", "LQD"},
    "rates_short":  {"HYG", "JNK"},
    "dollar_long":  {"UUP"},
    "commodities":  {"PDBC", "USO", "DBA", "DBB"},
    "gold":         {"GLD", "IAU"},
    "vol_long":     {"VIXY", "VXX"},
    "vol_short":    {"SVXY"},
    "em_equity":    {"EEM", "VWO", "EWT", "EWZ", "AAXJ", "EWY", "INDA"},
    "us_tech":      {"QQQ", "XLK", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN", "AVGO"},
    "us_defensive": {"XLU", "XLP", "XLV", "NEE", "WMT", "MRK", "JNJ", "PG"},
    "reits":        {"VNQ", "IYR", "EQIX"},
    "energy":       {"XLE", "MPC", "PSX", "XOM"},
}


def _compute_factor_exposures(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Map portfolio weights into factor buckets.
    Returns {factor_name: total_weight} for non-zero exposures.
    """
    exposures: Dict[str, float] = {}
    for factor, syms in FACTOR_BUCKETS.items():
        total = sum(weights.get(s, 0.0) for s in syms)
        if total > 0.001:
            exposures[factor] = round(total, 4)
    return exposures


def _compute_historical_var(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    lookback: int = 63,
    confidence_levels: tuple = (0.95, 0.99),
) -> Dict[str, float]:
    """
    Historical simulation VaR for the portfolio.
    Uses last `lookback` trading days of returns.

    Returns {var_95: float, var_99: float} — both negative (represent losses).
    """
    common = [s for s in weights if s in prices.columns]
    if not common:
        return {"var_95": 0.0, "var_99": 0.0}

    subset = prices[common].dropna(how="all").tail(lookback + 1)
    if len(subset) < 10:
        return {"var_95": 0.0, "var_99": 0.0}

    rets = subset.pct_change().dropna()
    w_arr = np.array([weights.get(s, 0.0) for s in common])
    total_w = w_arr.sum()
    if total_w > 0:
        w_arr /= total_w

    port_rets = rets.values @ w_arr
    result = {}
    for cl in confidence_levels:
        pctile = np.percentile(port_rets, (1 - cl) * 100)
        key = f"var_{int(cl*100)}"
        result[key] = round(float(pctile), 6)

    return result


def _compute_sector_concentration(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Group weights by broad factor and return the top concentrations.
    """
    exposures = _compute_factor_exposures(weights)
    total = sum(weights.values()) or 1.0
    # Add 'unclassified' for anything not in a factor
    classified = sum(exposures.values())
    unclassified = total - classified
    if unclassified > 0.01:
        exposures["unclassified"] = round(unclassified, 4)
    return {k: round(v / total, 4) for k, v in sorted(exposures.items(), key=lambda x: -x[1])}


def _compute_top_correlations(
    weights: Dict[str, float],
    prices: pd.DataFrame,
    lookback: int = 63,
    top_n: int = 3,
) -> List[Dict]:
    """
    Find the top_n highest-correlated pairs in the portfolio.
    High correlation = potential concentration risk.
    Returns list of {sym1, sym2, correlation}.
    """
    common = [s for s in weights if s in prices.columns and weights[s] > 0.02]
    if len(common) < 2:
        return []

    subset = prices[common].dropna(how="all").tail(lookback + 1)
    if len(subset) < 21:
        return []

    corr_matrix = subset.pct_change().dropna().corr()
    pairs = []
    for i, s1 in enumerate(common):
        for s2 in common[i+1:]:
            if s1 in corr_matrix.index and s2 in corr_matrix.columns:
                c = corr_matrix.loc[s1, s2]
                if not np.isnan(c):
                    pairs.append({"sym1": s1, "sym2": s2, "correlation": round(float(c), 3)})

    pairs.sort(key=lambda x: -abs(x["correlation"]))
    return pairs[:top_n]


def _build_summary_text(
    weights: Dict[str, float],
    var_data: Dict[str, float],
    factor_exposures: Dict[str, float],
    top_corr: List[Dict],
) -> str:
    """
    Render quantitative context as a plain-English block for agent prompts.
    """
    lines = ["QUANTITATIVE RISK CONTEXT (pre-computed):"]

    # VaR
    var_95 = var_data.get("var_95", 0.0)
    var_99 = var_data.get("var_99", 0.0)
    lines.append(f"\nPortfolio VaR (historical simulation, 63 trading days):")
    lines.append(f"  95th percentile (1-day): {var_95:.2%}  (1 in 20 chance of loss >= this)")
    lines.append(f"  99th percentile (1-day): {var_99:.2%}  (1 in 100 chance of loss >= this)")

    # Factor exposures
    lines.append("\nFactor exposures (sum of position weights per factor):")
    for factor, exp in sorted(factor_exposures.items(), key=lambda x: -x[1]):
        bar = "█" * int(exp * 20)
        lines.append(f"  {factor:<18} {exp:.1%}  {bar}")

    # High correlations
    high_corr = [p for p in top_corr if p["correlation"] > 0.65]
    if high_corr:
        lines.append("\nHigh-correlation pairs (potential double-counting of risk):")
        for p in high_corr:
            lines.append(f"  {p['sym1']} ↔ {p['sym2']}: {p['correlation']:.2f}")
    else:
        lines.append("\nCorrelation: No high-correlation pairs (>0.65) detected.")

    # Concentration warning
    top_factor = max(factor_exposures.items(), key=lambda x: x[1]) if factor_exposures else None
    if top_factor and top_factor[1] > 0.30:
        lines.append(f"\n⚠ Concentration: {top_factor[0]} is {top_factor[1]:.1%} of portfolio")

    return "\n".join(lines)


def build_quant_context(
    weights: Dict[str, float],
    prices: Optional[pd.DataFrame],
    lookback: int = 63,
) -> Dict:
    """
    Build quantitative context for debate agents.

    Args:
        weights:  {symbol: weight} — proposed portfolio weights
        prices:   DataFrame with symbol columns and date index (from price cache)
        lookback: Trading days for VaR and correlation computation

    Returns:
        Dict with portfolio_var_95, portfolio_var_99, factor_exposures,
        sector_concentration, top_correlated_pairs, summary_text.
    """
    factor_exposures  = _compute_factor_exposures(weights)
    sector_concentration = _compute_sector_concentration(weights)
    top_corr = []
    var_data = {"var_95": 0.0, "var_99": 0.0}

    if prices is not None and not prices.empty:
        raw_var = _compute_historical_var(weights, prices, lookback)
        var_data = {
            "var_95": raw_var.get("var_95", 0.0),
            "var_99": raw_var.get("var_99", 0.0),
        }
        top_corr = _compute_top_correlations(weights, prices, lookback)

    summary = _build_summary_text(weights, var_data, factor_exposures, top_corr)

    return {
        "portfolio_var_95":    var_data["var_95"],
        "portfolio_var_99":    var_data["var_99"],
        "factor_exposures":    factor_exposures,
        "sector_concentration": sector_concentration,
        "top_correlated_pairs": top_corr,
        "summary_text":        summary,
    }
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/pytest tests/test_plan_d.py::test_quant_context_keys tests/test_plan_d.py::test_quant_context_var_is_negative tests/test_plan_d.py::test_quant_context_factor_exposures tests/test_plan_d.py::test_quant_context_summary_text_is_string -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ascent/monitoring/quant_context.py tests/test_plan_d.py
git commit -m "feat(debate): quant_context builder — VaR, factor exposures, correlations for agents"
```

---

## Task D2: Inject quant context into debate agents

**Problem:** `_build_context()` in `debate/agents.py` shows portfolio weights and catalyst text but no quant numbers. Agents have the raw weights but argue in qualitative terms. The summary_text from `quant_context.py` needs to appear in every agent prompt.

**Files:**
- Modify: `debate/agents.py:_build_context`
- Modify: `debate/debate_runner.py` — call `build_quant_context` and store in `portfolio_state["quant_context"]`

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_d.py
def test_build_context_includes_quant_data():
    """_build_context must include VaR and factor data when quant_context present."""
    import sys; sys.path.insert(0, ".")
    from debate.agents import _build_context

    state = {
        "date": "2026-04-16",
        "us_regime": "stressed",
        "macro_regime": "stressed",
        "n_positions": 5,
        "allocation": {"us_equities": 0.45},
        "weights": {"AAPL": 0.15, "EEM": 0.12, "GLD": 0.08, "TLT": 0.10, "MRK": 0.10},
        "quant_context": {
            "portfolio_var_95": -0.0182,
            "portfolio_var_99": -0.0271,
            "factor_exposures": {"us_tech": 0.15, "em_equity": 0.12, "gold": 0.08},
            "top_correlated_pairs": [{"sym1": "AAPL", "sym2": "EEM", "correlation": 0.71}],
            "summary_text": "QUANTITATIVE RISK CONTEXT (pre-computed):\n\nPortfolio VaR: -1.82%",
        },
    }
    context = _build_context(state)
    assert "VaR" in context or "var" in context.lower(), "Context must include VaR from quant data"
    assert "-1.82%" in context or "QUANTITATIVE" in context, "quant_context summary_text must appear"


def test_build_context_no_quant_still_works():
    """_build_context must not break when quant_context is absent."""
    from debate.agents import _build_context

    state = {
        "date": "2026-04-16",
        "us_regime": "calm_bull",
        "n_positions": 3,
        "allocation": {},
        "weights": {"AAPL": 0.50, "MSFT": 0.50},
    }
    context = _build_context(state)  # must not raise
    assert "AAPL" in context
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_plan_d.py::test_build_context_includes_quant_data tests/test_plan_d.py::test_build_context_no_quant_still_works -v
```
Expected: FAIL — `_build_context` doesn't render quant context.

- [ ] **Step 3: Update `_build_context` in `debate/agents.py`**

Add after the memory context block (around line 50), before the return:

```python
    # Inject quantitative risk context
    quant_ctx = portfolio_state.get("quant_context")
    if quant_ctx and isinstance(quant_ctx, dict):
        summary = quant_ctx.get("summary_text", "")
        if summary:
            lines.append("")
            lines.append(summary)
```

- [ ] **Step 4: Wire `build_quant_context` into `debate_runner.py`**

In `debate_runner.py`, find where `portfolio_state` is assembled before the debate starts. Add:

```python
    # Pre-compute quantitative context for agents
    try:
        from ascent.monitoring.quant_context import build_quant_context
        from ascent.data.store.parquet import ParquetStore
        from ascent.config.settings import get_config

        cfg    = get_config()
        store  = ParquetStore(cfg)
        prices = store.load("prices_live")
        qctx   = build_quant_context(portfolio_state.get("weights", {}), prices)
        portfolio_state["quant_context"] = qctx
        print(f"[Debate] Quant context built — VaR 95: {qctx['portfolio_var_95']:.2%}, "
              f"factor exposures: {list(qctx['factor_exposures'].keys())[:4]}")
    except Exception as e:
        print(f"[Debate] Quant context skipped: {e}")
```

Find the exact location by searching for where `portfolio_state` is constructed in `debate_runner.py`:
```bash
grep -n "portfolio_state\|run_debate\|def run_debate_session" debate/debate_runner.py | head -20
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_plan_d.py::test_build_context_includes_quant_data tests/test_plan_d.py::test_build_context_no_quant_still_works -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add debate/agents.py debate/debate_runner.py tests/test_plan_d.py
git commit -m "feat(debate): inject VaR, factor exposures, and correlations into all agent prompts"
```

---

## Task D3: Extended thinking for the judge

**Problem:** The judge synthesizes 5 agents' arguments into a single verdict. The current prompt uses `temperature=0.3` which limits reasoning depth. Anthropic's extended thinking (`thinking` parameter) allows Claude to reason through a harder problem before outputting — this is exactly what a synthesis task needs.

**The fix:** Add `extended_thinking_completion()` to `client.py`. Use it in `judge.py` with a `thinking_budget` of 3000 tokens. The judge's verdict quality should measurably improve for complex, mixed-signal debates.

**Files:**
- Modify: `ascent/llm/client.py` — add `extended_thinking_completion()`
- Modify: `debate/judge.py` — use extended thinking for synthesis

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_d.py
from unittest.mock import patch, MagicMock

def test_extended_thinking_completion_signature():
    """extended_thinking_completion must exist and accept the right parameters."""
    from ascent.llm.client import extended_thinking_completion
    import inspect
    sig = inspect.signature(extended_thinking_completion)
    params = list(sig.parameters)
    assert "messages" in params
    assert "thinking_budget" in params


def test_judge_uses_extended_thinking_function():
    """judge.py must call extended_thinking_completion, not plain chat_completion."""
    import inspect
    import debate.judge as judge_module

    src = inspect.getsource(judge_module)
    assert "extended_thinking" in src, \
        "judge.py must call extended_thinking_completion for synthesis"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_plan_d.py::test_extended_thinking_completion_signature tests/test_plan_d.py::test_judge_uses_extended_thinking_function -v
```
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Add `extended_thinking_completion` to `client.py`**

Add after the `generate_structured` function:

```python
def extended_thinking_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4000,
    thinking_budget: int = 3000,
) -> str:
    """
    Send a request using Claude's extended thinking mode.
    The model reasons internally for `thinking_budget` tokens before answering.
    Use for high-stakes synthesis tasks (e.g., judge verdict).

    Extended thinking requires temperature=1 (Anthropic API constraint).
    Returns only the final text response (not the thinking blocks).

    Falls back to plain chat_completion if extended thinking fails.
    """
    _check_api_key()
    client = _get_client()

    system_prompt = ""
    filtered_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered_messages.append(m)

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=1,  # required for extended thinking
        thinking={"type": "enabled", "budget_tokens": thinking_budget},
        messages=filtered_messages,
    )
    if system_prompt:
        kwargs["system"] = system_prompt

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(**kwargs)
            # Extract text blocks only (skip thinking blocks)
            text_blocks = [
                block.text for block in response.content
                if hasattr(block, "text")
            ]
            return "\n".join(text_blocks)
        except Exception as e:
            if "thinking" in str(e).lower() or "budget" in str(e).lower():
                # Extended thinking not supported — fall back to plain completion
                log.warning(f"[LLM] Extended thinking not available ({e}), falling back")
                return chat_completion(messages, model=model, max_tokens=max_tokens, temperature=0.3)
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning(f"[LLM] Attempt {attempt+1} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
```

- [ ] **Step 4: Update `judge.py` to use extended thinking**

In `debate/judge.py`, find the call to `generate_structured` or `chat_completion` for the verdict synthesis. Replace it with `extended_thinking_completion`.

First, check the current call:
```bash
grep -n "generate_structured\|chat_completion\|run_judge\|def run_judge" debate/judge.py | head -20
```

Then replace the synthesis call. It should look like this in judge.py:

```python
# Change import at top of debate/judge.py:
from ascent.llm.client import extended_thinking_completion, DEFAULT_MODEL as DEBATE_MODEL, HAIKU_MODEL

# And replace the LLM call inside run_judge() with:
raw = extended_thinking_completion(
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
    model=DEBATE_MODEL,
    max_tokens=4000,
    thinking_budget=3000,
)
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/pytest tests/test_plan_d.py::test_extended_thinking_completion_signature tests/test_plan_d.py::test_judge_uses_extended_thinking_function -v
```
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add ascent/llm/client.py debate/judge.py tests/test_plan_d.py
git commit -m "feat(llm): extended thinking for judge synthesis — deeper reasoning on verdict"
```

---

## Task D4: Prompt caching for system prompts

**Problem:** Each debate session calls 5 agents (4 in round 1, rebuttals, judge). Each call includes a large system prompt. The system prompt is identical across all rebalance days. Anthropic charges full price for every token on every call — prompt caching would cut debate costs by ~80%.

**The fix:** Add a `cache_control` option to `chat_completion()` that marks the system prompt with `"cache_control": {"type": "ephemeral"}`. Anthropic's prompt cache TTL is 5 minutes — within a single debate session (all calls happen in < 2 minutes), all 5 agents share the cached prefix.

**Files:**
- Modify: `ascent/llm/client.py` — support `use_cache` parameter

- [ ] **Step 1: Write failing test**

```python
# Add to tests/test_plan_d.py
def test_chat_completion_accepts_use_cache():
    """chat_completion must accept use_cache parameter."""
    import inspect
    from ascent.llm.client import chat_completion
    sig = inspect.signature(chat_completion)
    assert "use_cache" in sig.parameters, "chat_completion must have use_cache parameter"


def test_generate_structured_accepts_use_cache():
    """generate_structured must accept and forward use_cache parameter."""
    import inspect
    from ascent.llm.client import generate_structured
    sig = inspect.signature(generate_structured)
    assert "use_cache" in sig.parameters, "generate_structured must have use_cache parameter"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_plan_d.py::test_chat_completion_accepts_use_cache tests/test_plan_d.py::test_generate_structured_accepts_use_cache -v
```
Expected: FAIL — `use_cache` parameter doesn't exist.

- [ ] **Step 3: Update `chat_completion` in `client.py` to support caching**

Replace the `chat_completion` function signature and system prompt handling:

```python
def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.7,
    use_cache: bool = False,
) -> str:
    """
    Send a chat completion request to Anthropic.

    Args:
        messages:    List of {"role": "user"/"system"/"assistant", "content": "..."}
        model:       Anthropic model string
        max_tokens:  Max output tokens
        temperature: Sampling temperature
        use_cache:   If True, mark system prompt with cache_control for prompt caching.
                     Use within a session where the same system prompt appears repeatedly
                     (e.g., debate agents — 5+ calls with identical system prompts).

    Returns:
        The assistant's response text.
    """
    _check_api_key()
    client = _get_client()

    system_prompt = ""
    filtered_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            filtered_messages.append(m)

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=filtered_messages,
    )

    if system_prompt:
        if use_cache:
            # Mark system prompt for caching — Anthropic caches for 5 min
            kwargs["system"] = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        else:
            kwargs["system"] = system_prompt

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            log.warning(f"[LLM] Attempt {attempt+1} failed ({e}), retrying in {wait}s")
            time.sleep(wait)
```

Update `generate_structured` to forward `use_cache`:

```python
def generate_structured(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2000,
    temperature: float = 0.4,
    use_cache: bool = False,
) -> str:
    """Convenience wrapper for structured generation tasks."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    return chat_completion(messages, model=model, max_tokens=max_tokens,
                           temperature=temperature, use_cache=use_cache)
```

- [ ] **Step 4: Enable caching in debate agents**

In `debate/agents.py`, add `use_cache=True` to all `generate_structured` calls (bull, bear, devil, regime_specialist):

```python
# Example — run_bull_agent:
return generate_structured(
    system_prompt=(...),
    user_prompt=user_prompt,
    model=DEBATE_MODEL,
    temperature=0.6,
    use_cache=True,   # ← add this line to all 4 agent calls
)
```

The same change goes in `run_bear_agent`, `run_devils_advocate_agent`, and `run_regime_specialist_agent`.

- [ ] **Step 5: Run all Plan D tests**

```bash
.venv/bin/pytest tests/test_plan_d.py -v
```
Expected: All PASS

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/pytest tests/ -v --tb=short 2>&1 | tail -20
```
Expected: All existing tests continue to pass.

- [ ] **Step 7: Smoke test quant context end-to-end**

```bash
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from ascent.monitoring.quant_context import build_quant_context
from ascent.data.store.parquet import ParquetStore
from ascent.config.settings import get_config

cfg    = get_config()
store  = ParquetStore(cfg)
prices = store.load('prices_live')

weights = {'AAPL': 0.07, 'MSFT': 0.07, 'EEM': 0.05, 'GLD': 0.04,
           'TLT': 0.05, 'JPM': 0.06, 'XOM': 0.04, 'NEE': 0.05,
           'MRK': 0.05, 'WMT': 0.05, 'NVDA': 0.07, 'V': 0.05, 'MA': 0.04,
           'AMZN': 0.06, 'CB': 0.03, 'CAT': 0.04, 'EQIX': 0.04,
           'CRWD': 0.04, 'SPGI': 0.03, 'HCA': 0.03, 'AWK': 0.02, 'WELL': 0.02}

ctx = build_quant_context(weights, prices)
print(ctx['summary_text'])
print()
print('VaR 95:', ctx['portfolio_var_95'])
print('VaR 99:', ctx['portfolio_var_99'])
"
```

- [ ] **Step 8: Commit**

```bash
git add ascent/llm/client.py debate/agents.py tests/test_plan_d.py
git commit -m "feat(llm): prompt caching for debate system prompts — ~80% token cost reduction in debate"
```
