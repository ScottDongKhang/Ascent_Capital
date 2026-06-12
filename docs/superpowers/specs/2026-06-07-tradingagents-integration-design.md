# TradingAgents Integration — Design Spec
**Date**: 2026-06-07  
**Scope**: Three additive improvements to AI nativism and alpha generation, derived from TradingAgents repo analysis.

---

## Background

TradingAgents (open-source multi-agent trading framework) was reviewed for improvements to Ascent Capital. The existing Ascent system already covers TradingAgents' deferred reflection pattern at the rebalance/verdict level (`memory/reflection_agent.py`, `ai_pm_learning.py`). Three genuine gaps were identified.

---

## Feature 1: Per-Ticker AI PM Outcome Memory

### Problem
The AI PM has no stock-level institutional memory. Reflections in `memory/reflection_agent.py` are per-verdict (rebalance-level). The AI PM doesn't know "last time I overweighted CAT it faded at 21d" or "I've amplified MRK twice and both times worked." TradingAgents injects "last 5 decisions on this ticker" into every Portfolio Manager call.

### Design

**New file**: `memory/ticker_memory.py`

```python
record_decision(
    symbol: str,
    date: str,          # ISO date of rebalance
    ai_w: float,        # AI PM's proposed weight
    quant_w: float,     # quant baseline weight
    decision_type: str, # amplify | reduce | new | exit
    rationale_snippet: str,  # first 200 chars of AI PM rationale
) -> None
# Appends to memory/ticker_memory.jsonl

score_outcomes(today: date) -> int
# For each unscored entry 10+ days old: fetch yfinance return,
# compute incremental_alpha = (ai_w - quant_w) * return,
# classify verdict (win/miss/fade/early), write back to jsonl.
# Returns count of newly scored entries.

get_ticker_context(symbol: str, n: int = 3) -> str
# Returns formatted block of last N AI PM decisions on this ticker with outcomes.
# Empty string if no history. Example:
#   "AI PM HISTORY — CAT (last 3 calls):
#    2026-05-20 amplify 8.0%→10.0%: +0.41% alpha at 21d [WIN] — thesis: momentum + regime tailwind
#    2026-04-10 amplify 7.0%→9.0%: -0.22% alpha at 21d [MISS] — thesis: capex cycle
#    ..."

get_cross_ticker_lessons(n: int = 3) -> str
# Returns last N scored decisions across any ticker for cross-asset context.
```

**Storage**: `memory/ticker_memory.jsonl` — append-only, one JSON per line. Fields: `symbol`, `date`, `ai_w`, `quant_w`, `type`, `rationale_snippet`, `outcome_10d`, `outcome_21d`, `verdict`, `scored`.

**Integration points**:
1. `agents/ai_pm_agent.py` — `_tool_propose_portfolio()`: when processing a symbol override, call `get_ticker_context(symbol)` and prepend to the per-symbol rationale prompt.
2. `agents/ai_pm_agent.py` — `_tool_propose_portfolio()`: after finalizing overrides, call `record_decision()` for each override applied.
3. `run_all_agents.py` — daily (non-rebalance) path: call `score_outcomes(today)` immediately after the existing `compute_feedback()` call in `_run_daily_learning()`.

**Cost**: Zero LLM cost. Pure Python + yfinance.

---

## Feature 2: Instrument Identity in Data Grounding

### Problem
`_build_data_grounding()` in `agents/ai_pm_agent.py` injects price momentum (21d/63d/252d returns + alpha_score) but no company identity. An LLM can still hallucinate company names, sectors, or industries when reasoning about a ticker. TradingAgents resolves company name/sector/industry/exchange from yfinance and injects into every agent prompt.

### Design

Modify `_build_data_grounding()` to prepend an identity line per symbol:

```
CAT: [Caterpillar Inc | Industrials | Construction & Mining Equip] 21d=+4.2% | 63d=+11.3% | ...
MRK: [Merck & Co Inc  | Health Care  | Pharmaceuticals           ] 21d=-1.1% | 63d=+3.4%  | ...
```

**Source**: Load `data_cache/profiles.parquet` (already used by stat-arb and portfolio construction). Fields: `symbol`, `name`, `sector`, `industry`. Fall back to ticker-only if profiles unavailable.

**Change scope**: Single function, ~15 lines added. No new files.

---

## Feature 3: StockTwits Grounded Sentiment

### Problem
`ascent/integrations/` is empty. Ascent has no crowd sentiment signal. StockTwits public API returns user-labeled Bullish/Bearish tag counts per ticker — zero hallucination risk because counts are pre-fetched before the LLM sees them. This is a validated alpha source in TradingAgents (users voluntarily label their own posts as Bullish/Bearish, creating a clean signal).

### Design

**New file**: `ascent/integrations/stocktwits.py`

```python
get_sentiment(symbols: list[str], max_messages: int = 30) -> dict[str, dict]
# Hits StockTwits public endpoint (no auth required):
# GET https://api.stocktwits.com/api/2/streams/symbol/{TICKER}.json?limit=30
# Counts user-labeled {"sentiment": {"basic": "Bullish"}} and {"sentiment": {"basic": "Bearish"}} tags.
# Returns:
# {
#   "CAT": {"bullish": 18, "bearish": 5, "n_labeled": 23, "n_total": 30,
#           "ratio": 0.78, "band": "bullish", "stale": False},
#   "MRK": {"bullish": 3, "bearish": 14, ...}
# }
# band: "strongly_bullish" (>0.75), "bullish" (>0.55), "neutral" (0.45–0.55),
#        "bearish" (<0.45), "strongly_bearish" (<0.25)
# stale: True if <5 labeled messages in last 30 (low signal, don't rely on it)
```

**Integration**: Inject into AI PM pre-thesis prompt as a `CROWD SENTIMENT` block alongside `_build_data_grounding()`. Pre-fetched before the LLM call — no tool invocation inside the prompt. Example injection:
```
══ CROWD SENTIMENT (StockTwits, last 30 messages, user-labeled) ══
  CAT: 78% bullish (23 labeled / 30 total) — band: bullish
  MRK: 18% bullish (17 labeled / 30 total) — band: strongly_bearish [DIVERGENCE: quant +momentum]
  ...
══════════════════════════════════════════════════════════════════
```

**Rate limiting**: StockTwits free tier allows ~200 req/hour. With 15-symbol universe, one call per symbol = 15 req per run. Well within limits. Add 0.2s delay between requests.

**IC validation**: `get_sentiment()` returns a dict; caller logs `{symbol, date, sentiment_ratio}` to `logs/stocktwits_ic.jsonl`. After 30+ runs, manual IC computation via `scripts/compute_stocktwits_ic.py` (not yet built). No auto-disable in this phase — disable manually if IC-t goes negative after inspection.

**No alpha sleeve yet**: Wire as context injection only. Quant alpha sleeve requires positive IC validation first.

---

## Implementation Order

1. Feature 2 (instrument identity) — 15 min, zero risk, immediate improvement.
2. Feature 1 (ticker memory) — 2–3 hours, highest AI nativism value.
3. Feature 3 (StockTwits) — 1–2 hours, new signal source.

## Files Touched

| File | Change |
|------|--------|
| `memory/ticker_memory.py` | New |
| `agents/ai_pm_agent.py` | `_build_data_grounding()` patch + ticker memory calls |
| `ascent/integrations/stocktwits.py` | New |
| `run_all_agents.py` | Add `score_outcomes()` to daily path |
| `tests/memory/test_ticker_memory.py` | New |
| `tests/integrations/test_stocktwits.py` | New |

## Non-Goals

- No changes to walk-forward runner, regime engine, or portfolio construction.
- StockTwits is context injection only until IC is validated — no weight in alpha stack.
- No checkpoint/resume system (LangGraph dependency not worth the complexity).
- No Reddit integration (permanently disabled per project memory).
