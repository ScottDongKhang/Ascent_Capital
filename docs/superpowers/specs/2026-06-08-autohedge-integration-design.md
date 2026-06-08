# AutoHedge Integration — Design Spec
**Date**: 2026-06-08  
**Scope**: Three additive improvements to Ascent Capital derived from AutoHedge repo analysis — real-time news awareness, structured financial grounding, and dynamic ticker discovery with mini-rebalance capability.

---

## Background

AutoHedge (open-source autonomous agent hedge fund) was reviewed for patterns transferable to Ascent Capital. The existing Ascent system already covers AutoHedge's multi-agent pipeline, risk gating, and execution layers. Three genuine gaps were identified that AutoHedge addresses with patterns directly applicable to Ascent's architecture.

**API cost: zero.** All three features use either Exa's free tier (≤30 searches/month at current scale), yfinance (already installed), or existing Claude usage.

---

## Feature 1: Exa Live News Layer

### Problem
The AI PM pre-thesis runs blind to news. It reads quant signals (momentum, alpha scores, regime) but has no awareness of real-time catalysts — earnings surprises, FDA decisions, analyst upgrades, macro events. A catalyst that moves a stock 5% pre-market is invisible until the next price bar is ingested.

### Design

**New file**: `ascent/integrations/exa_news.py`

```python
fetch_news(
    symbols: list[str],
    max_per_symbol: int = 2,
) -> dict[str, list[str]]
# Query: f"{symbol} stock news catalyst today" per symbol
# Extracts Exa structured summary.answer field (same pattern as AutoHedge)
# 0.2s delay between requests — respects free tier rate limit
# On failure for any symbol: returns [] for that symbol, logs warning, continues
# Returns: {"CAT": ["headline 1", "headline 2"], "MRK": [...], ...}
```

**Rebalance-day injection** — added to `_build_data_grounding()` in `agents/ai_pm_agent.py`:

```
══ LIVE NEWS (Exa, fetched today) ══════════════════════════════
  CAT: [1] Caterpillar beats Q1 estimates, raises FY guidance
       [2] Infrastructure bill spending accelerating through 2027
  MRK: [1] FDA issues complete response letter on Keytruda expansion
  NEE: [] (no news fetched)
════════════════════════════════════════════════════════════════
```

Pre-fetched before the LLM call — not a tool the AI PM can invoke. The LLM sees news as data, eliminating hallucination risk from self-directed search.

**Non-rebalance-day**: same `fetch_news()` call runs in the daily path of `run_all_agents.py` to feed Feature 3 (ticker discovery). No additional Exa cost — same call, different consumer.

**Rate limit math**: 15 symbols × 2 results × ~2 rebalances/month + ~16 non-rebalance daily runs = ~60 Exa calls/month. Free tier allows 1,000/month. Well within limits.

---

## Feature 2: Structured Financials via yfinance

### Problem
`_build_data_grounding()` injects price momentum and alpha scores but no financial health signal. The AI PM can hallucinate "MRK has a strong balance sheet" when debt/equity is actually elevated. The failed fundamental sleeve (IC-t = −4.75) used LLM-estimated fundamentals — that failure was about ranking stocks on LLM priors, not about structured financial data being bad.

### Design

**New helper inside `agents/ai_pm_agent.py`**:

```python
def _fetch_financials(symbols: list[str]) -> dict[str, dict]:
    # yf.Ticker(sym).quarterly_balance_sheet + .quarterly_income_stmt
    # Extracts 4 metrics only:
    #   current_ratio     = current_assets / current_liabilities
    #   debt_to_equity    = total_debt / stockholders_equity
    #   revenue_growth_yoy = (latest_quarter_rev - year_ago_rev) / year_ago_rev
    #   gross_margin      = gross_profit / total_revenue (latest quarter)
    # On yfinance 429 / missing data: returns {} for that symbol, continues
    # Cached to data_cache/financials_cache.json with 24h TTL
    # (quarterly data — no need to re-fetch intraday)
    # If cache is expired AND yfinance fails: skip financials block silently,
    #   log warning, do not fail the run
```

**Injected into `_build_data_grounding()`** alongside momentum and news:

```
══ FUNDAMENTALS (yfinance quarterly, cached 24h) ════════════════
  CAT: curr_ratio=1.4 | D/E=1.8 | rev_growth=+12% | margin=21%
  MRK: curr_ratio=1.1 | D/E=0.9 | rev_growth=-3%  | margin=68%
  NEE: curr_ratio=0.6 | D/E=1.1 | rev_growth=+8%  | margin=31%
════════════════════════════════════════════════════════════════
```

**Why 4 metrics:** Raw balance sheet JSON would exceed context budget and confuse the LLM with accounting line items. Four ratios cover the key dimensions (liquidity, leverage, growth, profitability) without noise.

**Integrity constraint preserved**: This is context injection only. The fundamental alpha sleeve (`ascent/alpha/fundamental.py`) stays disabled. No change to `DEFAULT_ALPHA_WEIGHTS`. The CLAUDE.md constraint — "do not re-enable without positive IC-t" — is not touched.

---

## Feature 3: Ticker Discovery + Mini-Rebalance

### Problem
Ascent's universe is static between rebalances. A compelling new name that emerges from a real news catalyst on a Tuesday is invisible until the next scheduled rebalance (up to 10 business days later). AutoHedge's Director agent dynamically discovers tickers from task context — this pattern, adapted to Ascent's earned-authority model, enables real-time universe expansion gated by existing safety layers.

### Design

**New file**: `ascent/strategy/ticker_discovery.py`

```python
@dataclass
class DiscoveryResult:
    symbol: str
    conviction_score: float   # 0–1, Haiku-scored
    catalyst_snippet: str     # first 200 chars of rationale
    rationale: str

def run_discovery(
    news_context: dict[str, list[str]],   # output of fetch_news()
    existing_universe: list[str],
) -> DiscoveryResult | None
# Uses HAIKU_MODEL (classifier task — cheap, fast)
# Prompt: given these Exa news items from current holdings, name ONE ticker
#         not in existing_universe that appears in or is directly related
#         to themes in this news. Score conviction 0–1.
# Grounds candidate in fetched news — no hallucination risk (ticker must
#   appear in or derive from real Exa text, e.g. CAT news cites VMC)
# Returns None if conviction < 0.75 or no candidate surfaced
```

**Mini-rebalance trigger** (added to non-rebalance daily path in `run_all_agents.py`):

```python
result = run_discovery(news_context, current_universe)
if result and result.conviction_score >= 0.75:
    if not _check_mini_rebalance_cooldown():
        _trigger_mini_rebalance(result)
```

**Cooldown**: `data_cache/last_mini_rebalance.json` — blocks if last mini-rebalance was < 5 trading days ago. Prevents churn.

**`_trigger_mini_rebalance(result: DiscoveryResult)`**:
1. Assembles `extended_universe = current_universe + [result.symbol]` in `run_all_agents.py` and passes it as an explicit `symbols` override to `us_equities_agent.run()` — no config file modified, no `UniverseConfig` mutation
2. Runs full `us_equities_agent` pipeline with `extended_universe` (identical to a scheduled rebalance)
3. Portfolio construction includes or excludes the candidate based on alpha rank — no special casing
4. Debate layer runs on the resulting portfolio (unchanged)
5. Approval threshold: **1% NAV** (stricter than the normal 2% — mini-rebalances are higher scrutiny)
6. On `proceed` / `reduce_size`: executes via `eod_runner.py` as normal
7. On `halt_and_review`: logs and exits without execution
8. Writes `data_cache/last_mini_rebalance.json`:
   ```json
   {"date": "2026-06-08", "symbol": "VMC", "conviction": 0.82}
   ```
9. Appends to `logs/eod_log.jsonl` with `"trigger": "discovery"` tag for attribution

**What does NOT change**: kill switch thresholds, debate gate, earned-authority cap, slippage tracking, sector constraints, max-weight hard cap. The mini-rebalance is the existing execution path fired by a new condition — not a new execution path.

---

## Implementation Order

1. **Feature 2** (structured financials) — ~30 min, single function, zero risk. Immediate grounding improvement.
2. **Feature 1** (Exa news) — ~1 hour, new file + `_build_data_grounding()` patch. Requires `EXA_API_KEY` in env.
3. **Feature 3** (ticker discovery + mini-rebalance) — ~3 hours, new file + `run_all_agents.py` daily path. Depends on Feature 1 output.

---

## Files Touched

| File | Change |
|------|--------|
| `ascent/integrations/exa_news.py` | New |
| `ascent/strategy/ticker_discovery.py` | New |
| `agents/ai_pm_agent.py` | `_build_data_grounding()` — add news block + financials block; `_fetch_financials()` helper |
| `run_all_agents.py` | Daily path — `fetch_news()` call + discovery trigger + mini-rebalance |
| `data_cache/financials_cache.json` | Runtime cache (24h TTL, auto-created) |
| `data_cache/last_mini_rebalance.json` | Cooldown state (auto-created on first mini-rebalance) |
| `tests/integrations/test_exa_news.py` | New |
| `tests/strategy/test_ticker_discovery.py` | New |

---

## Non-Goals

- No changes to walk-forward runner, regime engine, or portfolio construction logic.
- No changes to the fundamental alpha sleeve — stays disabled.
- No Polygon/Massive API dependency — yfinance covers the same data for free.
- No Reddit integration (permanently disabled per project memory).
- Ticker discovery does not write to `UniverseConfig` — universe expansion is in-memory and per-run only.
- No multi-ticker discovery per run — one candidate maximum per daily scan (anti-churn).
