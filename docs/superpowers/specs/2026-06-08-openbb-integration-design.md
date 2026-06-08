# OpenBB Integration — Design Spec
**Date**: 2026-06-08
**Scope**: Three-layer OpenBB integration — hub reliability, new alpha data (CBOE options history, CFTC COT, Fama-French factors), AI PM live tools.

---

## Background

Ascent Capital has three data weaknesses OpenBB directly addresses:

1. **yfinance fragility** — `hub.py` uses yfinance with no fallback. Failed symbols drop silently.
2. **Options history gap** — `ascent/data/ingest/options.py` explicitly documents: *"Historical options data is NOT available from yfinance — only today's chain."* The `options_flow` alpha sleeve has been running on a 1-day snapshot panel.
3. **Missing signals** — CFTC Commitments of Traders (macro speculator positioning) and Fama-French factors are completely absent from the system.

OpenBB provides 35 data providers via a unified Python SDK. The relevant free providers: CBOE (options, no key), Fama-French (factors, no key), yfinance (prices, no key), Tiingo (prices, free token), CFTC (COT, free Socrata token), FRED (macro, free key — already have it).

---

## Architecture

Single adapter pattern. All OpenBB interactions go through one file:

```
ascent/integrations/openbb_client.py
        │
        ├── ascent/data/hub.py                       (Layer 1 — price reliability)
        ├── ascent/data/ingest/cboe_options.py        (Layer 2 — historical options)
        ├── ascent/data/ingest/cftc_positioning.py    (Layer 2 — COT positioning)
        ├── ascent/data/ingest/famafrench_factors.py  (Layer 2 — FF factors)
        ├── memory/ticker_memory.py                   (Layer 1 — outcome scoring)
        └── agents/ai_pm_agent.py                     (Layer 3 — AI PM live tools)
```

No OpenBB imports outside `openbb_client.py`. One place to configure providers, credentials, and fallbacks. If OpenBB fails to install or import, every consumer falls back gracefully to its existing behavior.

**Install**:
```bash
pip install openbb openbb-cboe openbb-yfinance openbb-famafrench openbb-cftc openbb-tiingo
```

**New env vars** (all optional — system degrades gracefully without them):
- `TIINGO_TOKEN` — free registration at tiingo.com. Enables price reliability upgrade.
- `CFTC_APP_TOKEN` — free Socrata account. Without it, CFTC requests are rate-limited but functional.

---

## Layer 1: Hub Reliability

### Problem
`hub.py` fetches prices with direct yfinance calls inside a `ThreadPoolExecutor`. Rate limits, stale data, and timeouts drop symbols silently. `memory/ticker_memory.py:_fetch_return()` has the same vulnerability.

### Design

**`openbb_client.py`** exposes:
```python
def fetch_prices(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV for symbols. Tries Tiingo first (if TIINGO_TOKEN set),
    falls back to yfinance per-symbol on failure.
    Returns wide DataFrame: dates × symbols (Close prices), same schema hub expects.
    """

def fetch_return(symbol: str, from_date: str, horizon_days: int) -> float | None:
    """
    Fetch forward return for one symbol from from_date + horizon_days.
    Used by ticker_memory._fetch_return(). Tiingo first, yfinance fallback.
    """
```

**`hub.py` change**: Replace direct `yf.download()` call with `openbb_client.fetch_prices()`. Output is identical — same parquet schema, same `prices_live` cache name. Cache name does not change: Tiingo is live market data, not simulated.

**`memory/ticker_memory.py` change**: Replace `_fetch_return()` body with `openbb_client.fetch_return()`.

**Fallback chain**:
```
TIINGO_TOKEN set → try Tiingo → if fails → yfinance
TIINGO_TOKEN absent → yfinance (same behavior as today)
```

No behavioral change if `TIINGO_TOKEN` is not set. Zero risk to existing pipeline.

---

## Layer 2: New Alpha Data

### 2a. CBOE Historical Options — `ascent/data/ingest/cboe_options.py`

**Cache**: `options_flow` (existing cache, extends it)

**Problem solved**: Current `options.py` fetches only today's chain from yfinance. CBOE provides historical chains. The `options_flow` alpha sleeve needs a panel of at least 21 days to be usable.

**Data fetched** via `obb.derivatives.options.chains(symbol, provider="cboe")`:
- `iv_skew`: IV of closest OTM call (strike ≥ 1.03×spot) minus IV of closest OTM put (strike ≤ 0.97×spot). Positive = call-bid = bullish skew.
- `put_call_ratio`: volume-weighted PCR across all strikes (>1 = put-heavy = bearish hedging)
- `atm_iv`: IV of the strike closest to current spot price
- `iv_rank_52w`: `atm_iv` percentile rank vs trailing 52 weeks of stored `atm_iv` values (0–100). Requires ≥21 days of history to be valid.

**Schema**: One row per (symbol, date). Same columns as current `options.py` output so `feature_defs.py` picks it up without changes.

**Fallback**: If CBOE unavailable for a symbol, falls back to existing yfinance `options.py` logic for that symbol.

**No API key required.**

---

### 2b. CFTC Commitments of Traders — `ascent/data/ingest/cftc_positioning.py`

**Cache**: `cftc_positioning.parquet` (new)

**Data**: Weekly COT report for S&P 500 e-mini futures (most relevant equity futures contract).

**Fields stored**:
- `net_noncommercial_long`: speculator net long contracts (net = longs − shorts)
- `pct_long_noncommercial`: speculators long as % of total open interest
- `change_in_noncommercial_long`: week-over-week change (positioning momentum)
- `net_commercial_long`: commercial (hedger) net positioning (inverse signal)

**Update cadence**: Weekly, published by CFTC every Friday ~3:30 PM ET. Module checks if cache is stale (>7 days) before re-fetching.

**Two integration points**:
1. `run_all_agents.py` ingest: called alongside CBOE/FF in hub run, writes to `cftc_positioning.parquet`
2. AI PM tool `get_cot_positioning` reads from this cache (Layer 3)

**Free Socrata token** (`CFTC_APP_TOKEN`): optional, prevents rate limiting. Works without it.

---

### 2c. Fama-French Factors — `ascent/data/ingest/famafrench_factors.py`

**Cache**: `famafrench_factors.parquet` (new)

**Data**: Daily Fama-French 5 factors + Momentum: `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `Mom`.

**Purpose**: ML sleeve inputs only. Not a new alpha sleeve. Adds 12 new features to the ML feature matrix (21d and 63d rolling OLS loadings of each symbol's returns on each factor). Symbols with higher `HML` loading = more value-exposed; higher `Mom` loading = more momentum-driven. These improve the ML sleeve's ability to distinguish signal from style.

**Integration into `feature_defs.py`**: New `factor_loadings(returns, factor_df, window)` function. Returns per-symbol factor beta DataFrame. Added to `build_features.py` feature list, gated: if `famafrench_factors.parquet` absent, feature columns are omitted (not a pipeline failure).

**No API key required.**

---

## Layer 3: AI PM Live Tools

### Upgrade: `get_macro_data` → live

**Current**: reads `macro_live.parquet` (cached, potentially hours old).

**Upgrade**: `openbb_client.get_live_macro()` tries FRED via OpenBB (truly live), returns same structured response. Falls back to parquet on failure. No schema change. No prompt change. Opus sees fresher data.

```python
def get_live_macro() -> dict[str, float]:
    """
    Returns: {fed_funds_rate, treasury_10y, treasury_2y, yield_spread_10y2y,
               vix, cpi, unemployment, oil_wti, hy_spread, ig_spread}
    Tries live FRED via OpenBB, falls back to cached macro_live.parquet.
    """
```

---

### New Tool: `get_live_options_flow`

**Phase**: Phase 2 only. Never in `PRE_THESIS_TOOLS`.

**Purpose**: Opus checks options market sentiment on AMPLIFY candidates. Options market participants are typically better-informed than equity-only signals (they're paying for optionality). High PCR = market is buying puts = crowd is hedging against the thesis. Call-bid skew = market paying up for upside = confirmation.

**Tool schema**:
```python
{
    "name": "get_live_options_flow",
    "description": (
        "Fetch current options market signals for specific symbols: "
        "put/call ratio (PCR), IV skew direction, and IV rank vs 52-week history. "
        "Use on your top AMPLIFY candidates to check if the options market "
        "confirms or contradicts the thesis. "
        "High PCR (>1.2) = heavy put buying = crowd hedging against your thesis. "
        "Call-bid IV skew = market paying up for upside = thesis confirmation. "
        "IV rank > 80th pct = options expensive = expect-the-unexpected event risk. "
        "Call for 1-4 symbols max. Phase 2 only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbols": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["symbols"]
    }
}
```

**Output format**:
```
OPTIONS FLOW — CAT:
  Put/Call Ratio: 0.72 (call-dominant — market not hedging)
  IV Skew: +3.1% (call-bid — upside being priced in)
  IV Rank: 28th pct vs 52w (cheap options — low event risk priced)
  → CONFIRMS THESIS

OPTIONS FLOW — MRK:
  Put/Call Ratio: 1.41 (heavy put buying — market hedging downside)
  IV Skew: -4.8% (put-bid — crowd protecting against decline)
  IV Rank: 74th pct vs 52w (elevated — event risk in market)
  → CAUTION: options market hedging against this position
```

**Executor**: `_tool_get_live_options_flow(inputs)` calls `openbb_client.get_options_snapshot(symbols)`. If CBOE unavailable for a symbol, falls back to yfinance options chain for that symbol. Returns `"options_unavailable"` only if all sources fail.

---

### New Tool: `get_cot_positioning`

**Phase**: Phase 2 only.

**Purpose**: Macro-level crowding check distinct from the per-name `get_crowding_signal`. `get_crowding_signal` tells Opus if a specific stock is over-owned. `get_cot_positioning` tells Opus if the *entire equity market* is over-positioned by speculators — which matters for portfolio-level tail risk, not name selection. Extreme speculator long at a regime transition = elevated drawdown risk even if individual names look clean.

**Tool schema**:
```python
{
    "name": "get_cot_positioning",
    "description": (
        "Fetch the latest CFTC Commitments of Traders report for S&P 500 e-mini futures. "
        "Returns speculator (non-commercial) net long positioning and its 3-year percentile rank. "
        "Use to check if the broad equity market is over-positioned. "
        "Extreme speculator long (>85th pct) = macro crowding risk even if individual names are clean. "
        "Use once per Phase 2, alongside get_crowding_signal for your specific names. "
        "Does not take inputs."
    ),
    "input_schema": {"type": "object", "properties": {}}
}
```

**Output format**:
```
CFTC S&P 500 E-MINI POSITIONING (as of 2026-06-06):
  Speculator net long: +187,420 contracts
  Week change: +12,340 contracts (positioning increasing)
  Pct long (spec): 63.2% of open interest
  3-year percentile: 71st pct

→ ELEVATED but not extreme. Monitor for further increases.
  (>85th pct = CROWDED LONG — apply macro tail risk discount to full portfolio)
```

**Executor**: `_tool_get_cot_positioning(_)` reads `cftc_positioning.parquet`. If cache is absent or stale (>7 days), attempts live fetch via `openbb_client`. Falls back to `"COT data unavailable — proceed without macro positioning context"`.

---

## Data Flow Summary

```
run_all_agents.py (daily hub run)
  → openbb_client.fetch_prices()         → prices_live.parquet
  → cboe_options.py                       → options_flow.parquet
  → cftc_positioning.py                   → cftc_positioning.parquet
  → famafrench_factors.py                 → famafrench_factors.parquet

ascent/main.py (per-agent pipeline)
  → feature_defs.py reads options_flow    → iv_skew, put_call_ratio features
  → feature_defs.py reads famafrench      → factor loading features (ML sleeve)

agents/ai_pm_agent.py Phase 2 tool loop
  → get_macro_data (upgraded)             → live FRED via OpenBB
  → get_live_options_flow                 → live CBOE options snapshot
  → get_cot_positioning                   → cftc_positioning.parquet

memory/ticker_memory.py
  → _fetch_return()                       → openbb_client.fetch_return()
```

---

## File Map

| File | Action | Notes |
|---|---|---|
| `ascent/integrations/openbb_client.py` | **Create** | Central adapter — all OpenBB calls go here |
| `ascent/data/hub.py` | **Modify** | Replace yfinance fetch with `openbb_client.fetch_prices()` |
| `memory/ticker_memory.py` | **Modify** | Replace `_fetch_return` body with `openbb_client.fetch_return()` |
| `ascent/data/ingest/cboe_options.py` | **Create** | Historical options chains |
| `ascent/data/ingest/cftc_positioning.py` | **Create** | CFTC COT weekly report |
| `ascent/data/ingest/famafrench_factors.py` | **Create** | FF 5-factor + momentum returns |
| `ascent/features/feature_defs.py` | **Modify** | Add `factor_loadings()`, gate on cache presence |
| `agents/ai_pm_agent.py` | **Modify** | Upgrade `get_macro_data`, add 2 new tools + executors |
| `run_all_agents.py` | **Modify** | Call CBOE/CFTC/FF ingest in hub run block |
| `tests/integrations/test_openbb_client.py` | **Create** | Mocked tests for adapter |
| `tests/data/test_new_ingest.py` | **Create** | Mocked tests for 3 new ingest modules |

---

## Integrity Constraints

- `prices_live` cache name unchanged — Tiingo is live market data, not simulated. Constraint is about simulated/GBM data, not provider identity.
- CBOE options writes to existing `options_flow` cache name — same schema, additive history.
- `cftc_positioning.parquet` and `famafrench_factors.parquet` are new cache names — no collision risk.
- All new ingest modules are gated: absent cache → feature columns omitted, never pipeline failure.
- AI PM tools follow Phase 2 only rule established by `get_mirofish_sentiment`. Neither new tool appears in `PRE_THESIS_TOOLS`.
- No new alpha sleeves — Fama-French factors feed ML sleeve feature matrix only.

---

## What This Does Not Do

- Does not replace `sec_filings.py` or `capitol_trades.py` — both have real implementations already.
- Does not add FMP, Intrinio, Benzinga, or Biztoc (paid providers).
- Does not add a Reddit signal (permanently disabled per project memory).
- Does not add a new alpha sleeve — all new data feeds existing sleeves.
- Does not run `openbb-api` server — uses Python SDK directly.
