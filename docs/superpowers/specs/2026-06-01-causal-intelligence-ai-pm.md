# Causal Intelligence for the AI PM
**Date:** 2026-06-01  
**Status:** Approved for implementation  
**Goal:** Give the AI PM a structural causal model of the economy and each holding so it reasons about *why* trades work — not just *what* correlates — and earns authority faster by making higher-IC, regime-compatible bets.

---

## 1. Problem Statement

The AI PM is in Phase 0 (0% weight), shadow period started 2026-05-19. After 8 shadow days it is -3.45% vs quant. Root cause (diagnosed 2026-05-28): systematic anti-momentum bias. The AI PM applies DCF/valuation logic in a momentum regime, makes 5 correlated overrides per rebalance, and holds losing positions through the full horizon even when the thesis mechanism breaks.

The causal system fixes these failure modes structurally — not by lowering the Sharpe-edge threshold for authority advancement, but by making the AI PM's bets sharper, more independent, better-timed, and faster to exit when wrong.

---

## 2. Architecture Overview

Two-layer causal system:

```
Layer 1 — Macro Causal DAG (statistical, weekly)
  causallearn PC algorithm on 2-year FRED + sector returns
  → data_cache/macro_causal_dag.json
  → drives regime-causal compatibility gate

Layer 2 — Company Causal Graphs (LLM, quarterly per symbol)
  Haiku builds per-holding DAGs cached by (symbol, quarter_end)
  → data_cache/causal_graphs/{symbol}_{quarter_end}.json
  → read by AI PM Phase 1 + Phase 2, never rebuilt mid-quarter

Falsification Tracker (pure Python, weekly)
  reads logs/causal_predictions.jsonl
  checks realized data against stated conditions
  → accuracy score fed back into Phase 2 context
  → intra-horizon early exit flags on non-rebalance daily path
```

---

## 3. The Four Gates (Earn Authority Faster)

These four mechanisms directly improve AI PM shadow returns without changing the advancement criteria.

### Gate 1 — Regime-Causal Compatibility

Before any mechanism enters the pre-thesis, it is checked against the current regime posture derived from `macro_causal_dag.json`.

- `calm_bull`: momentum-compatible mechanisms pass (catalyst-driven rerating, supply inflection, margin recovery with evidence of inflection). Anti-momentum mechanisms (valuation compression, DCF mean reversion) are **dropped** unless `crowding = OVERCROWDED`.
- `stressed`: quality/defensive mechanisms pass. Cyclical expansion mechanisms are flagged.
- `crisis`: only macro-hedge and capital-preservation mechanisms pass.

This gate runs in Phase 1 pre-thesis before `propose_prethesis` is called. Incompatible mechanisms are excluded from the thesis — the AI PM cannot propose a DCF-style override in a momentum regime.

`compatibility.py` classifies mechanism compatibility using a keyword + Haiku-assigned `mechanism_type` field on the causal graph (set at build time). Types: `momentum_catalyst`, `quality_defensive`, `macro_hedge`, `mean_reversion`, `valuation`, `supply_demand_inflection`. The regime-to-allowed-types mapping is a static dict in `compatibility.py` — no LLM call at check time.

### Gate 2 — Priced-In Filter

Each `CausalMechanism` carries a `timing` field assigned by Haiku at graph build time:

| Value | Meaning | Action |
|-------|---------|--------|
| `priced_in` | mechanism already reflected in price | exclude from concentration candidates |
| `not_yet_priced` | mechanism valid, catalyst not yet in price | standard concentration candidate |
| `catalyst_imminent` | trigger expected within 21 trading days | highest priority concentration |

Haiku determines `timing` by comparing the mechanism's expected outcome variable against current price action and analyst consensus. Only `not_yet_priced` and `catalyst_imminent` mechanisms justify AI PM overweighting. `priced_in` mechanisms are logged but excluded — quant momentum handles them.

### Gate 3 — Mechanism Velocity Score

Pure Python, no LLM. For each mechanism, a velocity score `v ∈ [0, 1]` measures how fast the causal trigger is progressing toward its threshold.

```python
velocity = (current_value - baseline_value) / (threshold_value - baseline_value)
# clamped to [0, 1]
```

Example: NAND recovery mechanism requires +20% from trough. Current: +12% from trough → velocity = 0.60.

Velocity is computed from existing price data, FRED, and earnings ratios. No new data sources required.

The pre-thesis receives a ranked list of holdings sorted by `(timing_priority, velocity)` descending. The AI PM is given its best bets in ranked order before writing the thesis — not searching blind.

### Gate 4 — Intra-Horizon Early Exit

Falsification checking currently happens at horizon end. The tracker now runs weekly and raises early-exit flags when a falsification condition triggers mid-horizon.

- Tracker writes `early_exit: true` to the prediction record in `causal_predictions.jsonl`
- `run_all_agents.py` non-rebalance daily path reads these flags and adjusts the AI PM's shadow position to zero for that symbol
- The shadow position is marked as `causal_mechanism_broken` in `ai_pm_shadow_returns.jsonl`

This prevents holding a broken thesis for the full horizon — the AI PM cuts losers when the mechanism breaks, not when the price catches up.

---

## 4. Data Flow

### Sunday Weekend Pipeline (new jobs)

```
1. causal_discovery_runner.py
   - Pulls 2-year FRED (fed_rate, unemployment, credit_spreads, vix) + sector ETF returns
   - Runs PC algorithm (causallearn) with significance α=0.05
   - Writes macro_causal_dag.json: {nodes, edges: [{from, to, strength}]}
   - Runtime: ~30s, pure Python

2. dag_builder.py (for each current holding)
   - Check cache: if (symbol, quarter_end) exists → skip
   - Pull: latest 10-K summary, earnings call summary, key ratios from data_cache
   - Haiku call: ~500 tokens in, ~300 tokens out
   - Writes data_cache/causal_graphs/{symbol}_{quarter_end}.json
   - Cost: ~$0.001/symbol, ~15 symbols → ~$0.015/week
```

### Phase 1 — Pre-Thesis (Sonnet, unchanged call)

```
Existing: reads macro, SEC, earnings, narratives → propose_prethesis

New additions to existing call:
  - Input: mechanism_velocity_scores (Python-computed, injected as context)
  - Input: regime_posture from macro_causal_dag.json (injected as context)
  - Input: cached causal graphs for current holdings
  - Output schema gains: causal_mechanisms: list[CausalMechanism]
  - Gate 1 (regime-causal compatibility) runs before propose_prethesis
  - Gate 2 (priced-in filter) applied to candidate list before ranking

No extra LLM call. Additions are context injections and schema fields.
```

### Phase 2 — Synthesis (Opus, unchanged call)

```
Existing: receives sealed prethesis + quant validation → final weights

New additions to existing call:
  - Input: causal_track_record: {total, confirmed, falsified, accuracy_pct}
  - Input: causal_mechanisms from sealed prethesis (already present in AIPreThesis)
  - Logic: quant bullish + AI causal mechanism compatible + timing=catalyst_imminent → concentrate to 9-10%
  - Logic: quant bullish + AI mechanism priced_in → hold quant weight, no concentration
  - Logic: quant bullish + AI mechanism regime-incompatible → hold quant weight, log disagreement

No extra LLM call. Additions are context injections and synthesis logic in prompt.
```

### Daily Non-Rebalance Path (new check)

```
run_all_agents.py non-rebalance branch gains:
  - causal_tracker.check_early_exits() → list of symbols with broken mechanisms
  - For each: set ai_pm_shadow_weight[symbol] = 0, log as causal_mechanism_broken
  - No LLM call
```

### Devil's Advocate (existing debate call, prompt addition)

```
debate/agents.py devil's advocate prompt gains causal_mechanisms context.
Devil's advocate can now attack the mechanism: "Samsung announced capacity additions —
your NAND recovery mechanism is broken before it runs."
No extra call — prompt append to existing Sonnet debate call.
```

---

## 5. New Files

```
ascent/causal/
  __init__.py
  dag_builder.py          # Haiku-driven per-symbol graph builder; cache logic
  causal_discovery.py     # PC algorithm on FRED + sector returns → macro DAG
  velocity.py             # mechanism_velocity_score() — pure Python
  tracker.py              # reads causal_predictions.jsonl; check_outcomes(); check_early_exits()
  compatibility.py        # regime_compatible(mechanism, regime, macro_dag) → bool

data_cache/
  causal_graphs/          # {symbol}_{quarter_end}.json
  macro_causal_dag.json   # overwritten weekly

logs/
  causal_predictions.jsonl  # {symbol, mechanism, falsification_condition, horizon_days,
                            #  rebalance_date, timing, velocity, outcome, early_exit}
```

---

## 6. Modified Files

| File | Change |
|------|--------|
| `ascent/config/types.py` | Add `CausalMechanism` dataclass; add `causal_mechanisms` field to `AIPreThesis` |
| `agents/ai_pm_agent.py` | Phase 1: inject velocity + regime posture + graphs; add `get_causal_graph(symbol)` tool; add gate 1 + gate 2 pre-thesis logic; Phase 2: inject track record |
| `ascent/monitoring/weekend_runner.py` | Add `causal_discovery` + `causal_graph_builder` jobs |
| `run_all_agents.py` | Non-rebalance path: call `causal_tracker.check_early_exits()`; pass `causal_track_record` to Phase 2 |
| `debate/agents.py` | Append `causal_mechanisms` context to devil's advocate prompt |

---

## 7. Data Structures

### `CausalMechanism`
```python
@dataclass
class CausalMechanism:
    symbol: str
    mechanism: str           # one sentence: "X causes Y via Z"
    intervention: str        # "IF [observable trigger] THEN [expected outcome]"
    falsification_condition: str  # "IF [observable] < [threshold], thesis broken"
    horizon_days: int        # trading days until falsification check
    timing: str              # "priced_in" | "not_yet_priced" | "catalyst_imminent"
    velocity: float          # 0.0–1.0, Python-computed
    regime_compatible: bool  # gate 1 result
```

### `macro_causal_dag.json`
```json
{
  "as_of": "2026-06-01",
  "regime": "calm_bull",
  "nodes": ["fed_rate", "credit_spreads", "vix", "xly_return", "xlf_return", ...],
  "edges": [
    {"from": "fed_rate", "to": "credit_spreads", "strength": "strong", "direction": "positive"},
    {"from": "credit_spreads", "to": "xlf_return", "strength": "moderate", "direction": "negative"}
  ],
  "active_transmission_chains": [
    "fed_rate → credit_spreads → xlf_return",
    "vix → xly_return"
  ]
}
```

### `causal_predictions.jsonl` (one record per position per rebalance)
```json
{
  "symbol": "WDC",
  "mechanism": "NAND oversupply correction → margin expansion → EPS rerating",
  "intervention": "IF NAND spot price +15% from trough THEN WDC gross margin > 40%",
  "falsification_condition": "IF WDC Q3 gross margin < 38%, thesis broken",
  "horizon_days": 63,
  "rebalance_date": "2026-06-15",
  "timing": "catalyst_imminent",
  "velocity": 0.72,
  "regime_compatible": true,
  "outcome": "pending",
  "early_exit": false,
  "checked_date": null
}
```

---

## 8. Cost Analysis

| Component | Model | Frequency | Cost |
|-----------|-------|-----------|------|
| Causal mechanisms in pre-thesis | Sonnet (existing) | 26x/yr | +$0.00 |
| Company graph build | Haiku | Per symbol per quarter | ~$0.06/yr |
| Macro DAG discovery | causallearn (Python) | Weekly | $0.00 |
| Velocity scores | Pure Python | Daily | $0.00 |
| Falsification tracking | Pure Python | Weekly | $0.00 |
| Track record in Phase 2 | Opus (existing) | 26x/yr | +$0.00 |
| Devil's advocate causal context | Sonnet (existing) | 26x/yr | +$0.00 |

**Total incremental cost: ~$0.06/year.**

---

## 9. Testing Strategy

| Test file | What it covers |
|-----------|---------------|
| `tests/test_causal_dag_builder.py` | Mock Haiku response; assert graph schema valid; assert cache hit skips LLM call; assert quarterly cache key correct |
| `tests/test_causal_discovery.py` | PC algorithm on synthetic FRED-shaped data; assert DAG has edges; assert no cycles; assert JSON output schema |
| `tests/test_causal_tracker.py` | Synthetic predictions jsonl; advance mock date past horizon; assert outcome classified; assert early_exit flag raised on falsification trigger |
| `tests/test_causal_velocity.py` | Known baseline/current/threshold values; assert velocity in [0,1]; assert catalyst_imminent at velocity > 0.80 |
| `tests/test_causal_compatibility.py` | Valuation mechanism + calm_bull regime → incompatible; momentum mechanism + calm_bull → compatible; crisis regime blocks all non-defensive |
| `tests/test_ai_pm_prethesis_causal.py` | Assert `causal_mechanisms` field present in `AIPreThesis`; assert all mechanisms pass gate 1 before propose_prethesis; assert priced_in mechanisms excluded from concentration list |

All existing 675 tests must remain green. All schema changes are additive.

---

## 10. Implementation Phases

**Phase A — Foundation (causal infrastructure, no AI PM changes yet)**
- `ascent/causal/` module skeleton
- `causal_discovery.py` + macro DAG weekly job
- `dag_builder.py` + Haiku cache logic
- `velocity.py`
- `CausalMechanism` datatype + `AIPreThesis` schema field
- Tests for all of the above

**Phase B — Gate 1 + Gate 2 (pre-thesis filtering)**
- `compatibility.py` + gate 1 logic in Phase 1
- Priced-in filter (gate 2) in pre-thesis candidate ranking
- Velocity-ranked candidate list injected into Phase 1
- Tests for gates

**Phase C — Phase 2 integration + falsification tracker**
- `tracker.py` + `causal_predictions.jsonl` write on rebalance
- Causal track record injected into Phase 2 synthesis
- Weekly falsification outcome checks
- Gate 4 early exit on non-rebalance daily path
- Tests for tracker + early exit

**Phase D — Debate integration**
- Devil's advocate prompt gains causal mechanism context
- Tests for debate agent causal attack

---

## 11. Success Criteria

- AI PM shadow return vs quant: positive edge within 42 trading days of deployment (Phase A–C complete)
- Causal mechanism falsification rate: < 40% (i.e., AI PM's causal theses are correct > 60% of the time)
- No regression in existing 675 tests
- Zero additional LLM cost vs current spend (all additions are context injections to existing calls, except Haiku graph builds at ~$0.06/yr)
- AI PM advances to Phase 1 (ai_weight=0.25) on earned merit within the existing 21-trading-day window
