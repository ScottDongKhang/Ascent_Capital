# Proof Audit Extension — Design Spec

**Sub-project 1b** — closes the measurement gap left by sub-project 1 (proof audit, merged
2026-08-12) before sub-project 2 (target architecture) starts. Sub-project 1's real-data run
scored only 10 of 23 components; 13 stayed `INSUFFICIENT_DATA` because the CLI never loaded
the optional data sources several sleeves need, and fed all three non-US-equities agents the
wrong price universe. `INSUFFICIENT_DATA` means unmeasured, not proven negative — a target
architecture built on that scorecard as-is would cut components on absence of evidence rather
than evidence of absence. This closes that gap so sub-project 2 has a real verdict for as many
of the 23 components as the data on disk allows.

**Context established by research (2026-08-12):** all data this extension needs already exists
on disk, standalone, with zero live API calls or credentials required:

- `data_cache/fundamentals.parquet`, `earnings.parquet`, `analyst_revisions.parquet`,
  `options_flow.parquet`, `insider_transactions.parquet`, `short_interest.parquet` — the six
  `FeatureBuilder` optional inputs, all populated.
- `data_cache/prices_macro.parquet`, `prices_international.parquet`,
  `prices_alternatives.parquet` — the three specialist agents' real universes, all populated.
- `altdata_alpha` and `earnings_tone_alpha` already self-load from their own parquet caches
  independent of `FeatureBuilder` — they were never blocked by missing `FeatureBuilder` inputs,
  only by the run wiring never reaching them with valid data in the same run.

## What changes

**1. `scripts/run_proof_audit.py` / `run.py`'s `__main__` block: load the six optional
`FeatureBuilder` inputs.**

```python
fundamentals_df = load_parquet("fundamentals") if has_data("fundamentals") else None
earnings_df     = load_parquet("earnings") if has_data("earnings") else None
analyst_df      = load_parquet("analyst_revisions") if has_data("analyst_revisions") else None
options_df      = load_parquet("options_flow") if has_data("options_flow") else None
insider_df      = load_parquet("insider_transactions") if has_data("insider_transactions") else None
short_df        = load_parquet("short_interest") if has_data("short_interest") else None

features = FeatureBuilder(
    price_df, fundamentals_df=fundamentals_df, earnings_df=earnings_df,
    analyst_df=analyst_df, options_df=options_df, insider_df=insider_df, short_df=short_df,
).compute_features()
```

Each load is independently optional (`has_data(...)` guard, matching `ascent/main.py`'s own
pattern) — a missing cache degrades that one sleeve to `INSUFFICIENT_DATA` via the existing
`DegenerateSignalError` guard, not a crash. This resolves `fundamental`, `earnings`, `analyst`,
`options_flow`, `insider`, `short_interest`. `altdata` and `earnings_tone` need no wiring change
here — they'll be re-verified as part of the real-data run, not assumed fixed.

**2. `run.py`: give each agent its own real price matrix.**

`score_agent(name, prices)` (`wf_scorer.py`) already takes a `prices` parameter per call — this
is a caller-side fix, not a rescoring-logic change. `run()`'s dispatch currently passes the
single shared `prices` (US-equity matrix) to every agent. Change: build a
`agent_prices: dict[str, pd.DataFrame]` keyed by agent name (`macro_agent` →
`load_parquet("prices_macro")`, `international_agent` → `load_parquet("prices_international")`,
`alternatives_agent` → `load_parquet("prices_alternatives")`), each deduped with the same
`_dedupe_prices_by_calendar_day` + `pivot_prices` treatment already applied to `prices_live`, and
pass `agent_prices[c.name]` instead of the shared `prices` in the `kind == "agent"` branch.
`us_equities_agent` stays `covered_by_sleeves`, untouched.

**3. Re-run the full audit and let the numbers land where they land.**

No change to `components.py`'s pinned list, `stats.py`'s math, `scorecard.py`'s verdict rule, or
the `DegenerateSignalError`/duplicate-agent-score guards added in sub-project 1's final review —
this sub-project only fixes what data reaches the existing scoring machinery. Whatever verdicts
come out (including a component staying `INSUFFICIENT_DATA` for a reason other than "data never
loaded" — e.g. genuinely too-sparse real coverage) are reported as-is, not adjusted to hit a
target.

## Explicitly out of scope

- `altdata_reddit.parquet` is missing (sec/transcripts/trends are present) — `altdata_alpha`
  degrades gracefully per its own existing logic; not fetched or fixed here.
- Any change to sleeve/agent/subsystem scoring math, the verdict rule, or the component fixture.
- `ml`, `llm_fundamental`, `narrative` stay deferred (per-fold retraining / LLM re-simulation is
  still out of scope, unchanged from sub-project 1's design).
- Sub-project 2 (target architecture) itself — this only produces the scorecard it will read.
