# HANDOFF — paste everything below the divider into the new project's first conversation

**Also attach the four diagrams** (`docs/architecture/*.png` inside this package):
`01-overall-pipeline.png`, `02-plan-phase.png`, `03-codegen-phase.png`, `04-execution-loop.png`

---

I'm building a research compiler: a natural-language research question goes in, a
validated analysis with real charts and tables comes out, and the whole path is
reproducible. It is modelled on Bridgewater's Pocket Analyst Tool (PAT). A working
v0 already exists and runs end to end on real market data. I want to continue from
there, not restart.

## Why this project exists

I previously built a quant research and trading platform (ascent-capital: data →
features → alpha → portfolio → walk-forward → regime → multi-agent → paper
trading). The infrastructure works and I'm happy with it. **I pivoted away from it
because I could not find an edge that made it profitable or worth the opportunity
cost — and because I need to understand finance much better before I can.**

This tool is the response to that. It is a research instrument: a way to ask real
macro and market questions and get rigorous, reproducible answers. Building it also
forces me to state questions precisely enough to be executed, which is itself the
thing that teaches.

**Do not treat this as a subsystem of a trading platform.** The earlier design spec
framed it as a refactor of `ascent/monitoring/`, and every decision in that spec
inherited that subordination — it lived in `ascent/analyst/`, its data catalog was
built from that repo's 41 `logs/*.jsonl` schemas, and its anchor question was about
that repo's internal AI agents. **That framing is dead.** This is a standalone
research tool. The code in this package already imports nothing from `ascent`.

## The idea being copied, and the one that is not

From Santi Weight's part of the PAT talk: **treat agentic coding as a compiler
problem, not an agentic problem.** An LLM writes the code, but plain deterministic
Python owns the plan, the DAG, the validation, and the execution. Agents cannot skip
validation because they never invoke it. That inversion is the containment
mechanism.

**What does not copy: the data.** PAT works because Bridgewater has tens of millions
of internally modelled time series going back 50 years, millions of documents, and
five decades of investment logic already written down in machine-readable form.
Brendan McManus says it outright: "We didn't have to go back and write down
everything for agents. It was already there." I have public data — Yahoo, FRED,
and a few API keys. So this is not a small PAT. It is a different animal with the
same skeleton, and being honest about which animal it is determines what "good"
means here.

## The three-phase architecture (confirmed against the diagrams)

1. **PLAN** — a question goes to an orchestrator which decomposes it into a task
   list. Each task is tagged **Load Data / Transform Data / Visualization** and
   declares its output schema and its dependencies. PAT's line: "the plan is the
   analysis." They pay an expensive planning cost deliberately because it makes
   everything downstream cheap.
2. **CODEGEN** — the orchestrator fans out to N parallel coding agents, one
   dedicated LLM call per task. This works *only* because the plan is fully
   specified first: a visualization task already knows the schema of a dataframe
   whose code has not been written yet. That is why a 20-task plan takes about the
   same wall-clock as a 3-task plan.
3. **EXECUTION LOOP** — the code blocks form a dependency DAG, not a flat list. A
   node runs only once its upstreams succeed. Each node cycles execute → validate →
   repair → execute until it passes or exhausts its retries. A validator checks the
   output; on failure a debugger patches the code and returns it to the loop.

**Discrepancies already found in the diagrams — do not re-derive them:**
- The same task is named "Calc Asset Price Moves" / "Calc Asset Moves" / "Calculate
  Asset Moves" across the three diagrams. Hence `task_id` must be a stable slug,
  separate from a prose `title`.
- The Validator→Debugger arrow is drawn unconditionally; the "if validation fails"
  branch and its exit condition are unspecified.
- The repair loop appears at two granularities (whole code block in diagram 1,
  per-node in diagram 4). **Resolved: per-node repair.**

Full talk transcript: `docs/pat-talk-transcript.md` in this package.

## What already exists and works

A complete v0 for exactly one hardcoded question type: *"how did [asset A] and
[asset B] move during [named event]?"* Roughly 950 lines across eight modules.

```
analyst/
  types.py     Task, TaskCategory, OutputSchema, AnalysisPlan, NodeState, NodeResult
  events.py    4 named events, each with the market-reaction date pinned and justified
  plans.py     the hardcoded 5-task DAG + validate_plan() + topological layers()
  toolkit.py   the ONLY surface generated code may touch: load_prices, pct_move
  llm.py       thin Anthropic wrapper (standalone; no ascent imports)
  codegen.py   one LLM call per task, all generated in parallel
  execute.py   DAG runner, shape validation, per-node repair
  report.py    figures, CSVs, the generated source, report.md, run.json
  cli.py       entrypoint
  docs/        the four architecture diagrams + the PAT talk transcript
```

Run it:

```bash
.venv/bin/python -m analyst.cli --event ukraine_2022 --asset-a oil --asset-b dxy
.venv/bin/python -m analyst.cli --event oct7_2023 --asset-a brent --asset-b gold
```

Events: `abqaiq_2019`, `ukraine_2022`, `oct7_2023`, `iran_israel_2024`.
Assets: `oil` (CL=F), `brent` (BZ=F), `dxy` (DX-Y.NYB), `eurusd`, `gold` (GC=F).

The 5-task DAG resolves to three layers:

```
layer 1: load_asset_a, load_asset_b          (parallel)
layer 2: asset_a_chart, calc_asset_moves     (parallel)
layer 3: asset_moves_table
```

### Verified output

Ukraine 2022, WTI vs dollar index. 5 tasks, codegen 3.9s, execution 0.1s, all
tasks passed on first attempt:

| asset | +1d | +5d | +20d | +60d |
|---|---|---|---|---|
| WTI crude | −1.31 | 16.01 | 21.04 | 22.00 |
| US dollar index | −0.54 | 0.67 | 1.70 | 6.19 |

Oct 7 2023, Brent vs gold, also all first attempt:

| asset | +1d | +5d | +20d | +60d |
|---|---|---|---|---|
| Brent crude | −0.57 | 1.70 | −3.37 | −11.23 |
| Gold | 0.62 | 3.87 | 7.14 | 10.42 |

All sixteen figures were re-derived from raw Yahoo closes using no shared code path
with the pipeline, and matched exactly. The chart output is correct: event line on
the right session, the early-March 2022 spike to ~$124 is real.

The repair loop is proven: injecting a nonexistent helper name made attempt 1 throw;
the model received the traceback and attempt 2 returned 206 rows.

## The known hole — fix this first

I tried to prove the exhaust-and-surface path by pointing a task at ticker
`NOTAREALTICKER123`. **It did not exhaust.** It reported `done` in 2 attempts: the
debugger silently rewrote the ticker to `CL=F` and returned a *different asset's*
prices. Shape validation passed, because the shape was perfect.

That is the failure mode that actually matters — not a crash, but a plausible wrong
answer. Shape checking cannot tell "right shape, right asset" from "right shape,
wrong asset." The repair prompt already says *do not paper over it*; that was not
enough.

**The fix is per-task postconditions the debugger cannot satisfy by substitution** —
e.g. asserting the loader used the ticker the plan named. Postconditions are already
in the design vocabulary; they need to become real and enforced.

Two consequences to carry forward honestly:
- The exhaust-then-surface path is **implemented but not proven end to end**.
- `SKIPPED` propagation to dependents of a failed node is **implemented but not
  proven** either.

A smaller related gap: `NodeResult.error` is cleared when a repair succeeds, so a
run that self-healed leaves no record of what it healed from. On the third test run
(`iran_israel_2024`) the chart task took 2 attempts and the cause is now
unrecoverable. Keep a repair history per node instead — a node that quietly needed
two tries is exactly the node worth reading.

## Deliberately absent (scope, not oversight)

No general planner, no value caching, no self-review pass, no chat/coding agent
split, no static analysis, no unstructured or web search, no teach loop, no
benchmark suite. The `load_prices` disk cache stores raw downloads only — never
computed intermediates — so it is a fetch cache, not the PAT value cache.

## Decisions already made — do not relitigate

- **Standalone.** Not a subsystem of a trading platform.
- **Per-node repair**, not whole-plan repair.
- **Plain Python for the deterministic half.** No LangGraph, no agentic
  orchestration. That is what makes it reproducible.
- **No chat agent** — I am a solo operator who codes, and the coding session is the
  chat surface. But the **clarifying-question phase survives** and belongs in the
  planner: it should emit open questions and refuse to commit to a plan until they
  are answered.
- **Task categories are enforced, not decorative.** LOAD may do IO and has no
  upstreams; TRANSFORM must be pure and must have upstreams; VISUALIZE registers no
  series. This is the containment mechanism, and it is enforced in `validate_plan`.
- **Generated code reaches data by exactly one path** (`toolkit`), so a data problem
  is one bug rather than five.

## Ground rules

- Use `.venv/bin/python`. Never bare `python`.
- `import logging`. Never `from loguru import logger` — loguru is not installed.
- All LLM calls go through `analyst/llm.py`. Claude 5 rules, all hard:
  never index `resp.content[0].text` (thinking is on by default, so block 0 is
  usually a thinking block — walk the blocks); never pass `temperature`/`top_p`/
  `top_k`; never pass `thinking={...}` (depth comes from `output_config.effort`);
  `max_tokens` caps thinking and visible text together.
- `ast.parse` after each Python patch.
- **The grounding rule:** every number reported must come from an artifact actually
  read in that session. If it was reconstructed, say so. A confident wrong number is
  worse than an acknowledged gap — the previous project published a synthetic
  drawdown as fact once, and that is the mistake this whole tool exists to prevent.
- Verify before proposing. Trace the existing logic first.

## Dependencies

Present and working: `yfinance` 1.4.1, `pandas` 3.0.2, `numpy`, `matplotlib` 3.10.8,
`anthropic` 0.95.0, `jinja2`, `pydantic`.
Absent: `fredapi`, `pandas_datareader`, `plotly`, `tabulate` (the report renders its
own markdown tables rather than depending on tabulate).
`ANTHROPIC_API_KEY` is read from the environment or a repo-root `.env`.
`FRED_API_KEY`, `EXA_API_KEY`, `FMP_API_KEY`, `TIINGO_API_KEY` also exist in `.env`
and are unused so far.

## Where to go next — my open questions

The v0 proves the skeleton. The obvious next moves, roughly in the order I think
they matter:

1. **Postconditions**, to close the silent-substitution hole above.
2. **A real planner** — replace the hardcoded `build_event_move_plan` with an LLM
   that emits an `AnalysisPlan` and must survive `validate_plan`. The seam is
   already there.
3. **A series catalog** so the planner has a real vocabulary of data to name, and
   the plan validator can reject a plan that invents a series. FRED is the obvious
   first expansion beyond Yahoo.
4. **Determinism measurement** — PAT claims 95% identical code across two agents on
   the same plan. That figure is unvalidated outside Bridgewater. I should measure
   my own rather than inherit their confidence.

Ask me before writing code if anything is ambiguous. Start by reading the eight
modules — they are short.
