# HANDOFF PROMPT — copy everything below the line into the new conversation

**Also attach these 4 images** (drag them into the chat):
```
/Users/scott/IdeaProjects/ascent-capital/docs/architecture/01-overall-pipeline.png
/Users/scott/IdeaProjects/ascent-capital/docs/architecture/02-plan-phase.png
/Users/scott/IdeaProjects/ascent-capital/docs/architecture/03-codegen-phase.png
/Users/scott/IdeaProjects/ascent-capital/docs/architecture/04-execution-loop.png
```

---

I'm rebuilding the research layer of my quant platform (`~/IdeaProjects/ascent-capital`) on the architecture behind Bridgewater's Pocket Analyst Tool. A previous session did the research, wrote a design spec, ran a four-agent adversarial audit of that spec, rewrote it, and produced an implementation plan for the first slice. I want to continue from there — not restart.

## Read these first

- `docs/superpowers/specs/2026-07-31-pat-research-compiler-design.md` — the design, revision 2, post-audit
- `docs/superpowers/plans/2026-07-31-phase0-naming-and-liveness.md` — the 4-task implementation plan for Phase 0
- `CLAUDE.md` — durable project rules
- `CURRENT_VERIFIED_NUMBERS.md` — **the only citable source for any performance figure**
- `docs/REPO_MAP.md` — where things live (note: its `tests/` and `verify_docs.py` counts are stale)

## Ground rules that matter in this repo

- Every number must come from an artifact read **in this session**. If reconstructed, say so. This project has published a synthetic drawdown as fact before.
- Use `.venv/bin/python`. Never bare `python`.
- All LLM calls go through `ascent/llm/client.py`. Never index `resp.content[0].text` for a Claude 5 model; never pass `temperature`/`top_p`/`top_k`; never pass `thinking={...}`.
- `import logging` — loguru is NOT installed.
- `ast.parse` after each Python patch. Run `.venv/bin/python scripts/verify_docs.py` before committing.
- Verify before proposing. Trace existing logic first.

## The architecture (4 attached diagrams + confirmed spec)

**Overall: Plan → Codegen → Execution Loop**

1. **PLAN** — a natural-language question ("How much have currency and oil prices moved during past major war events?") goes to an Orchestrator, which decomposes it into a task list. Each task is tagged **Load Data / Transform Data / Visualization**. Example tasks: Load Oil Data, Load Currency Data, Calc Asset Price Moves, Oil Price Chart, Asset Price Moves Table.
2. **CODEGEN** — the Orchestrator fans out to N parallel Coding Agents, one per task. Each task gets its **own dedicated LLM call**, run independently (task 1 → LLM 1 → code block 1, etc). Output: one Python code block per task.
3. **EXECUTION LOOP** — the code blocks form a **dependency DAG**, not a flat list. "Load Oil Data" feeds both "Oil Price Chart" AND "Asset Moves Table"; "Load Currency Data" feeds "Calculate Asset Moves"; "Calculate Asset Moves" also feeds "Asset Moves Table" — so some nodes have multiple upstream parents. A node runs only once its upstreams succeed. Each node cycles **Execute → Validate → Edit Code → Execute** until validation passes. Python Execution contains Static Analysis, Value Caching, Optimization. A **Validator Agent** checks correctness; on failure a **Debugger Agent** patches the code and returns it to the loop.

**Discrepancies already found in the diagrams — do not re-derive:**
- The same task is named "Calc Asset Price Moves" / "Calc Asset Moves" / "Calculate Asset Moves" across the three diagrams. Hence `task_id` must be a generated stable slug, separate from a prose `title`.
- The Validator→Debugger arrow is drawn unconditionally; the "if validation fails" branch and its exit condition are unspecified in the diagrams.
- The repair loop appears at two granularities (whole "Python Code" block in diagram 1, per-node in diagram 4). **Resolved: per-node repair.**

## Decisions already made (do not relitigate)

- This lives in `ascent/analyst/`, inside ascent-capital. Not standalone.
- **Per-node repair.** Retry state lives on the node. `cache_key = hash(task_id, code_hash, upstream cache_keys, input series versions)`, so a repaired node re-runs against cached upstream values. A node exhausting `max_repair_attempts` (default 3) → `FAILED`; dependents → `SKIPPED` (distinct), so partial reports survive.
- Plain Python for the deterministic half — no LangGraph, no agentic orchestration. That is what makes it reproducible.
- No chat agent (solo operator who codes). But the **clarifying-question phase is kept**, moved into the planner: it emits open questions and refuses to commit to a plan until answered.
- Task categories are **enforced, not decorative**: LOAD may do IO and has empty `depends_on`; TRANSFORM must be pure; VISUALIZE registers no series. This is the containment mechanism.
- Build order requested by me: **Orchestrator + Plan phase first** (everything downstream keys off its output schema). I was told this is the "platform-first" path a feasibility review flagged as the most likely way the project stalls; I chose it anyway, knowingly.

## Agreed module layout

```
ascent/analyst/
  types.py              # Task, AnalysisPlan, OutputSchema, NodeState
  orchestrator.py       # run(question) -> Report; only public entrypoint
  plan/{generator,schema,validate}.py
  codegen/{dispatcher,agent}.py
  execute/{dag,runner,sandbox,cache}.py
  validate/{validator,static}.py
  debug/debugger.py
  catalog/registry.py
```

Draft schema already agreed:

```python
class TaskCategory(str, Enum):
    LOAD = "load"; TRANSFORM = "transform"; VISUALIZE = "visualize"

@dataclass(frozen=True)
class OutputSchema:
    kind: Literal["dataframe","series","figure","table"]
    columns: dict[str, str]
    index_kind: str          # market_trading_day | calendar_day | utc_timestamp | none
    row_semantics: str

@dataclass(frozen=True)
class Task:
    task_id: str             # generated stable slug
    title: str               # prose label
    category: TaskCategory
    description: str
    inputs: list[str]        # catalog series names — LOAD only
    depends_on: list[str]    # task_ids — never for LOAD
    output_name: str | None
    output_schema: OutputSchema
    postconditions: list[str]
    max_repair_attempts: int = 3

class NodeState(Enum):
    PENDING; READY; RUNNING; VALIDATING; REPAIRING; DONE; FAILED; SKIPPED
```

## Audit findings the spec already absorbed — do not repeat these mistakes

A four-agent review (evidence audit, PAT fidelity, feasibility, red team) overturned material parts of revision 1:

- **"Nothing that submits an order changes" was FALSE.** `run_all_agents.py` has 26 references into `ascent/monitoring/`, and `rebalance_trigger.is_triggered()` sets `is_rebalance = True` (flips real order submission on); `pre_rebalance_checklist.run_checklist()` gates execution. Containment must be **enforced by a check**, never asserted by directory convention.
- **"Demoting the AI PM to Level 0 improves measurement" was FALSE.** `blend()` never mutates `ai_portfolio`, so `snapshot_ai_pm` captures the pure pre-blend vector at any `ai_weight` — Track D was already clean. Demoting only makes Track B ≡ A★, deleting information. The demotion is **withdrawn and deferred** pending the 2026-08-05 rebalance.
- **Not all 29 `ascent/monitoring/` modules are analyses.** ~11–14 are gates, schedulers, or live IO. Carved out explicitly: `pre_rebalance_checklist`, `rebalance_trigger`, `weekend_runner`, `alert_system`, `forward_pnl_tracker`. These stay hand-written.
- **"The catalog is a lens, not a database" was misleading.** 41 distinct `logs/*.jsonl` schemas, 6–8 adapter families, 4–6 weeks.
- **The "schemas would have caught 10 bugs" claim is really ~5.** The others (`reduce_size` renormalization, Sortino ÷ √252, the walk-forward cache-key collision) are logic errors in code declared out of scope, already fixed by ordinary means.
- **`scripts/generate_performance_page.py:847` imports from `ascent/monitoring/ai_pm_counterfactual.py` to build the PUBLIC GitHub Pages output.** No generated analysis may feed that page until determinism is measured.
- Realistic sizing: working v1 is **10–14 weeks FTE ≈ 4–6 months calendar** for a solo builder.

## Where we stopped

Phase 0 (`docs/superpowers/plans/2026-07-31-phase0-naming-and-liveness.md`) is written but **not executed**. Four tasks: decision-log upsert on `(date, phase)`, dedupe the 9 existing rows to 2, an `earned_authority` liveness check in `verify_docs.py`, and the canonical series registry.

**The Plan phase depends on the catalog**: `LOAD` tasks take `inputs: list[str]` of catalog series names and the plan validator must check they resolve. So the Phase 0 registry is a hard prerequisite, not a sequencing preference.

**Both the spec and the plan are uncommitted, on `main`.**

## What I want next

Write the implementation plan for the Orchestrator + Plan phase — either assuming the Phase 0 registry lands first, or folding the registry in as its first task. Ask me if anything is ambiguous before writing code.

---

## APPENDIX — Full transcript of the Bridgewater PAT talk (Interrupt 2026)

Speakers: Brendan McManus (team lead, applied AI), Michael Ran (investor lead), Santi Weight (technical lead).

> Hello, everyone. My name is Brendan McManus, the team lead of the applied AI team at Bridgewater Associates, a systematic macro hedge fund. I've been at Bridgewater for almost a decade now, where I started as a software engineer, then became a systematic investor and researcher, and for the past few years I've been focused heavily on bridging the gap between investing and technology. I'm here today with my colleagues Michael Ran and Santi Weight, our investor lead and technical lead of this project respectively, to talk to you about a great tool that we built internally called PAT, the Pocket Analyst Tool. By the end of this talk, all of you should have seen how we built an AI analyst that's capable of performing hours of expert research work in minutes — that's been internally deployed to hundreds of investors and that learns from every single interaction. On top of that, we'll also show you exactly how we architected such a thing.
>
> So before we show you what we built, I wanted to go through some quick framing on Bridgewater's approach to AI. Bridgewater has been spending decades — 50 years — thinking about how to encode markets and economies into compounding systems. And all of this really started with what you're seeing on the screen: our 1980 bond system, written down on a yellow legal pad. The idea here was quite simple. Every time you wanted to make a trade, write down exactly the rules for why you think that trade makes sense. Write down the exact causal logic. Because once you do this, another investor can take a look at what you've written, help you figure out where you went wrong, and help you improve upon this process. Every time you learn something new, you scratch out a rule, you write down a new one, and this creates a pretty incredible learning process that has been the foundation of everything that we've done at Bridgewater for the past 50 years.
>
> Over the decades, we've compounded upon this process significantly, taking every lesson, every methodology, every rule that we've developed about the trades we make — how and why — and codifying them into an expert system that's both machine-readable and human-readable. And it's really now that we're sitting on a pretty tremendous trove of data. And all of this data is really what sets us up well for the AI era. We didn't have to go back and write down everything for agents. It was already there for us to draw upon.
>
> Before we get into the tool, I wanted to talk about how we're approaching AI at Bridgewater more generally. We're doing this in two ways. The first is as investors. As investors, we need to deeply understand every major dynamic that's shaping global markets and economies. Just as we had to understand COVID or the recent oil supply shock, we've also had to understand AI. What's the shape of the supply-demand mismatch? What's driving the buildout? And how do these things ultimately impact markets? This understanding is table stakes for us as investors. The second way we're approaching AI is as practitioners — which is what we're mostly going to be spending our time on here today. As practitioners, we're applying AI across all aspects of our research process, with the ultimate goal of building out an artificial investor that's capable of the full range of activities that our human investors are performing every single day.
>
> So what do our human investors actually do? Well, we think about it as a research circle. Investors are constantly perceiving what's happening in the outside world, formulating questions about what is true and what they might potentially be missing, performing analytical investigations to try to answer these questions, synthesizing their findings, and ultimately taking anything that they learned and putting it back into the compounded understanding — the expert systems that we've built out. It's really this last step that's key. Everything that we learn as a function of this process gets put back into a shared memory that humans are able to draw upon. And so you might imagine that if you're going to build out an artificial investor, they need to be able to do all of the different steps of this research process. You can imagine having built out discrete sub-agents focused on each portion of the research circle. And that's exactly how we're approaching AI. We're building out discrete sub-agents for each of these different things that human investors must do, drawing upon the same understanding that we've built up over the last 50 years.
>
> But today, we're only going to be talking about one of these agents — the one focused on the investigation portion of the research process: the kinds of deep analytical work that take our human analysts days and weeks to perform. We called this tool PAT, the Pocket Analyst Tool. And just to set expectations: PAT is not about how we trade. It's really about performing deep exploratory research, enabling our investors to go after questions that they never would have had the bandwidth to pursue before.
>
> So what did we actually build? We built PAT, the Pocket Analyst Tool. And from day one, the spec was simple. We had to enable PAT to do all of the things that our human investors are doing as they perform investigatory or analytical work. And that really starts with data. PAT had to be able to search across and read all of the different data that we have internally — from structured time series data, for example, stock prices going back decades, to unstructured data, for example, broker-dealer research pieces that we subscribe to, or internal memos that we produce. PAT has to be able to search across and read all of it. PAT also has to be able to use all of the different tools that our human analysts have access to. All of the proprietary tools that we've built out for visualization, for diagnostics, or for evaluating the quality of the indicator ideas that we've built out. In addition to this — and this is where things might get a little bit more interesting for a technical audience — many of the analyses that PAT is running would take a human analyst many hours to run. And this means that the analyses are quite complex, which means that PAT's analyses have to be completely diagnosable — not just for humans, but also for agents running in the background, reading through the traces, understanding and making sure that every calculation is correct. On top of this — and this is really where building upon all of the things that we've written down over the past 50 years starts to pay off — PAT knows all of our context. It has access to all of our investment processes, our frameworks. It knows exactly how our analysts are supposed to be doing work, because we've been writing it down for the past 50 years. And lastly, PAT has to learn. It has to compound upon its own learning, not just for one investor, but for every investor at the firm.
>
> PAT is not a prototype today. It was actually deployed internally several months ago. And we now have hundreds of investors using it every single day, which is leading to a pretty incredible flywheel of improvement: as investors use PAT for real research, we have agents continuously running in the background, scanning through these interactions, figuring out where PAT went wrong, developing human-audited benchmarks, which then results in changes to context, and also to the harnesses that we built for PAT — which results in PAT improving not just for one, but for everyone.
>
> And lastly, before we get into what we built, I wanted to leave you with one question that I pretty commonly get, which is: how does a 50-year-old hedge fund actually build out something like what you're about to see? Well, it really starts with being willing to shake yourself up. The team that built out PAT was ultimately an internal applied AI startup that was incubated within the broader firm, with the ability to move incredibly flexibly and quickly while being able to draw upon the resources of the broader firm. In addition to this, we've established multi-archetype teams where we have investors, technologists, and scientists sitting side by side, building alongside each other. And I think this is really a key thing if you're going to build out a product like this. Investors bring the context and the domain-specific expertise. Technologists bring the architectural capability. And scientists bring the rigor. It's really these types of multi-archetype teams that are necessary if you're going to build AI systems for expert users. And speaking of expert users, we have hundreds of them internally using all of our AI tools — not just PAT — generating signal daily on how these things should be evolving and improving over time. And lastly, we have a pretty incredible compounding ecosystem to plug into. 50 years of shared data, tooling, and methodologies, all built out not just for our human analysts to plug into, but also for agents on our journey to build out a fully artificial investor that's capable of everything that humans are able to do today.
>
> And so with that, I'm going to pass things off to Michael Ran, investor lead, to go through a demo of what we built and take you through the product architecture.
>
> All right. Thank you, Brendan. My name is Michael Ran. I'm the investor lead working on the Pocket Analyst Tool, and a little bit about myself: I've been at Bridgewater for five years, where I joined as a technologist but have spent most of my time in an investor role. And nowadays, the thing I'm mostly focused on is how we can infuse AI into our investment process. With that, I'd love to jump into a demo of the Pocket Analyst Tool to highlight its capabilities and show you how investors at Bridgewater are using it.
>
> Behind me you can see the PAT landing page, which has the prompt we'll be using for today's demo. The gist of it: we're asking PAT to look at how markets have responded to the recent conflict in the Middle East and to compare today's events to similar historical episodes. Then lastly, we asked PAT to put together some visualizations that highlight how similar or dissimilar today is relative to previous oil supply shocks. This is a real question we've been working on at Bridgewater over the past couple of months, and PAT has been a tool leveraged by investors to help expedite our research process.
>
> But before we submit the prompt, I'd like to spend a moment talking about an interesting security problem we ran into when designing PAT's harness. Our starting point was: for PAT to truly be helpful, it needs to be able to access all of the information that investors have access to when conducting research. But the thing is, at Bridgewater, different investors have access to different information. So for example, you may have one investor who has access to what our positions are across all markets today, and it's important for that person's PAT to also have this information. But then you might have analysts who are not exposed to this, and it's similarly critical that we do not accidentally leak that secure IP to those analysts. So unlike a harness like Claude Code, where everyone is interfacing with the same system prompt and the same tools, each person at Bridgewater has a unique version of PAT tailored to what they can and cannot see. Practically speaking, this uniqueness is just a function of what context and what tools each person's PAT has.
>
> So with that, we'll go ahead and submit the prompt. The first thing that takes place in this analysis is PAT searches the web and our vault of unstructured data to better understand what's happening in the world today, contextualized against history. Web search itself is table stakes when it comes to these modern chat applications. But the real differentiator here is the breadth of unstructured content that we subscribe to that PAT is able to search over. We have a database containing millions of documents from all around the world that includes content such as broker pieces, earnings transcripts, internal emails, and more. And this database is updating in near real time, with thousands of new pieces coming in each day. And this ties back to the point I was making on the previous slide: for PAT to really provide leverage, it has to access everything that its user has access to, such that it can try and emulate what its users actually do.
>
> Once this context is gathered, the next thing PAT does is search our time series database for the data required for the analysis. This is a database that contains tens of millions of series that we've been modeling internally for 50 years, and this database contains data from the outside world — like the price of oil — and also internally derived concepts, such as what we think inflation will be 12 months out. The search agents use traditional search techniques like RAG, re-ranking, et cetera, but the thing we found to be a huge difference-maker was layering on an element of human-like inspection. What I mean by that is: when a human researcher is looking for data, they don't just anchor to what the name of the time series is. They'll look at the frequency, the currency of the series, and then most importantly, whether the values in the series align with their priors. So embedding this sort of reasoning into our search agent is something that got us up from roughly 50% accuracy all the way to 90%.
>
> Now, after PAT has context and data at hand, it comes back to the user with clarifying questions and potentially ideas for additional angles to explore. During PAT's development, we came to the view that the plan really is the analysis. If we can create a high-quality, detailed plan, we felt confident that we could consistently take this plan, intelligently execute it, and produce our desired outputs. Santi will be diving into this a bit more later. So while agents asking clarifying questions is pretty standard in chatbots nowadays, the thing we really focused on was the substance of those questions. We devoted lots of time and energy into developing the context and benchmarks that helped shape this capability. We taught PAT what makes a good research question versus a bad one, so this back-and-forth helps humans — who tend to under-invest in planning — flesh out what we consider to be a high-quality research plan.
>
> Now, after the ambiguous points are all resolved, PAT enters its planning phase. During the planning phase, the things it's doing are: A, coming up with all of the data frames that will be produced in the analysis; coming up with the schemas of all of those data frames; and then most importantly, coming up with how all those data frames connect. This planning phase is relatively expensive from a time perspective, but it's a cost we pay deliberately because of what it lets us do during execution.
>
> So now we have the plan locked in, and the first step of plan execution is generating the code. Given we have such a detailed plan, we're able to generate the code for each data frame in our analysis in parallel using sub-agents. And this works because each sub-agent knows what data frames it depends on, knows the schemas of those data frames, and also knows what the schema of the data frame it's producing should be. So this lets an analysis with three data frames or a medium-sized one with something like 30 data frames take roughly the same amount of time to generate code in both scenarios.
>
> Once we finish generating the code, we execute the Python functions with an agent overseeing this execution and stepping in if it sees things like runtime errors or nonsensical values. And before we proceed from execution, I want to call out the fact that the time series outputs from these analyses land in the same database that we actually pulled our inputs from. And I think this is important for a couple of reasons. One: it shows that any output from a PAT analysis is indistinguishable from any of the human-uploaded series that we've been producing for many years. And more importantly, any output from a PAT analysis can serve as an input to a subsequent one. So this creates an environment where humans and agents can very easily compound and leverage each other's work.
>
> Now, after execution completes, just like you'd want your junior analyst to double-check their work before coming back to you, we want PAT to do the same. So at this point in the analysis, PAT will look at the data that's been computed and the visualizations it's produced and make sure that the numbers seem sensible and the charts look clean. If it sees something that looks a bit off, PAT will take a step back, diagnose the issue, refine the analysis, and make sure it's satisfied with its result before coming back to the user.
>
> And then the result itself is this interactive report where the visualizations look just like the ones investors at Bridgewater are producing. PAT is using the same in-house charting library and leveraging the same existing text deck that we've been developing internally for decades. So you can see here how you can zoom in, zoom out of these charts, and you can even send the data from this interactive report to our internal charting tool, where you can make further modifications on the fly.
>
> Before handing it over to Santi, I want to speak a bit to how PAT gets better as you use it. There are two primary ways in which this happens. The first is the autonomous one that Brendan alluded to earlier, where we have agents reviewing completed conversations looking for ways in which PAT can get smarter. And the second one — and the one we'll be showing here — is the more explicit option, where a user can kick off this learning process within the context of an analysis if they think there is something to be learned from their interaction with PAT. So right here, we just have the user asking for a different set of visualizations. Note the user didn't say anything was wrong — they just wanted a different perspective that they thought was important to answering the question at hand. If they think that PAT should have produced or suggested this set of visualizations from the jump, they can click the Teach button, which will then spawn an agent to go through the conversation looking for things like behavioral mistakes, context gaps, or user steering that can be front-run. The user is then able to modify this or send it as-is, and when they submit it, what happens is: first, an agent on the back end will create a benchmark that we expect to fail. This shows that we can reproduce this poor behavior. It will then iterate on our context repositories or the harness itself until that benchmark passes. And then after confirming that by making this benchmark pass, we didn't cause the rest of our suite to fail, we get a Slack message with the pull request, including the changes the agent wants to make to PAT. And the thing this does is that the next time a human comes to PAT with a similar question, we expect them to get the better version of PAT right out of the box.
>
> So now I'd like to hand it over to Santi for the technical overview.
>
> Hello, everyone. Thanks, Michael. I'm Santi, the technical lead for the Pocket Analyst Tool. I'm sure many of you in the crowd are building coding agent products just like we are. And it's really hard to build one. Coding agents are really fickle and unpredictable, they often make mistakes. And when you're really unlucky, they'll go off the rails and try to nuke your data and all the rest of it. And so it's already hard to build a good product that people like, but then it's really hard to build a product they want to embed into their daily workflows. At a hedge fund, we're trying to trade billions of dollars, and so we can't have just vibe-coded analysis be the underpinning of how these analyses go.
>
> My background is in compiler theory and programming language design. Compilers have a very similar footprint of requirements: they're fully deterministic, fully correct, and reliable. You can't have an off-by-one error when you're flying a plane, for example. And there's a similar shape here. A compiler takes user code and compiles it down to something like JavaScript, and a coding agent takes a user prompt or plan and compiles it down to Python. We really like this approach, and I don't have a lot of time today, but we're going to focus on this as a learning that maybe you might be able to take home to your own work.
>
> We're going to start with the chat agent. The goal here is for the chat agent and the user to come to a common understanding of what they're trying to accomplish. The chat agent is implemented in LangGraph. We primarily use it for persistent state. It has out-of-the-box cancellations and continuations. We used to manage that stuff ourselves with a much worse effect. The chat agent has access to tool calls. Michael called some of these out. So there's data series search, unstructured search, and each of these would be a tool call themselves. And then once the chat agent knows all the data it needs for its analysis, it's going to make a plan and invoke a tool call to a sub-agent, which is our coding agent. And that's going to produce a Python-Pandas analysis.
>
> So why would you separate your two agents? Early on, we decided that our investors are not programmers by trade — they care about investment. So we decided to keep the chat purely about investment content. And the result is that we have a product where coding is a pure implementation detail. From the chat, you can't tell that there's code under the hood. Other happy accidents: you get clean context, so each agent becomes specialized at its job and naturally improves. And we get to tailor the chat experience a lot.
>
> What we're going to talk about now is that our investment domain context is really high quality. We teach the chat agent how to talk like a Bridgewater investor. There's a lot of jargon to teach it. And so the user and the agent both talk to each other like coworkers at Bridgewater. We also let our investors contribute — Michael was showing that — contribute like a developer on the codebase. And this is a pretty good point. Your user is better at writing context than you are, most likely. And so having low ego about it and letting them contribute directly is a good way to win. The last point I make about the chat: we teach the chat agent not just context — which you end up with this nebulous context that is very high-information but doesn't really feel like a workflow. Instead, we have step-by-step guides for the agent on how to handle certain types of analyses. And it feels much more like a product at that point. Dependable workflows.
>
> The rest of the talk is about my personal favorite part, which is the coding agent. This is a high-level architecture diagram of the coding agent. Everything you see here is actually just Python code. It's influenced by LangGraph, but there's no agentic orchestration.
>
> We're going to start with the left, which is the analysis plan that the chat agent makes. The analysis plan is broken up into tasks. Each task maps approximately to one Python function that's going to calculate a data frame. Here's an example schema. It gets a name, a description of what to calculate, and then structural and semantic information about the data frame that should come out. We expect every task to deterministically compile via LLM to a piece of code. So two LLMs operating on the same task should produce code that, when run, is semantically equivalent — the same output values, exactly the same. The result is that our analysis plan isn't just a to-do list like you'd see in Claude Code. Instead, we think of it as a natural language Python project.
>
> And because we have so much detail, now we're going to go into code gen and we can apply some fancy techniques. The first thing: we split it into tasks and then do parallel LLM generation. And because our plan is so detailed, a visualization task at the end of the plan already knows everything it needs to consume from code generation that hasn't yet been completed for loading data, say. And when we compare this to Claude Code, you can see that on average — for the same context and same plan — we're about four times faster at generating code. But then we have this hyper-scaling, so a 20-task plan takes the same amount of time as a three-task plan.
>
> So now we have code and we're going to execute it. You'd want to just execute the code naively, but sadly, LLMs today aren't perfect on our tasks, so they don't normally one-shot it. Instead, what we do is take the task that came in from the plan and the code that was generated from the task, then run the code, compare it to the task, see if it's correct, and if it's not, edit the code and keep going until we're done. The first thing we do on the code is static analysis, then figure out the DAG, and we apply our validation agents in parallel. Here, there are two tasks being validated at the same time. So a five-task plan comes down to three layers, and a 20-task plan might be four or five layers of validation. The main point I want you to take away from this part of the slides is that we enforce correctness in the architecture. Again, no agentic orchestration. This is regular Python code, so the guardrails are really hard, and the agents cannot forget to validate. They are forced to validate. The result is that when we run our test suite on any plan, 95% of the time, the code that comes out is exactly the same for two different agents. So it's essentially a deterministic coding agent. And then because we have such a reproducible agent, as we're scaling and hill climbing and evaluating, we have something much more dependable than vibes-based or LLM-as-judge evals.
>
> Cool. One more topic I'm going to touch on, which is the execution layer. Normally, coding agents invoke their code themselves. They generate code, and then call it themselves via terminal. There are a couple of trade-offs here. That's high latency with the tool calls, and as you've all experienced, they'll sometimes get lost along the way. So instead, we run the code for the LLMs. We do a classical static analysis pipeline where we inject caching annotations into the Python code to avoid re-execution and run it through a custom framework. This is an example benchmark we have for Claude Code invoking its own code versus Pocket Analyst. We're faster because we never double-load data or double-execute intermediates. But the actual win is not on the first time you run the code, it's the second time. This is a benchmark where we take the last chart in a plan and just change the name. Claude Code is going to go and rerun all the code — basically the same amount of time, though it is faster at editing the code. But Pocket Analyst basically has instantaneous code execution for the second round. And this means that when you're working on the product as an investor, you can make small tweaks to your investment analysis without the overhead of a regular iteration.
>
> OK, we're out of time. We had many learnings, but unfortunately we can only cover these. The first takeaway: we really believe in specializing our agents. We don't really believe in generic, powerful agents. They make really cool demos. I'm sure many of us have given them. But it's really hard to make that a daily workflow that you can depend on. Instead, we take often very narrow workflows and benchmark them very heavily, and then we hill climb those benchmarks. You can then compound the agents after the fact, but it's harder to go back the other way. And this last thing is kind of a thought exercise — something I hope is exciting and someone might take away: think of agentic coding as a compiler problem, not an agentic problem. Compilers have been around for decades and they have a ton of techniques for how to generate code more reliably, correctly, and deterministically. We'd love to chat about this.
>
> With that, I'd like to shout out our direct team. There were many others, but this is the direct team. Thank you.
