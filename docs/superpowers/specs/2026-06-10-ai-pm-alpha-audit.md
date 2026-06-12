# AI PM Alpha Audit — 2026-06-10

Target: OOS Sharpe > 0.65 (from 0.483), Alpha > 5% vs SPY (from 2.54%), close the +8.8% vs +15.9% live gap.

## Answers to the four diagnostic questions

### Q1 — Zero-return root cause (two independent mechanisms, both confirmed)

**Mechanism A — the blend math makes the AI PM invisible by construction.**
`ascent/strategy/earned_authority.py:257-274`:

```python
blended[sym] = ai_w * ai_portfolio.get(sym, 0.0) + qt_w * quant_portfolio.get(sym, 0.0)
if w >= MIN_WEIGHT:   # MIN_WEIGHT = 0.02
```

`ai_weight=0.05` is a **portfolio-mixing coefficient**, not a change cap or delta scalar. Final = 0.05·AI + 0.95·quant. Consequences:
- A ±2pp override recommendation (e.g. AI proposes 9% where quant has 7%) produces 0.05 × 2pp = **±0.10pp** of final weight. Below trading noise; the order engine may not even generate a trade for it.
- A name the AI PM adds that quant doesn't hold at, say, 7%: blended weight = 0.05 × 0.07 = **0.35% < MIN_WEIGHT (2%)** → silently deleted. At Level 2 (15%): 1.05% → still deleted. The AI PM is mathematically incapable of introducing a position until Level 3 (0.30 × 0.07 = 2.1%).
- Conversely, AMPLIFY picks (the system prompt's "where you MAKE money") at 10% vs quant 7% move the book by 0.15pp. Seven zero-IC days is the *expected* output of this math.

**Mechanism B — the measured zeros are not even the diluted blend; they are a hardcoded default.**
Track D (pure AI PM) daily returns come from `ascent/monitoring/ai_pm_counterfactual.py:117-127`:

```python
d_ret = _portfolio_return(ai_pm_weights or {}, prices) if ai_pm_weights else 0.0
```

and are consumed at `run_all_agents.py:1925-1926`:

```python
_d_ret_today  = _cf_record.get("track_d_return", 0.0) if "_cf_record" in dir() else 0.0
```

The Track D snapshot is written **only** by `snapshot_ai_pm()` at `run_all_agents.py:1326`, which sits inside the `else:` branch of `if ai_pm_result.fallback:` — i.e. only on a rebalance day where Phase 2 succeeded. Between the June 4 Level-1 bootstrap and the first rebalance there was no snapshot, so every daily `update_authority()` call was fed `(0.0, 0.0)`. **Proof:** `data_cache/ai_pm_shadow_returns.jsonl` shows real, distinct ai/quant returns through May 29 (shadow mode), then `ai_return=0.0, quant_return=0.0` on Jun 4, 5, 8, 9 — both tracks zeroed since promotion. The earned-authority Sortino buffers are polluted with fabricated zeros: promotion is impossible (edge ≡ 0) and real underperformance is masked.

**Mechanism C — on the one Level-1 rebalance (Jun 10), the AI PM result was discarded entirely.**
`logs/ai_pm_decision_log.jsonl` **does not exist** (never written once), and `outputs/ai_pm_theses/` has no thesis after 2026-05-27, while `verdict_2026-06-10.json` exists — so the rebalance ran but `ai_pm_result.fallback` was True. The probable chain: `_tool_propose_portfolio()` (`agents/ai_pm_agent.py:1656-1666`) rejects any submission lacking `feedback_acknowledged=true` whenever `data_cache/ai_pm_perf_feedback.json` exists (it does), **and appends a fallback AIPMResult to the result store before returning the rejection**. The red-team revision pass (`run_ai_pm`, `agents/ai_pm_agent.py:2436-2462`) re-prompts with a revision message that never mentions `feedback_acknowledged`; if the resubmission omits it, `result_store_v2[-1]` is the fallback and `run_ai_pm` returns an **empty portfolio in place of a valid initial proposal**. Downstream (`run_all_agents.py:1312-1313`): "AI PM fallback — using quant portfolio unchanged" → no blend, no Track D snapshot, no decision log, no thesis file. Exactly the observed artifact state.

### Q2 — Prethesis binding: quoted, but the structured handoff is broken and the binding has no teeth

`_SYNTHESIS_PROMPT_TEMPLATE` (`agents/ai_pm_agent.py:1112-1168`) **does** quote the sealed prethesis verbatim ("══ YOUR SEALED PRE-THESIS ══ {prethesis_text}") via `_format_prethesis_for_prompt()`, and gives per-name rules (confirm → 9-10%, contradict → stand down or defend with a dated catalyst) plus `thesis.pre_thesis_names must list your original high-conviction names and whether quant confirmed each`. So at the *name* level binding is real prompt-side.

But:
1. **The structured handoff is dead code.** `_strip_prethesis_for_phase2()` (`agents/ai_pm_agent.py:1018-1040`) reads `getattr(prethesis, "conviction_reasons", [])` and `getattr(prethesis, "sector_thesis", [])`. Neither attribute exists on `AIPreThesis` (fields: `high_conviction_names, names_to_avoid, sector_tilts, …`); the model's `conviction_reasons`/`sector_thesis` land only in `prethesis.raw`. Result: "PHASE 1 SOURCED CLAIMS" and "PHASE 1 SECTOR THESIS" blocks are **never injected**, and the recency gate has never gated a single claim. The Phase 1 prompt's threat — "Claims without source/data_date are stripped before Phase 2 sees them" — is unconditionally true for *all* claims.
2. **No code enforcement.** `feedback_acknowledged` is enforced in code; `pre_thesis_names` is not. Phase 2 can ignore the prethesis with zero consequence.
3. **The macro view is ambient context, not a position.** Nothing requires Phase 2 to state "FOLLOW prethesis" or "OVERRIDE because X changed." And the prethesis schema itself never demands a directional, falsifiable stance (see finding 8) — so on Apr 15 "elevated uncertainty, reduce exposure" was a fully compliant prethesis.

### Q3 — Decision log write path: confirmed broken, three independent gates

`_write_decision_log()` (`run_all_agents.py:131-177`) writes `"overrides_applied": overrides` where `overrides = ai_pm_result.thesis.get("quant_overrides", [])` — a **list**, and `run_post_mortem()` filters `if dec.get("overrides_applied")` (truthy). So:
1. The log has **never been written** — the only call site (`run_all_agents.py:1332`) is inside the non-fallback branch, and every Level-1 rebalance so far ended in fallback (Mechanism C). `_load_decisions()` returns `[]` → post-mortem returns `None` → `update_pattern_memory()` never called. Confirmed: pattern memory empty after 5 weeks.
2. Even once writes succeed, a rebalance where the AI PM made **zero overrides** (empty list = falsy) is skipped forever. "I agreed with the quant" is a decision with an outcome; the system can never learn from it.
3. `run_post_mortem` additionally requires `feedback["last_5_decisions"]` entries scored after the rebalance date. Current value: `[]` (verified in `data_cache/ai_pm_perf_feedback.json`), because ticker-memory scoring is itself gated on override-bearing decision-log entries. Triple-gated: pattern memory stays empty **even after the fallback bug is fixed**, unless gates 2 and 3 are changed.

### Q4 — Judge structural bias: confirmed, strictly negative by construction

`debate/judge.py` validation loop: `if new_w >= weights[sym]: continue` — increases are discarded; the apply site (`run_all_agents.py:1683`) re-enforces `new_w < old_w`. Parse failure defaults to `reduce_size` at confidence 0.3. The intervention-type enum (`adversarial_thesis|regime_sizing|coherence_risk|event_risk`) contains no upward type, and the authority clamp only bounds reductions.

Net judge weight delta across all three rebalances ever run:
- Apr 15: `reduce_size` @ 0.88 confidence, 0 position changes → book-wide exposure trim (Haiku weight adjuster) at the exact recovery low. Position-level delta 0, portfolio-level delta large-negative.
- May 27: EWY 10.26% → 9.30% = **−0.96pp**.
- Jun 10: PK 7.0% → 6.0% = **−1.00pp**.

Sum of position-level deltas: **−1.96pp, plus one global de-risk. Always ≤ 0 — the architecture cannot produce a positive delta.** Combined with the AI PM's amplifications being diluted to ~0.1pp by the blend (Q1-A), the system's only effective lever on the book is downward. This is the single largest structural alpha leak [POD].

---

## SECTION 1 — PUNCH LIST

### CRITICAL
- `[agents/ai_pm_agent.py:1656-1666, 2436-2462; run_all_agents.py:1312-1337]` — feedback gate + red-team revision pass replace a valid Phase 2 portfolio with empty fallback; decision log, Track D snapshot, thesis, and blend all skipped — **[QUANT/POD]** — 100% of AI PM output discarded on the only live Level-1 rebalance.
- `[ascent/strategy/earned_authority.py:257-274 blend()]` — ai_weight applied as 5% mixing coefficient + 2% min-weight floor → ±2pp overrides become ±0.1pp; AI cannot introduce names below Level 3 — **[QUANT]** — AI PM is IC-neutral noise at Levels 1-2; the "earned autonomy" ladder tests nothing.
- `[run_all_agents.py:1888-1934; ascent/monitoring/ai_pm_counterfactual.py:117-127]` — Track D/A★ silently default to 0.0 when snapshots or prices are missing; authority ladder fed (0,0) daily since Jun 4 — **[QUANT]** — promotion impossible on fabricated data; demotion signal masked.
- `[debate/judge.py validation loop + parse-failure default; run_all_agents.py:1683]` — judge can only reduce, defaults to reduce_size on failure; net delta across 3 rebalances strictly negative — **[POD]** — system trims when uncertain, never presses when confident; primary driver of the −7.1pp live gap.
- `[run_all_agents.py:131-177, 1330-1337; ascent/strategy/ai_pm_learning.py run_post_mortem]` — learning loop triple-gated (log never written / override-only filter / empty last_5_decisions) — **[MACRO-PM/QUANT]** — zero institutional memory after 5 weeks; pattern memory will remain empty forever without code change.

### HIGH
- `[agents/ai_pm_agent.py:2167-2170]` — `_prethesis_universe` undefined → Phase 1 always receives empty data grounding AND loses the Exa news context — **[MACRO-PM]** — the prethesis is formed blind on verified data; the anti-hallucination layer is off exactly where theses originate.
- `[agents/ai_pm_agent.py:1018-1040]` — `_strip_prethesis_for_phase2` reads nonexistent attributes; sourced claims + sector thesis never reach Phase 2; recency gate gates nothing — **[MACRO-PM]** — Phase 1's only falsifiable, sourced content is discarded every rebalance.
- `[agents/ai_pm_agent.py:737-815 propose_prethesis schema; 1112-1168 Phase 2 prompt]` — no required directional, falsifiable macro stance; Phase 2 never forced to FOLLOW/OVERRIDE it — **[MACRO-PM]** — the system produces hedges dressed as theses (Apr 15: high-confidence de-risk into the recovery).

### MEDIUM
- `[debate/judge.py except-clause]` — parse failure → `reduce_size` is a directional bet, not a safe default for an advisory layer — **[POD]** — guaranteed exposure cut on any LLM/JSON hiccup.
- `[agents/ai_pm_agent.py:1664]` — rejection path appends a fallback result to the store before returning "REJECTED", poisoning `result_store[-1]` even if the model retries late — **[QUANT]** — converts a recoverable prompt miss into a discarded rebalance.
- `[agents/ai_pm_agent.py propose_portfolio schema]` — no required `data_gaps[]`; tool failures (MiroFish timeout, options/COT unavailable) are visible in-band but never recorded in the thesis/decision log — **[MACRO-PM]** — post-mortems can't distinguish "wrong call" from "blind call".

---

## SECTION 2 — FULL SPEC PER FINDING

## 1. Fallback swallows valid AI PM proposals (feedback gate × red-team revision)
Severity: CRITICAL | Lens: [QUANT / POD]
**Problem:** The AI PM's entire output for a rebalance is replaced by `AIPMResult(portfolio={}, fallback=True)` whenever the red-team revision resubmission omits `feedback_acknowledged` — which the revision prompt never asks for. Result on Jun 10: no blend, no decision log, no Track D snapshot, no thesis. The fund paid for two Opus tool-use loops and applied none of it.
**Root cause:** `_tool_propose_portfolio()` at `agents/ai_pm_agent.py:1660-1666` appends a fallback result on rejection; `run_ai_pm()` at `agents/ai_pm_agent.py:2457-2459` returns `result_store_v2[-1]` unconditionally if non-empty.
**Fix:**
1. In `_tool_propose_portfolio`, delete the `result_store.append(AIPMResult(..., fallback=True))` line in the rejection branch — return only the "REJECTED: …" string so the loop can retry without poisoning the store.
2. In `run_ai_pm`, change the revision return to: `if result_store_v2 and not result_store_v2[-1].fallback and result_store_v2[-1].portfolio: return result_store_v2[-1]` else `return initial_result`. A red-team revision must never be able to *lose* the initial proposal.
3. Append to `revision_prompt`: "Remember to include feedback_acknowledged=true and worst_call_response in your thesis, as in your original submission."
4. In `run_all_agents.py`, call `_write_decision_log()` on **every** rebalance, including the fallback path, with a `"fallback": true` field — so discarded rebalances are visible in the log instead of leaving no trace.
**Files:** `agents/ai_pm_agent.py:_tool_propose_portfolio(), run_ai_pm()`; `run_all_agents.py` rebalance block.
**Success metric:** Next rebalance produces (a) `logs/ai_pm_decision_log.jsonl` entry, (b) `outputs/ai_pm_theses/<date>-thesis.json`, (c) `logs/counterfactual_ai_snapshots.jsonl` entry, (d) "[Runner] AI PM blend applied" in the run output. Add a regression test: red-team rejection on revision → `run_ai_pm` returns initial (non-fallback) result.
**Estimated impact:** Prerequisite for any AI PM alpha at all — currently the realized AI PM contribution is exactly 0. Unblocks every other fix.

## 2. ai_weight is a mixing coefficient, not an authority budget
Severity: CRITICAL | Lens: [QUANT]
**Problem:** `final = 0.05·AI + 0.95·quant` dilutes every AI decision 20×, and `MIN_WEIGHT=0.02` deletes any AI-only name below Level 3. "5% authority" should mean the AI PM can move 5pp of the book; it currently means the AI PM's *opinions* count for 5% of an average. The earned-autonomy ladder is therefore evaluating noise: even a genuinely skilled AI PM cannot generate a measurable Track-B edge at Levels 1-2, so promotion criteria can only be met via the (currently broken) Track D shadow comparison.
**Root cause:** `ascent/strategy/earned_authority.py:257-274 blend()`.
**Fix:** Redefine `ai_weight` as an **active-weight (tracking-error) budget** against quant:
```python
def blend(ai_portfolio, quant_portfolio):
    state = get_state(); budget = state.get("ai_weight", 0.0)   # 0.05 = 5pp one-way active weight
    deltas = {s: ai_portfolio.get(s, 0.0) - quant_portfolio.get(s, 0.0)
              for s in set(ai_portfolio) | set(quant_portfolio)}
    gross = sum(abs(d) for d in deltas.values()) / 2            # one-way deviation
    scale = min(1.0, budget / gross) if gross > 0 else 0.0
    blended = {s: max(0.0, quant_portfolio.get(s, 0.0) + scale * deltas[s])
               for s in deltas}
    # drop dust *after* scaling, then renormalize
    blended = {s: w for s, w in blended.items() if w >= 0.005}
    total = sum(blended.values())
    return {s: w / total for s, w in blended.items()} if total > 0 else dict(quant_portfolio)
```
At Level 1 a focused AI PM (one +2pp amplify, one −2pp trim, one 1pp new name ≈ 2.5pp one-way) gets its changes applied at or near **full proposed size**, while total deviation from quant is hard-capped at 5pp. Higher levels widen the budget instead of the mixing ratio. Keep the existing post-blend `validate_pm_proposal` and max-weight checks. Lower the dust threshold to 0.5% so budgeted new names survive.
**Files:** `ascent/strategy/earned_authority.py:blend()`; tests in `tests/` covering: ±2pp override survives at Level 1; aggregate deviation ≤ ai_weight; empty AI portfolio → pure quant.
**Success metric:** On the next rebalance, `sum(abs(final - quant))/2` is between 1pp and 5pp (not <0.2pp); the AI PM's amplify pick's final weight differs visibly from quant's.
**Estimated impact:** This is the difference between an AI PM and a rounding error. If the AI PM's calls have any positive IC, this converts ~0.1pp of expression into up to 5pp — a 20-50× increase in realized AI alpha per unit of skill, while keeping worst-case damage bounded at 5pp of active weight.

## 3. Authority ladder fed fabricated zeros (Track D scoring)
Severity: CRITICAL | Lens: [QUANT]
**Problem:** Since the Jun 4 promotion, `update_authority()` has received `(0.0, 0.0)` every day (proof: `ai_pm_shadow_returns.jsonl` Jun 4-9). Sortino edge ≡ 0 → promotion mathematically impossible; demotion triggers (catastrophic/hard/soft) also blind. The first 7-9 entries of the track buffers are fake zeros sitting next to real quant returns, corrupting the 21-day comparison window.
**Root cause:** `ascent/monitoring/ai_pm_counterfactual.py:117-119` defaults each track to 0.0 when its snapshot or the price dict is missing; `run_all_agents.py:1925-1926` defaults to 0.0 again (`if "_cf_record" in dir()`); `snapshot_ai_pm()` is only called on a non-fallback rebalance (`run_all_agents.py:1326`), so no Track D snapshot exists between promotion and the first successful rebalance.
**Fix:**
1. `score_daily()`: return `None` (not 0.0) for any track whose weights or prices are unavailable; include `"track_d_return": None` in the record.
2. `run_all_agents.py:1920-1934`: only call `update_authority()` when **both** `track_d_return` and `track_astar_return` are real numbers; otherwise print "[Runner] Authority update skipped — no Track D snapshot yet" and do not append to buffers. Replace the `in dir()` idiom with `_cf_record = None` initialization and an explicit `if _cf_record and _cf_record.get("track_d_return") is not None`.
3. Seed Track D at promotion/bootstrap: when a level changes (or `earned_authority.json` is hand-bootstrapped), write a Track D snapshot from the most recent AI PM proposal (or quant weights with a `"seeded": true` flag) so daily scoring starts immediately.
4. One-time repair: strip the leading zeros from `track_d_returns`/`ai_returns_21d` in `data_cache/earned_authority.json` (entries where the shadow log shows 0.0/0.0 pairs).
**Files:** `ascent/monitoring/ai_pm_counterfactual.py:score_daily()`; `run_all_agents.py` daily-learning block; `ascent/strategy/earned_authority.py` (accept-None guard); `data_cache/earned_authority.json` (repair).
**Success metric:** `ai_pm_shadow_returns.jsonl` shows distinct, nonzero ai/quant returns on the next 3 trading days; no `0.0, 0.0` pair appears unless markets were closed.
**Estimated impact:** No direct P&L, but gates all authority progression: with fixes 1-2 live, the AI PM can actually earn Level 2 (15pp budget) on real evidence — or get demoted on real evidence. Without it the "earned autonomy" core thesis of the fund is untested.

## 4. Judge can only subtract (one-way intervention authority)
Severity: CRITICAL | Lens: [POD]
**Problem:** A pod risk officer cuts losers AND presses validated winners. This judge is a one-way ratchet: every intervention removes exposure (−0.96pp, −1.00pp, plus the Apr 15 book-wide trim at 0.88 confidence into the recovery). Expected value of the debate layer is structurally ≤ 0 in up-trending regimes — which is most of the time.
**Root cause:** `debate/judge.py` validation: `if new_w >= weights[sym]: continue`; intervention enum has no upward type; apply site `run_all_agents.py:1683` re-enforces reduction-only.
**Fix:**
1. Add `conviction_press` to the intervention types. Judge may propose ONE change in **either** direction. For an increase require, cited in `reason`: quant top-quartile rank AND a bull/devil's-advocate argument referencing crowding=CLEAN or positive tail asymmetry (data already in the debate context).
2. Replace the reduction-only check with symmetric validation: `if abs(new_w - weights[sym]) < 0.005: continue`; clamp `new_w` to `[curr_w - max_change, curr_w + max_change]` from `adversarial_authority`, and cap increases at the 10% max-weight constraint.
3. Apply site (`run_all_agents.py:1674-1733`): allow `new_w > old_w` by funding the increase proportionally from all other positions (mirror image of the existing redistribution), then renormalize; keep the existing max-weight post-condition.
4. Track upward interventions in `adversarial_authority` separately so `conviction_press` earns/loses authority on its own 10-day predictions, exactly like reduction types.
**Files:** `debate/judge.py:run_judge()` (system prompt + validation loop); `run_all_agents.py` adversarial-intervention block; `debate/adversarial_authority.py` (new intervention type).
**Success metric:** Within 5 rebalances, at least one validated `conviction_press` is applied and scored at 10 days; cumulative judge intervention delta is no longer monotonically negative; intervention win rate tracked per direction.
**Estimated impact:** Largest single structural fix for the live gap. The Apr 15 failure mode (over-hedge the recovery) becomes correctable in-kind: when bull/asymmetry evidence is strong, the system can add 1-2pp to its best name instead of only ever trimming. Order of +1-2% annualized vs SPY if intervention IC is merely symmetric with the existing reduction IC.

## 5. Learning loop is triple-gated shut (pattern memory empty forever)
Severity: CRITICAL | Lens: [MACRO-PM / QUANT]
**Problem:** Five weeks live, ~270 planned learning sessions/year, zero post-mortems, empty pattern memory. The AI PM rediscovers the world every rebalance.
**Root cause:** (1) `logs/ai_pm_decision_log.jsonl` never written (see finding 1); (2) `run_post_mortem()` (`ascent/strategy/ai_pm_learning.py`) filters `if dec.get("overrides_applied")` — list-truthiness skips zero-override rebalances forever; (3) it then requires `feedback["last_5_decisions"]` scored entries (currently `[]`, itself gated on override-bearing log entries).
**Fix:**
1. Finding 1's fix makes the log exist (write every rebalance, fallback included).
2. In `_write_decision_log`, additionally write `"n_overrides": len(overrides)` and keep the list under `"overrides"`; in `run_post_mortem`, change the filter to fire on **any** decision ≥21 days old (drop the `overrides_applied` condition). A no-override rebalance gets a post-mortem of the *agreement*: did carrying quant weights work? That is exactly the Druckenmiller question — was doing nothing the right call?
3. Remove the hard `if not scored: return None` dependency on `last_5_decisions`: when no scored overrides exist, compute outcomes directly inside `run_post_mortem` from `target["final_blended"]` (or `quant_proposed`) using cached prices over the 21-day window — the data is in `prices_live.parquet` already.
4. Verify `ai_pm_pattern_context.txt` injection: `_build_temporal_context()` reads `data_cache/ai_pm_pattern_context.txt`, but `update_pattern_memory()` writes `ai_pm_pattern_memory.json`. Add a serializer that renders the JSON playbook to the `.txt` file after every update (or read the JSON directly via `get_pattern_summary()` in `_build_temporal_context`). Today, even a populated pattern memory would never reach the prompts — two different files.
**Files:** `run_all_agents.py:_write_decision_log()`; `ascent/strategy/ai_pm_learning.py:run_post_mortem(), update_pattern_memory()`; `agents/ai_pm_agent.py:_build_temporal_context()`.
**Success metric:** By 21 days after the next rebalance: one entry in `logs/ai_pm_postmortems.jsonl`, `ai_pm_pattern_memory.json` has ≥1 rule, and the next rebalance's Phase 1/2 prompt contains the "PATTERN MEMORY" block (assert in a test on `_build_temporal_context()` output).
**Estimated impact:** Compounding. Pattern memory is the only mechanism by which the AI PM's IC can *grow*; with it dead, the fund pays Opus prices for a goldfish. Indirect Sharpe impact via calibration: post-mortems also feed the conviction gate and calibration IC, which currently never fire.

## 6. Phase 1 runs blind — grounding and news never injected
Severity: HIGH | Lens: [MACRO-PM]
**Problem:** The prethesis — the seed of every Phase 2 portfolio — is formed with zero verified price/momentum/alpha-score data and zero news, despite both being fetched. The hallucination-prevention layer ("Attack #1") is disabled exactly where theses originate, and the Liberation-Day-style macro misread is the predictable result: the model reasons from training memory, which always counsels generic caution.
**Root cause:** `agents/ai_pm_agent.py:2167-2170`: `_prethesis_universe` is referenced but never defined; `"_prethesis_universe" in dir()` is always False inside the function, so `_build_data_grounding([], news_context=...)` is called — and `_build_data_grounding` returns `""` immediately when `symbols` is empty (line 48-49), discarding `news_context` too.
**Fix:**
1. Define the universe before the call: current holdings (`data_cache/merged_weights.json` keys, already loaded as `_portfolio_symbols` at line 2138) plus the top-25 names from `data_cache/last_alpha_scores.json`. Pass that list.
2. In `_build_data_grounding`, move the news block above the early return so `news_context` is rendered even when price grounding is unavailable.
**Files:** `agents/ai_pm_agent.py:run_ai_pm_prethesis(), _build_data_grounding()`.
**Success metric:** Log the grounding block length in Phase 1; next run shows a non-empty "VERIFIED DATA" + "LIVE NEWS" section in the Phase 1 system prompt (add a unit test that `run_ai_pm_prethesis`'s constructed prompt contains "VERIFIED DATA" when the parquet exists).
**Estimated impact:** Directly improves prethesis quality and falsifiability; reduces hedge-thesis output by giving the model actual momentum/news asymmetry to take a side on. Prerequisite for finding 8.

## 7. Phase 1 → Phase 2 structured handoff is dead code
Severity: HIGH | Lens: [MACRO-PM]
**Problem:** The only *sourced, dated, falsifiable* content the prethesis produces (`conviction_reasons`, `sector_thesis`) never reaches Phase 2, so Phase 2 synthesizes from freeform prose summaries only and the recency gate has never stripped a claim. The system's defense against stale-claim hallucination amplification exists but is unplugged.
**Root cause:** `agents/ai_pm_agent.py:1018-1040`: `getattr(prethesis, "conviction_reasons", [])` and `getattr(prethesis, "sector_thesis", [])` — neither is a field of `AIPreThesis`; the model's values live in `prethesis.raw`. Also `propose_prethesis`'s `input_schema` (lines 737-815) doesn't declare either property, so the model emits them inconsistently.
**Fix:**
1. Add `conviction_reasons` and `sector_thesis` to the `propose_prethesis` input schema (arrays of objects with `symbol/claim/source/data_date` and `sector/view/conviction/reason/source/data_date` respectively), marked required alongside `high_conviction_names`.
2. Add both as fields on `AIPreThesis` and populate them in `run_ai_pm_prethesis()` from `raw`.
3. In `_strip_prethesis_for_phase2`, read `prethesis.raw.get(...)` as the fallback for older stored theses.
**Files:** `agents/ai_pm_agent.py:_PROPOSE_PRETHESIS_TOOL, AIPreThesis, run_ai_pm_prethesis(), _strip_prethesis_for_phase2()`.
**Success metric:** Next rebalance's Phase 2 system prompt contains "PHASE 1 SOURCED CLAIMS (N validated)" with N > 0; recency-gate log line appears when a stale claim is submitted (unit-testable with a fixture prethesis).
**Estimated impact:** Phase 2 decisions become anchored to dated, sourced claims instead of prose vibes — improves override quality and makes post-mortems (finding 5) attributable to specific claims.

## 8. The system produces hedges, not views (no directional stance required)
Severity: HIGH | Lens: [MACRO-PM]
**Problem:** Apr 15: tariff risk fully visible in public data; a real macro PM's question was "is tariff fear priced?" — a directional, falsifiable call with asymmetric payoff. The system instead emitted "elevated uncertainty" and a 0.88-confidence `reduce_size`. Nothing in the prethesis schema or Phase 2 prompt requires a direction, an asymmetry estimate, or a falsifier at the *macro* level; "risk management dressed as analysis" is a fully compliant output.
**Root cause:** `agents/ai_pm_agent.py:737-815` (`macro_view` is freeform string); `_SYNTHESIS_PROMPT_TEMPLATE` (1112-1168) treats the macro view as context with no follow/override obligation.
**Fix:**
1. Add to `propose_prethesis` schema a required `directional_stance` object: `{direction: "risk_on"|"risk_off"|"neutral", thesis: str, upside_case_pct: number, downside_case_pct: number, falsifier: str, horizon_days: int}`. Require `upside_case_pct/downside_case_pct` so the model must state the asymmetry it is sizing for; require `falsifier` to be a market-observable condition ("SPY closes below X", "HY spread > Y bp").
2. In `_format_prethesis_for_prompt`, render the stance first, and append a hard instruction to the Phase 2 prompt: "Your sealed directional stance was {direction} with falsifier '{falsifier}'. Your thesis must include `prethesis_disposition`: FOLLOWED, or OVERRIDDEN with the specific new information that arrived after sealing."
3. Enforce in code (same pattern as the feedback gate): `_tool_propose_portfolio` rejects a submission whose thesis lacks `prethesis_disposition` when a prethesis exists.
4. Inject the stance into the debate `portfolio_state` so the judge sees the PM's directional view next to the bear case — `reduce_size` against a stated, unfalsified risk-on stance should require the judge to name what falsified it.
**Files:** `agents/ai_pm_agent.py` (schema, `_format_prethesis_for_prompt`, `_tool_propose_portfolio`, `_SYNTHESIS_PROMPT_TEMPLATE`); `run_all_agents.py` (forward stance into `portfolio_state`); `debate/judge.py` (context line).
**Success metric:** Every prethesis from now on contains a direction + numeric asymmetry + falsifier; the next `reduce_size` verdict's reasoning explicitly references either the falsifier firing or new post-seal information. Score stance accuracy in post-mortems (finding 5).
**Estimated impact:** Converts the debate layer's input from "uncertainty exists" to "is the PM's falsifiable view intact?" — directly targets the Apr 15 failure mode, the single largest realized loss vs SPY (~most of the 7.1pp gap).

## 9. Judge parse failure is a directional bet
Severity: MEDIUM | Lens: [POD]
**Problem:** Any JSON/assertion/API failure in the judge yields `reduce_size` — a real position-sizing action triggered by a parsing bug. An advisory layer's failure mode must be "no opinion", not "sell".
**Root cause:** `debate/judge.py` except-clause: `"recommendation": "reduce_size"`.
**Fix:** Default to `{"recommendation": "proceed", "position_changes": [], "confidence": 0.0, "key_risks": ["judge_parse_failure"], "degraded": true}` and add one retry of `extended_thinking_completion` before giving up. Surface `degraded: true` in the run log and dashboard so silent failures are visible. (`halt_and_review` remains reachable only via a successfully parsed verdict — a parser cannot diagnose catastrophe.)
**Files:** `debate/judge.py:run_judge()` except-clause.
**Success metric:** Simulated malformed LLM output (unit test) produces proceed/no-changes + degraded flag; no live exposure change ever attributable to a parse failure.
**Estimated impact:** Removes a guaranteed negative-drift tail: every future LLM hiccup currently costs a book-wide trim.

## 10. Rejection path poisons the result store
Severity: MEDIUM | Lens: [QUANT]
**Problem:** `_tool_propose_portfolio` appends a fallback `AIPMResult` *before* returning "REJECTED…", so even when the model retries correctly, the store carries a junk entry; in any path that reads `result_store[-1]` after a late rejection, a good earlier submission can be shadowed.
**Root cause:** `agents/ai_pm_agent.py:1664`.
**Fix:** Covered mechanically by finding 1 item 1 (remove the append). Listed separately so the regression test asserts: rejection → `result_store` unchanged; rejection-then-valid-resubmission → exactly one result.
**Files:** `agents/ai_pm_agent.py:_tool_propose_portfolio()`.
**Success metric:** The two unit tests above pass.
**Estimated impact:** Robustness; closes the second half of the finding-1 failure chain.

## 11. Tool failures are visible in-band but never recorded
Severity: MEDIUM | Lens: [MACRO-PM]
**Problem:** Executors correctly return "unavailable — proceed" strings (MiroFish, options flow, COT, SEC, transcripts), so Phase 2 *sees* failures — but nothing persists which inputs were missing. Post-mortems (once alive) cannot distinguish "the AI PM judged wrong" from "the AI PM was blind that day", so the playbook will learn wrong rules from blind calls.
**Root cause:** `propose_portfolio` schema (`agents/ai_pm_agent.py:706-732`) has no `data_gaps` key; `_write_decision_log` doesn't capture tool health.
**Fix:** (1) Add optional `data_gaps: [string]` to the thesis schema and one line to the system prompt: "List every tool that returned unavailable/timeout in thesis.data_gaps." (2) Cheaper and authoritative: have the tool executor in `_make_executor` record any return value containing "unavailable"/"failed"/"timeout" into a session-local list, and write it into the decision-log entry as `"tool_failures": [...]` — code-enforced, not prompt-dependent.
**Files:** `agents/ai_pm_agent.py:_make_executor()`; `run_all_agents.py:_write_decision_log()`.
**Success metric:** Decision-log entries contain `tool_failures`; post-mortem prompt (finding 5) includes them.
**Estimated impact:** Protects the learning loop's signal quality; near-zero cost.

---

## Sequencing

1. **Findings 1 + 10** (unblock the pipeline — without this nothing else is observable) →
2. **Finding 3** (make the scoreboard honest) →
3. **Finding 2** (make 5% authority mean 5pp) →
4. **Finding 4 + 9** (symmetric judge) →
5. **Findings 5 + 11** (learning loop) →
6. **Findings 6 + 7 + 8** (prethesis quality and binding).

Findings 1, 3, 6, 7, 10 are pure bug fixes with no strategy-risk and should ship before the June 24 rebalance. Findings 2, 4, 8 change live behavior and warrant one shadow rebalance each (log proposed-vs-applied without executing) before going live.
