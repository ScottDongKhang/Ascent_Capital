# Ascent Capital System Upgrade — Master Overview

> **For agentic workers:** Implement sub-plans in order A → B → C → D. Each plan is independently shippable. Do not start B until A's PnL infrastructure is verified.

**Goal:** Make Ascent Capital beat the S&P 500 consistently, run with hedge-fund-grade discipline, and self-improve on real signal rather than noise.

**Architecture:** Four independent upgrade tracks executed sequentially. Each track ships working, tested software before the next begins.

---

## Sub-plans (implement in this order)

| Plan | File | What it fixes |
|------|------|---------------|
| A | `2026-04-16-plan-A-monitoring.md` | SPY benchmark, US equities PnL gap, daily attribution report |
| B | `2026-04-16-plan-B-portfolio-hardening.md` | EM cap, reduce_size enforcement, regime detector staleness |
| C | `2026-04-16-plan-C-self-learning.md` | Real verdict scoring, live self-improve evaluator |
| D | `2026-04-16-plan-D-llm-enhancement.md` | Richer agent context, tool-augmented debate, prompt caching |

---

## Quant + AI Balance Assessment

**Current state:**

| Layer | Quant | AI |
|-------|-------|-----|
| Alpha | HMM regime ✓, CPCV ML ✓, Almgren-Chriss ✓ | None |
| Portfolio | Sector constraints ✓, water-fill cap ✓ | None |
| Debate | Monte Carlo sim ✓ (numbers exist) | 5 LLM agents ✓ but prompts are generic |
| Execution | Kill switch ✓, cost model ✓ | Haiku weight adjustment (weak) |
| Learning | Heuristic self-improve ✗ | Blind spot detection ✓ (Haiku) |

**The core gap:** The AI agents make claims about correlation, VaR, and factor exposure but have no access to compute them. The quant layer produces these numbers but doesn't pipe them to the agents in structured form. The debate is LLM reasoning without quantitative grounding.

**After the upgrade:** Agents receive pre-computed factor exposures, realized correlation matrix, VaR, and attribution breakdown as structured inputs. The debate becomes quantitatively grounded.

---

## Why this beats SPY

Right now: 37% EM+commodity, 7.5% true defensive, stale regime, fake self-improvement.

After upgrade:
- EM+commodity hard-capped at 20% → more US equity exposure
- Regime-conditional defensive floor enforced → right posture for environment
- reduce_size verdict actually reduces size → debate decisions have teeth
- Self-improve runs on real 30-day forward returns → variants that actually work get promoted
- LLM agents see factor exposure, VaR, attribution → debate quality jumps
