# CLAUDE.md — Ascent Capital (Public Showcase)

## What this project is

Modular Python quant research and trading platform. Data → features → alpha → portfolio construction → walk-forward evaluation → regime modeling → 4 specialist agents → orchestration → AI PM → LLM debate → execution via Alpaca paper trading.

This is the **public showcase branch**. Proprietary signal logic, regime-conditional weights, and LLM reasoning models are redacted. See README.md for architecture overview.

---

## Environment

Python 3.12, venv at `.venv/`. Use `.venv/bin/python`.

---

## Running

```bash
# Generate placeholder model artifacts first
python scripts/generate_mock_models.py

# Copy and populate the prompts file
cp private_prompts.yaml.example private_prompts.yaml

# Daily run (dry-run mode — no orders)
python run_all_agents.py --dry-run

# Tests
python -m pytest --tb=short -q
```

---

## What is redacted in this branch

| Area | File(s) | What was removed |
|------|---------|-----------------|
| LLM prompts | `agents/ai_pm_agent.py`, `agents/red_team_agent.py`, `debate/agents.py`, `debate/judge.py`, `debate/adversarial_engine.py`, `ascent/alpha/llm_fundamental.py`, `ascent/alpha/narrative_alpha.py`, `ascent/causal/dag_builder.py`, `ascent/reporting/` | All system prompts replaced with `PromptLoader` calls reading from `private_prompts.yaml` (gitignored) |
| Alpha math | `ascent/alpha/*.py`, `ascent/features/feature_defs.py` | Signal calculations replaced with `np.random` stubs; function signatures and DataFrame shapes preserved |
| Weights / priors | `ascent/alpha/stack.py`, `ascent/research/self_improve.py`, `data_cache/active_alpha_config.json` | Reset to equal-weight (1/13 ≈ 7.69%) baseline |
| Trained models | `data_cache/*.pkl`, `data_cache/sleeve_posteriors.json` | Gitignored; `scripts/generate_mock_models.py` generates structural placeholders |

---

## What is NOT redacted

`execution/`, `risk/`, `portfolio/` (MVO/Black-Litterman), `regime/`, `orchestrator/`, `ascent/research/walk_forward_runner.py`, `compliance/`, and all infrastructure code are published as-is — standard convex optimization, execution routing, and walk-forward evaluation are not proprietary.
