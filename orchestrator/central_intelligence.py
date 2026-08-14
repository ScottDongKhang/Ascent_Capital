"""
orchestrator/central_intelligence.py
Central Intelligence — Orchestrator v2

Upgrades over v1:
1. Conviction-weighted allocation — agents that agree on the same names get
   a conviction bonus on top of skill score. Pure Sharpe told you who was good.
   Conviction tells you who is confident right now.
2. Regime-conditional veto — in crisis regime, macro agent gets override
   authority. Merged weights are forced toward macro's positioning above a floor.
   NOTE: macro_agent is no longer invoked from the daily run (scored CUT on its
   real universe), so this veto (_apply_crisis_veto) is defined but not called —
   see its module comment below.
3. Partial contradiction detection — factor-level conflict check, not just
   hardcoded symbol pairs. Catches rate-sensitive clustering, sector crowding,
   and directional conflicts across agents even when instruments differ.

Usage:
    Called by run_all_agents.py — not run standalone. Only us_equities_agent's
    output feeds this module in production; macro_agent/international_agent/
    alternatives_agent are excluded from the daily run (see run_all_agents.py)
    but their AgentOutput shape and this module's merge functions still support
    multiple agents, for tests and any future re-measurement/re-enablement.
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from ascent.config.types import AgentOutput

SKILL_SCORES_PATH = Path("dashboard/agent_skill_scores.json")

# Early-zero: agents warming up but clearly losing get zeroed before 63-day threshold.
# Threshold: ≥21 days logged AND negative rolling Sharpe.
# Recovers automatically — rechecked every run, no manual intervention needed.
EARLY_ZERO_MIN_DAYS = 21

# Defensive agents are regime-conditional diversifiers. They are designed to
# underperform in calm_bull — that's not a signal of bad strategy, it's intended
# behavior. Only zero them if they're losing in the regimes they're built for
# (stressed / crisis). In calm_bull, give them a free pass.
DEFENSIVE_AGENTS = {"macro", "alternatives"}
DEFENSIVE_AGENT_REGIMES = {"stressed", "crisis"}  # regimes where early-zero applies

# Factor buckets for partial contradiction detection
FACTOR_BUCKETS = {
    "rates_long":      {"TLT", "IEF", "LQD"},
    "rates_short":     {"HYG", "JNK"},
    "dollar_long":     {"UUP"},
    "commodities":     {"PDBC", "USO", "DBA", "DBB"},
    "gold":            {"GLD", "IAU"},
    "managed_futures": {"KMLM", "DBMF", "CTA", "WTMF"},
    "vol_long":        {"VIXY", "VXX", "UVXY"},
    "vol_short":       {"SVXY", "SVOL"},
    "em_equity":       {"EEM", "VWO", "EWT", "EWZ", "AAXJ", "EWY"},
    "dev_intl_equity": {"EFA", "EWJ", "EWC", "EWG", "EWU", "VEA", "IEFA"},
    "us_tech":         {"QQQ", "XLK", "FTEC"},
    "us_defensive":    {"XLU", "XLP", "XLV", "NEE", "WMT", "MRK"},
    "reits":           {"VNQ", "IYR", "SCHH"},
    "energy":          {"XLE", "MPC", "PSX", "VLO"},
}

FACTOR_CONTRADICTIONS = [
    ("dollar_long",  "commodities",  "USD strength suppresses commodity returns"),
    ("dollar_long",  "gold",         "USD strength suppresses gold"),
    ("dollar_long",  "em_equity",    "USD strength pressures EM equities"),
    ("vol_long",     "vol_short",    "long and short volatility cancel directly"),
    ("rates_long",   "rates_short",  "duration long vs credit risk-on conflict in stress"),
    ("us_tech",      "rates_long",   "tech valuation sensitive to rising rates — directional conflict"),
]

# EM+commodity+managed-futures hard cap
# managed_futures (KMLM etc.) included: they absorb EM/commodity redistribution
# weight and should be subject to the same aggregate cap.
EM_COMMODITY_CAP = 0.20
EM_COMMODITY_BUCKETS = {"em_equity", "commodities", "gold", "managed_futures"}

# International equity hard cap (EM + developed international)
# EWY, EFA, EEM etc. — capped together to limit single-region tail exposure.
INTL_EQUITY_CAP = 0.15
INTL_EQUITY_BUCKETS = {"em_equity", "dev_intl_equity"}


MAX_POSITION_WEIGHT = 0.10  # hard cap per symbol after all blending
MACRO_DIVERGENCE_SCALE = 0.90  # gross exposure reduction when macro says stressed/crisis but equity says calm_bull


def _cap_positions(weights: Dict[str, float], max_w: float = MAX_POSITION_WEIGHT) -> Dict[str, float]:
    """
    Single-pass water-fill cap on the merged portfolio.
    Symbols above max_w are capped; overflow redistributed pro-rata to uncapped names.
    Prevents orchestrator blending from pushing a single name over the per-agent cap.
    """
    capped: Dict[str, float] = {}
    overflow = 0.0
    uncapped_total = 0.0

    for sym, w in weights.items():
        if w > max_w:
            overflow += w - max_w
            capped[sym] = max_w
        else:
            capped[sym] = w
            uncapped_total += w

    if overflow > 0 and uncapped_total > 0:
        for sym in capped:
            if capped[sym] < max_w:
                capped[sym] = round(capped[sym] + overflow * (capped[sym] / uncapped_total), 6)

    total = sum(capped.values())
    if total > 0:
        capped = {s: round(w / total, 6) for s, w in capped.items()}
    return capped


def _cap_em_commodity(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Hard cap: sum of symbols in em_equity + commodities + gold <= 20%.
    Trims EM/commodity proportionally; redistributes freed weight to non-EM.
    """
    em_syms = set()
    for bucket_name in EM_COMMODITY_BUCKETS:
        em_syms.update(FACTOR_BUCKETS.get(bucket_name, set()))

    em_weight    = sum(w for s, w in weights.items() if s in em_syms)
    non_em_syms  = {s for s in weights if s not in em_syms}
    non_em_total = sum(weights.get(s, 0) for s in non_em_syms)

    if em_weight <= EM_COMMODITY_CAP + 1e-6:
        return dict(weights)

    scale  = EM_COMMODITY_CAP / em_weight
    result = {}
    freed  = 0.0
    for sym, w in weights.items():
        if sym in em_syms:
            new_w        = w * scale
            freed       += w - new_w
            result[sym]  = new_w
        else:
            result[sym] = w

    if non_em_total > 0 and freed > 0:
        for sym in non_em_syms:
            result[sym] = result[sym] + freed * (result[sym] / non_em_total)

    total = sum(result.values())
    if total > 0:
        result = {s: round(w / total, 6) for s, w in result.items()}

    return result

def _cap_intl_equity(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Hard cap: sum of symbols in em_equity + dev_intl_equity <= 15%.
    Trims international equity proportionally; redistributes freed weight to domestic.
    """
    intl_syms = set()
    for bucket_name in INTL_EQUITY_BUCKETS:
        intl_syms.update(FACTOR_BUCKETS.get(bucket_name, set()))

    intl_weight    = sum(w for s, w in weights.items() if s in intl_syms)
    non_intl_syms  = {s for s in weights if s not in intl_syms}
    non_intl_total = sum(weights.get(s, 0) for s in non_intl_syms)

    if intl_weight <= INTL_EQUITY_CAP + 1e-6:
        return dict(weights)

    scale  = INTL_EQUITY_CAP / intl_weight
    result = {}
    freed  = 0.0
    for sym, w in weights.items():
        if sym in intl_syms:
            new_w        = w * scale
            freed       += w - new_w
            result[sym]  = new_w
        else:
            result[sym] = w

    if non_intl_total > 0 and freed > 0:
        for sym in non_intl_syms:
            result[sym] = result[sym] + freed * (result[sym] / non_intl_total)

    total = sum(result.values())
    if total > 0:
        result = {s: round(w / total, 6) for s, w in result.items()}

    return result


FACTOR_WEIGHT_THRESHOLD    = 0.08  # agent must have >8% in a bucket to be considered exposed
PARTIAL_CONFLICT_THRESHOLD = 0.06  # both sides must exceed 6% in merged portfolio to trigger

# Conviction parameters
CONVICTION_BONUS_MAX  = 0.15   # max extra allocation boost for high conviction
CONVICTION_MIN_AGENTS = 2      # need at least 2 agents to compute agreement

# Crisis veto parameters
CRISIS_VETO_FLOOR  = 0.40   # macro agent controls at least 40% of merged weights in crisis
CRISIS_VETO_BLEND  = 0.60   # 60% macro, 40% remaining agents in crisis veto

# Base capital allocation (when no skill data or agents warming up).
#
# macro_agent/international_agent scored CUT on their real universes (proof
# audit); alternatives_agent is still unmeasured, excluded pending future
# re-measurement. Only us_equities_agent is invoked from the daily run now
# (see run_all_agents.py), so it is the only key with a nonzero base share.
# The regime-conditional BASE/STRESSED/CRISIS split that used to shift weight
# toward macro/alternatives in stress no longer has other agents to shift
# toward; _compute_allocation() renormalizes to 1.0 regardless of the value
# here as long as it's positive, so a single flat allocation covers all
# regimes.
BASE_ALLOCATION = {"us_equities": 1.0}
STRESSED_ALLOCATION = BASE_ALLOCATION
CRISIS_ALLOCATION = BASE_ALLOCATION


# ── Early-zero check ──────────────────────────────────────────────────────────

def _compute_early_sharpe(agent_id: str) -> Optional[float]:
    """
    Compute rolling Sharpe from the agent's PnL log for the early-zero check.
    Returns None if fewer than EARLY_ZERO_MIN_DAYS entries exist.
    Uses all available entries (not capped at 63) so warming-up agents get a
    fair assessment across their full history.
    """
    from ascent.monitoring.forward_pnl_tracker import PNL_LOGS
    log_path = PNL_LOGS.get(agent_id)
    if not log_path or not log_path.exists():
        return None

    rets = []
    for line in log_path.read_text().splitlines():
        try:
            entry = json.loads(line)
            r = entry.get("return")
            if r is not None:
                rets.append(float(r))
        except Exception:
            pass

    if len(rets) < EARLY_ZERO_MIN_DAYS:
        return None

    mean = sum(rets) / len(rets)
    variance = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    if variance <= 0:
        return 0.0
    return float(mean / variance ** 0.5 * (252 ** 0.5))


def _apply_early_zeros(
    agent_ids: list,
    base: dict,
    current_regime: Optional[str] = None,
) -> tuple:
    """
    Zero out warming-up agents that have ≥ EARLY_ZERO_MIN_DAYS of data
    and a negative rolling Sharpe. Redistribute their base allocation
    proportionally to the remaining agents.

    Returns (zeroed_ids, adjusted_base).
    Agents that recover (Sharpe turns positive) are automatically reinstated
    on the next run — no manual intervention needed.

    Defensive agents (macro, alternatives) are exempt when regime is calm_bull
    or neutral — they are designed to underperform in risk-on regimes.
    """
    zeroed = []
    for aid in agent_ids:
        sharpe = _compute_early_sharpe(aid)
        if sharpe is not None and sharpe <= 0:
            if aid in DEFENSIVE_AGENTS and current_regime not in DEFENSIVE_AGENT_REGIMES:
                print(f"[Orchestrator] Early-zero skipped: {aid} Sharpe={sharpe:.3f} but regime={current_regime} (defensive agent — exempt outside stressed/crisis)")
                continue
            zeroed.append((aid, sharpe))

    if not zeroed:
        return [], base

    zeroed_ids = [aid for aid, _ in zeroed]
    for aid, sharpe in zeroed:
        print(f"[Orchestrator] Early-zero: {aid} Sharpe={sharpe:.3f} (≥{EARLY_ZERO_MIN_DAYS}d, negative) — zeroing allocation")

    # Redistribute freed base weight proportionally to surviving agents
    freed = sum(base.get(aid, 0.0) for aid in zeroed_ids)
    survivors = [aid for aid in agent_ids if aid not in zeroed_ids]
    survivor_total = sum(base.get(aid, 0.0) for aid in survivors)

    adjusted = dict(base)
    for aid in zeroed_ids:
        adjusted[aid] = 0.0
    if survivor_total > 0 and freed > 0:
        for aid in survivors:
            adjusted[aid] = round(adjusted[aid] + freed * (base.get(aid, 0.0) / survivor_total), 4)

    # Renormalize
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

    return zeroed_ids, adjusted


# ── Conviction scorer ─────────────────────────────────────────────────────────

def _compute_conviction_scores(agent_outputs: List[AgentOutput]) -> Dict[str, float]:
    """
    Measure how much agents agree with each other on the same names.
    When multiple agents independently overweight the same symbol,
    that's a conviction signal — not just one agent's opinion.

    Method:
      For each agent, count what fraction of its top holdings (>2% weight)
      appear in at least one other agent's top holdings.
      Agents with high overlap = high conviction = bonus allocation.

    Returns:
      Dict {agent_id: conviction_score} where score is 0.0–1.0.
      0.0 = no overlap with any other agent
      1.0 = all top holdings confirmed by at least one other agent
    """
    if len(agent_outputs) < CONVICTION_MIN_AGENTS:
        return {a.agent_id: 0.0 for a in agent_outputs}

    # Build set of "top holdings" per agent (weight > 2%)
    top_holdings: Dict[str, set] = {}
    for ao in agent_outputs:
        top_holdings[ao.agent_id] = {
            sym for sym, w in ao.target_weights.items() if w > 0.02
        }

    conviction: Dict[str, float] = {}
    for ao in agent_outputs:
        my_tops = top_holdings[ao.agent_id]
        if not my_tops:
            conviction[ao.agent_id] = 0.0
            continue

        # Count how many of my top holdings appear in ANY other agent's top holdings
        other_tops = set()
        for other_id, other_set in top_holdings.items():
            if other_id != ao.agent_id:
                other_tops |= other_set

        overlap = len(my_tops & other_tops)
        conviction[ao.agent_id] = round(overlap / len(my_tops), 4)

    return conviction


# ── Skill scores ───────────────────────────────────────────────────────────────

def _load_skill_scores() -> Dict[str, Optional[float]]:
    """
    Load latest skill scores from the dashboard JSON.
    Returns {} (triggers base allocation) if the file is missing or scores are
    more than 1 day stale — stale scores are worse than no scores.
    The 1-day buffer allows Friday scores to remain valid on Monday.
    """
    if not SKILL_SCORES_PATH.exists():
        return {}
    try:
        with open(SKILL_SCORES_PATH) as f:
            data = json.load(f)

        as_of = data.get("skill_score_as_of", "")
        if as_of:
            today_str = date.today().isoformat()
            staleness_days = (pd.Timestamp(today_str) - pd.Timestamp(as_of)).days
            if staleness_days > 1:
                print(
                    f"[Orchestrator] Skill scores are {staleness_days}d stale "
                    f"(as_of={as_of}) — falling back to base allocation"
                )
                return {}

        agents = data.get("agents", {})
        return {
            agent_id: info.get("skill_score")
            for agent_id, info in agents.items()
        }
    except Exception:
        return {}


# ── Capital allocation ─────────────────────────────────────────────────────────

def _compute_allocation(
    agent_outputs: List[AgentOutput],
    skill_scores: Dict[str, Optional[float]],
    conviction_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute capital allocation across agents.

    Rules (in priority order):
      1. Agents with negative 63-day Sharpe → zero allocation
      2. Agents with no skill data (warming up) → use base allocation
      3. All agents have skill data → 50% skill-weighted + 50% base blend
      4. Conviction bonus — agents with high cross-agent agreement get up to
         CONVICTION_BONUS_MAX extra allocation, taken proportionally from others
      5. Regime override: stressed/crisis → STRESSED/CRISIS_ALLOCATION as base
    """
    agent_ids = [a.agent_id for a in agent_outputs]

    # Regime check — use US equities regime signal
    us_regime = next(
        (a.regime_signal for a in agent_outputs if a.agent_id == "us_equities"),
        None,
    )
    use_crisis   = us_regime == "crisis"
    use_stressed = us_regime == "stressed"

    if use_crisis:
        base = CRISIS_ALLOCATION
    elif use_stressed:
        base = STRESSED_ALLOCATION
    else:
        base = BASE_ALLOCATION

    # Split agents: those with skill data vs warming up (per-agent, not all-or-nothing)
    agents_with_skill = {
        aid: skill_scores[aid]
        for aid in agent_ids
        if skill_scores.get(aid) is not None
    }
    agents_warming_up = [aid for aid in agent_ids if aid not in agents_with_skill]

    if agents_warming_up:
        print(f"[Orchestrator] Warming up (base alloc): {agents_warming_up}")

    if not agents_with_skill:
        # Apply early-zero to warming-up agents with ≥21 days of negative Sharpe.
        # Recovers automatically when Sharpe turns positive.
        early_zeroed, base = _apply_early_zeros(agent_ids, base, current_regime=us_regime)
        allocation = {aid: base.get(aid, 0.0) for aid in agent_ids}
        label = "Base allocation (all warming up)"
        if early_zeroed:
            label += f" [early-zeroed: {early_zeroed}]"
        print(f"[Orchestrator] {label}: {allocation}")
        return allocation

    # Zero out agents with non-positive skill score (within those that have data)
    active_agents = {
        aid: s for aid, s in agents_with_skill.items() if s > 0
    }
    zeroed = [aid for aid in agents_with_skill if aid not in active_agents]
    for aid in zeroed:
        print(f"[Orchestrator] {aid} skill={skill_scores[aid]:.3f} ≤ 0 — zeroing allocation")

    if not active_agents:
        allocation = {aid: base.get(aid, 0.0) * 0.5 for aid in agent_ids}
        print(f"[Orchestrator] All skill-scored agents non-positive — reduced base allocation: {allocation}")
        return allocation

    # Skill-weighted blend (50% skill, 50% base) for agents with data;
    # warming-up agents get their full base allocation directly.
    total_score = sum(active_agents.values())
    allocation  = {}
    for aid in agent_ids:
        if aid in active_agents:
            skill_share     = active_agents[aid] / total_score
            allocation[aid] = round(0.5 * skill_share + 0.5 * base.get(aid, 0.0), 4)
        elif aid in agents_warming_up:
            allocation[aid] = round(base.get(aid, 0.0), 4)
        else:
            allocation[aid] = 0.0  # negative skill score — zeroed

    # Conviction bonus — scale up agents with high cross-agent agreement
    # Bonus pool is taken proportionally from low-conviction agents
    if conviction_scores and any(v > 0 for v in conviction_scores.values()):
        total_conviction = sum(conviction_scores.get(aid, 0.0) for aid in active_agents)
        if total_conviction > 0:
            bonus_pool = 0.0
            bonuses    = {}
            for aid in agent_ids:
                if aid not in active_agents:
                    bonuses[aid] = 0.0
                    continue
                conv  = conviction_scores.get(aid, 0.0)
                bonus = (conv / total_conviction) * CONVICTION_BONUS_MAX
                bonuses[aid]  = round(bonus, 4)
                bonus_pool   += bonus

            # Take bonus pool from low-conviction agents proportionally
            low_conv_total = sum(
                allocation[aid] for aid in active_agents
                if conviction_scores.get(aid, 0.0) < 0.3
            )
            if low_conv_total < 1e-9:
                print(f"[Orchestrator] Conviction bonus: all agents high-conviction — skipping redistribution")
            elif low_conv_total > bonus_pool:
                for aid in agent_ids:
                    conv = conviction_scores.get(aid, 0.0)
                    if aid in active_agents and conv < 0.3:
                        fraction = allocation[aid] / low_conv_total
                        allocation[aid] = round(
                            allocation[aid] - fraction * bonus_pool + bonuses[aid], 4
                        )
                    elif aid in active_agents:
                        allocation[aid] = round(allocation[aid] + bonuses[aid], 4)

                fired_conv = {aid: f"{conviction_scores.get(aid, 0):.0%}"
                              for aid in active_agents}
                print(f"[Orchestrator] Conviction scores: {fired_conv}")
                print(f"[Orchestrator] Conviction bonus pool: {bonus_pool:.3f}")

    # Normalize to 1
    total_alloc = sum(allocation.values())
    if total_alloc > 0:
        allocation = {k: round(v / total_alloc, 4) for k, v in allocation.items()}

    regime_tag = " [CRISIS]" if use_crisis else (" [stressed]" if use_stressed else "")
    print(f"[Orchestrator] Final allocation{regime_tag}: {allocation}")
    return allocation


# ── Weight merging ─────────────────────────────────────────────────────────────

def merge_agent_outputs(
    agent_outputs: List[AgentOutput],
    allocation: Dict[str, float],
) -> Dict[str, float]:
    """
    Merge all agents' target weights into a single portfolio,
    scaled by capital allocation.

    Symbols held by multiple agents get their weights summed.
    Tiny weights (<0.5%) are dropped. Result is renormalized to 1.
    """
    merged: Dict[str, float] = {}

    for output in agent_outputs:
        agent_alloc = allocation.get(output.agent_id, 0.0)
        if agent_alloc <= 0:
            continue
        for sym, w in output.target_weights.items():
            scaled = w * agent_alloc
            merged[sym] = merged.get(sym, 0.0) + scaled

    # Drop tiny weights and round
    merged = {sym: round(w, 6) for sym, w in merged.items() if w > 0.005}

    # Renormalize if needed
    total = sum(merged.values())
    if total > 0 and abs(total - 1.0) > 0.01:
        merged = {sym: round(w / total, 6) for sym, w in merged.items()}

    # Hard cap EM+commodity to 20%
    em_before = sum(w for s, w in merged.items()
                    if any(s in FACTOR_BUCKETS.get(b, set()) for b in EM_COMMODITY_BUCKETS))
    merged = _cap_em_commodity(merged)
    em_after = sum(w for s, w in merged.items()
                   if any(s in FACTOR_BUCKETS.get(b, set()) for b in EM_COMMODITY_BUCKETS))
    if abs(em_before - em_after) > 0.005:
        print(f"[Orchestrator] EM+commodity cap applied: {em_before:.1%} → {em_after:.1%}")

    # Hard cap international equity (EM + developed) to 15%
    intl_before = sum(w for s, w in merged.items()
                      if any(s in FACTOR_BUCKETS.get(b, set()) for b in INTL_EQUITY_BUCKETS))
    merged = _cap_intl_equity(merged)
    intl_after = sum(w for s, w in merged.items()
                     if any(s in FACTOR_BUCKETS.get(b, set()) for b in INTL_EQUITY_BUCKETS))
    if abs(intl_before - intl_after) > 0.005:
        print(f"[Orchestrator] Intl equity cap applied: {intl_before:.1%} → {intl_after:.1%}")

    # Per-position cap — prevent multi-agent overlap pushing any name above max_weight
    overcapped = {s: w for s, w in merged.items() if w > MAX_POSITION_WEIGHT}
    if overcapped:
        print(f"[Orchestrator] Position cap applied: {', '.join(f'{s} {w:.1%}→{MAX_POSITION_WEIGHT:.0%}' for s, w in overcapped.items())}")
        merged = _cap_positions(merged)

    return merged



# ── Crisis regime veto ────────────────────────────────────────────────────────
# NOT CALLED from run_orchestrator() below — macro_agent is no longer invoked
# from the daily run (it scored CUT on its real universe; see run_all_agents.py),
# so this veto's trigger condition (a present macro agent output) can never be
# met in production and calling it would just log "macro agent absent" every
# crisis day. Left defined, not deleted, in case macro_agent is individually
# re-measured and re-added later.

def _apply_crisis_veto(
    merged: Dict[str, float],
    agent_outputs: List[AgentOutput],
    us_regime: Optional[str],
) -> Dict[str, float]:
    """
    In crisis regime, macro agent gets override authority.

    What this does:
      - Takes the macro agent's target weights
      - Forces the merged portfolio to be CRISIS_VETO_BLEND% macro weights
        + (1 - CRISIS_VETO_BLEND)% remaining merged weights
      - Macro agent must be present and have non-empty weights to trigger
      - Logs clearly when veto fires

    This ensures that in a crisis, the agent with the most defensive,
    regime-sensitive positioning has the loudest voice.
    """
    if us_regime != "crisis":
        return merged

    macro_output = next(
        (ao for ao in agent_outputs if ao.agent_id == "macro"),
        None,
    )
    if macro_output is None or not macro_output.target_weights:
        print("[Orchestrator] Crisis veto: macro agent absent — veto not applied")
        return merged

    print(f"[Orchestrator] *** CRISIS VETO FIRING — macro agent override at {CRISIS_VETO_BLEND:.0%} ***")

    macro_w   = macro_output.target_weights
    vetoed    = {}

    all_syms = set(merged) | set(macro_w)
    for sym in all_syms:
        macro_share   = macro_w.get(sym, 0.0)   * CRISIS_VETO_BLEND
        merged_share  = merged.get(sym, 0.0)    * (1.0 - CRISIS_VETO_BLEND)
        vetoed[sym]   = round(macro_share + merged_share, 6)

    # Drop tiny weights and renormalize
    vetoed = {sym: w for sym, w in vetoed.items() if w > 0.005}
    total  = sum(vetoed.values())
    if total > 0:
        vetoed = {sym: round(w / total, 6) for sym, w in vetoed.items()}

    print(f"[Orchestrator] Post-veto portfolio: {len(vetoed)} positions")
    for sym, w in sorted(vetoed.items(), key=lambda x: -x[1])[:10]:
        prev = merged.get(sym, 0.0)
        delta = w - prev
        print(f"  {sym}: {prev:.2%} → {w:.2%} ({'+' if delta >= 0 else ''}{delta:.2%})")

    return vetoed


# ── Thesis coherence check ─────────────────────────────────────────────────────

# Known symbol-level contradictory pairs (direct cancellations)
CONTRADICTORY_PAIRS = [
    ("UUP",  "PDBC", "dollar strength offsets broad commodity returns"),
    ("UUP",  "GLD",  "dollar strength offsets gold returns"),
    ("UUP",  "USO",  "dollar strength offsets oil returns"),
    ("VIXY", "SVXY", "long vol and short vol cancel out"),
    ("TLT",  "HYG",  "duration long vs credit risk-on are offsetting in stress"),
]

CONTRADICTION_THRESHOLD = 0.04


def _get_factor_exposure(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Compute total weight in each factor bucket for a given weight dict.
    Returns {factor_label: total_weight}.
    """
    exposure = {}
    for factor, symbols in FACTOR_BUCKETS.items():
        total = sum(weights.get(sym, 0.0) for sym in symbols)
        if total > 0:
            exposure[factor] = round(total, 4)
    return exposure


def _check_thesis_coherence(merged: dict, agent_outputs: List[AgentOutput] = None) -> dict:
    """
    Detect and resolve contradictory positions in the merged portfolio.

    Two layers:
    1. Symbol-level: hardcoded pairs that directly cancel (VIXY/SVXY etc.)
    2. Factor-level: checks if agents are crowded into conflicting factor
       exposures even through different instruments. E.g. both agents heavy
       in rate-sensitive names during a rates regime — catches the partial
       contradiction the symbol-level check would miss.
    """
    adjusted = dict(merged)
    fired    = []

    # Layer 1 — direct symbol pairs (same as v1)
    for sym_a, sym_b, reason in CONTRADICTORY_PAIRS:
        w_a = adjusted.get(sym_a, 0.0)
        w_b = adjusted.get(sym_b, 0.0)

        if w_a > CONTRADICTION_THRESHOLD and w_b > CONTRADICTION_THRESHOLD:
            if w_a <= w_b:
                cut = adjusted[sym_a] * 0.5
                adjusted[sym_a] = round(adjusted[sym_a] - cut, 6)
            else:
                cut = adjusted[sym_b] * 0.5
                adjusted[sym_b] = round(adjusted[sym_b] - cut, 6)

            fired.append((sym_a, sym_b, reason))
            print(f"[Orchestrator] Coherence L1: {sym_a}/{sym_b} — {reason}")

    # Layer 2 — factor-level partial contradictions
    merged_factors = _get_factor_exposure(adjusted)

    for factor_a, factor_b, reason in FACTOR_CONTRADICTIONS:
        exposure_a = merged_factors.get(factor_a, 0.0)
        exposure_b = merged_factors.get(factor_b, 0.0)

        if exposure_a > PARTIAL_CONFLICT_THRESHOLD and exposure_b > PARTIAL_CONFLICT_THRESHOLD:
            # Reduce the smaller factor exposure by 40% across its constituent symbols
            if exposure_a <= exposure_b:
                target_factor, target_exposure = factor_a, exposure_a
            else:
                target_factor, target_exposure = factor_b, exposure_b

            reduce_by = 0.40
            for sym in FACTOR_BUCKETS.get(target_factor, set()):
                if sym in adjusted:
                    adjusted[sym] = round(adjusted[sym] * (1 - reduce_by), 6)

            print(
                f"[Orchestrator] Coherence L2: {factor_a} ({exposure_a:.1%}) vs "
                f"{factor_b} ({exposure_b:.1%}) — {reason}"
            )
            fired.append((factor_a, factor_b, reason))
            # Refresh factor exposures after adjustment
            merged_factors = _get_factor_exposure(adjusted)

    if fired:
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {sym: round(w / total, 6) for sym, w in adjusted.items()}
        adjusted = {sym: w for sym, w in adjusted.items() if w > 0.005}

    return adjusted

# ── Main entry point ───────────────────────────────────────────────────────────

def run_orchestrator(agent_outputs: List[AgentOutput]) -> Dict[str, float]:
    """
    Main entry point.

    Args:
        agent_outputs: List of AgentOutput from all specialist agents

    Returns:
        Merged target weight vector {symbol: weight}
    """
    print(f"\n{'='*60}")
    print(f"[Orchestrator] Central Intelligence v2")
    print(f"{'='*60}")
    print(f"[Orchestrator] Received {len(agent_outputs)} agent outputs:")
    for ao in agent_outputs:
        print(f"  {ao.summary()}")

    # Load skill scores
    skill_scores = _load_skill_scores()
    if skill_scores:
        print(f"[Orchestrator] Loaded skill scores: { {k: f'{v:.3f}' if v else 'N/A' for k,v in skill_scores.items()} }")
    else:
        print("[Orchestrator] No skill scores found — using base allocation")

    # Conviction scores — cross-agent agreement on names
    conviction_scores = _compute_conviction_scores(agent_outputs)
    if any(v > 0 for v in conviction_scores.values()):
        print(f"[Orchestrator] Conviction scores: { {k: f'{v:.0%}' for k,v in conviction_scores.items()} }")

    # Compute allocation (skill + conviction weighted)
    allocation = _compute_allocation(agent_outputs, skill_scores, conviction_scores)

    # Merge weights
    merged = merge_agent_outputs(agent_outputs, allocation)

    # Correlation guard
    try:
        from ascent.risk.correlation_guard import check_cross_agent_correlation, apply_correlation_adjustments
        agent_weights = {ao.agent_id: ao.target_weights for ao in agent_outputs if ao.target_weights}
        if len(agent_weights) >= 2:
            violations = check_cross_agent_correlation(agent_weights)
            if violations:
                merged = apply_correlation_adjustments(merged, violations)
    except Exception as _cg_err:
        print(f"[Orchestrator] Correlation guard failed ({_cg_err}) — skipping")

    # Thesis coherence — symbol-level + factor-level partial contradictions
    merged = _check_thesis_coherence(merged, agent_outputs)

    # Crisis veto (_apply_crisis_veto) is intentionally not called here — its
    # trigger (a present macro agent output) can never fire now that macro_agent
    # is excluded from the daily run. See the function's module comment above.
    us_regime = next(
        (ao.regime_signal for ao in agent_outputs if ao.agent_id == "us_equities"),
        None,
    )

    # Final position cap — correlation guard and thesis coherence can push names
    # above MAX_POSITION_WEIGHT through renormalization; re-cap before returning.
    overcapped_final = {s: w for s, w in merged.items() if w > MAX_POSITION_WEIGHT}
    if overcapped_final:
        print(f"[Orchestrator] Final position cap: {', '.join(f'{s} {w:.1%}→{MAX_POSITION_WEIGHT:.0%}' for s, w in overcapped_final.items())}")
        merged = _cap_positions(merged)

    # Macro-equity regime divergence: when macro says stressed/crisis but equity says
    # calm_bull, reduce gross exposure — macro regime tends to lead equity in stress.
    # Weights intentionally sum to <1.0 (implicit cash); do NOT renormalize.
    macro_regime = next(
        (ao.regime_signal for ao in agent_outputs if ao.agent_id == "macro"),
        None,
    )
    if (macro_regime in {"stressed", "crisis"}
            and us_regime not in {"stressed", "crisis"}):
        merged = {sym: round(w * MACRO_DIVERGENCE_SCALE, 6) for sym, w in merged.items()}
        print(
            f"[Orchestrator] Macro-equity divergence: macro={macro_regime}, "
            f"equity={us_regime} — scaling to {MACRO_DIVERGENCE_SCALE:.0%} gross exposure"
        )

    # Print summary
    print(f"\n[Orchestrator] Final portfolio — {len(merged)} positions:")
    for sym, w in sorted(merged.items(), key=lambda x: -x[1])[:20]:
        print(f"  {sym}: {w:.2%}")

    total = sum(merged.values())
    print(f"[Orchestrator] Total weight: {total:.4f}")

    return merged
