"""
ascent/monitoring/ai_pm_eval_rule.py

PRE-REGISTERED AI PM evaluation rule.

This module exists to make the "is the AI PM adding value?" decision MECHANICAL
and committed to IN ADVANCE, so it cannot be rationalised after the fact. See
docs/AI_PM_EVAL_RULE.md for the full pre-registration and the reasoning.

Two non-negotiable design choices, both about not fooling ourselves:

  1. The unit of evidence is an INDEPENDENT DECISION (one rebalance), never a
     daily mark-to-market of the same book. Marking one 19-name book every day
     for 23 days produces 23 highly autocorrelated numbers, not 23 facts. A
     t-test on daily marks would claim significance that isn't there.

  2. Below MIN_DECISIONS independent decisions, the verdict is ALWAYS HOLD,
     regardless of how good or bad the running mean looks. With n=3 you have a
     coin flipped three times. This is the exact trap (acting on a tiny sample)
     that WFE -0.65 and the -6.52pp/n=1 AI-PM figure both represent.

The rule is deliberately simple and falsifiable:
    mean of per-decision (Track D - Track A*) differences, one-sample t-stat.
      t  >  +T_THRESHOLD  and  n >= MIN_DECISIONS  -> PROMOTE
      t  <  -T_THRESHOLD  and  n >= MIN_DECISIONS  -> DEMOTE
      otherwise                                    -> HOLD
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

# ── Pre-registered constants (do not tune to fit an outcome) ─────────────────────
# Minimum number of INDEPENDENT decisions before any promote/demote verdict.
MIN_DECISIONS: int = 20
# Two-sided t-stat magnitude required to call the mean different from zero.
T_THRESHOLD: float = 2.0

_REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_LOG = _REPO_ROOT / "logs" / "ai_pm_calibration.jsonl"


def evaluate_rule(
    decision_diffs: Sequence[float],
    min_decisions: int = MIN_DECISIONS,
    t_threshold: float = T_THRESHOLD,
) -> Dict:
    """Apply the pre-registered rule to a list of per-decision (D - A*) diffs.

    `decision_diffs` MUST be one number per independent decision (per rebalance),
    not per day. Returns a verdict dict; never raises on normal input.
    """
    diffs = [float(x) for x in decision_diffs if x is not None]
    n = len(diffs)

    base = {
        "n": n,
        "min_decisions": min_decisions,
        "t_threshold": t_threshold,
        "mean_diff": None,
        "t_stat": None,
    }

    if n == 0:
        return {**base, "verdict": "HOLD", "reason": "no resolved decisions yet"}

    mean_diff = sum(diffs) / n

    if n < min_decisions:
        return {
            **base,
            "mean_diff": mean_diff,
            "verdict": "HOLD",
            "reason": (
                f"insufficient sample: {n} independent decision(s) < "
                f"{min_decisions} required. Mean looks "
                f"{'positive' if mean_diff >= 0 else 'negative'} "
                f"({mean_diff*100:+.2f}pp/decision) but is not yet evidence."
            ),
        }

    # One-sample t-stat vs 0. Need variance; a single decision can't have one.
    if n < 2:
        return {**base, "mean_diff": mean_diff, "verdict": "HOLD",
                "reason": "need >= 2 decisions to estimate variance"}

    var = sum((x - mean_diff) ** 2 for x in diffs) / (n - 1)
    std = math.sqrt(var)
    if std == 0.0:
        # Perfectly consistent sign with zero variance -> maximally significant.
        t_stat = math.inf if mean_diff > 0 else (-math.inf if mean_diff < 0 else 0.0)
    else:
        t_stat = mean_diff / (std / math.sqrt(n))

    out = {**base, "mean_diff": mean_diff, "t_stat": t_stat}

    if t_stat > t_threshold:
        out["verdict"] = "PROMOTE"
        out["reason"] = (f"AI book beat the quant counterfactual by "
                         f"{mean_diff*100:+.2f}pp/decision over {n} decisions "
                         f"(t={t_stat:.2f} > {t_threshold}).")
    elif t_stat < -t_threshold:
        out["verdict"] = "DEMOTE"
        out["reason"] = (f"AI book trailed the quant counterfactual by "
                         f"{mean_diff*100:+.2f}pp/decision over {n} decisions "
                         f"(t={t_stat:.2f} < -{t_threshold}).")
    else:
        out["verdict"] = "HOLD"
        out["reason"] = (f"{n} decisions but not significant "
                         f"(t={t_stat:.2f}, |t|<{t_threshold}). No edge proven either way.")
    return out


def count_independent_decisions(log_path: Optional[Path] = None) -> int:
    """Count INDEPENDENT, resolved AI PM decisions in the calibration log.

    One decision = one distinct date that has a realized 21-day portfolio outcome.
    Deliberately date-keyed so that the same book logged on consecutive days (or a
    daily mark) collapses to the decisions it actually represents. Never raises.
    """
    path = log_path or CALIBRATION_LOG
    if not path.exists():
        return 0
    dates = set()
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e.get("realized_21d_portfolio") is not None:
                dates.add(e.get("date"))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("[AIPMEvalRule] count_independent_decisions failed: %s", exc)
        return 0
    return len(dates)


def load_decision_diffs(log_path: Optional[Path] = None) -> List[float]:
    """Per-decision (portfolio realized 21d) values from the calibration log.

    NOTE: this returns the AI book's realized return per decision. The A* (quant
    counterfactual) per-decision series is not yet persisted per-decision; until
    it is, feed `evaluate_rule` paired diffs from a decision-level counterfactual.
    This loader is provided so `count_independent_decisions` and dashboards share
    one definition of "a decision". Never raises.
    """
    path = log_path or CALIBRATION_LOG
    if not path.exists():
        return []
    out: List[float] = []
    seen = set()
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            d = e.get("date")
            r = e.get("realized_21d_portfolio")
            if r is not None and d not in seen:
                seen.add(d)
                out.append(float(r))
    except Exception:
        return out
    return out
