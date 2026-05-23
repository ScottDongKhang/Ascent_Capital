# ascent/strategy/conviction_gate.py
"""
Conviction gate — data-driven override approval.

Replaces the static "calibration warning in prompt" with a dynamic gate that reads
the AI PM's actual historical win rate per override type and regime, then recommends
whether to proceed, reduce, or block.

Rules-based now. ML-ready interface: when n_cases >= MIN_ML_CASES, trains a logistic
regression on the decision memory features and uses model probability instead of
heuristic rules. The interface to the caller never changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MIN_ML_CASES = 30        # train logistic regression once we have this many matured outcomes
BLOCK_WIN_RATE = 0.35    # block if win_rate below this AND n_cases >= MIN_BLOCK_CASES
MIN_BLOCK_CASES = 8      # need at least this many cases to block an override type
REDUCE_WIN_RATE = 0.50   # reduce size if win_rate between BLOCK and REDUCE
STRONG_WIN_RATE = 0.60   # full conviction above this


@dataclass
class GateResult:
    proceed: bool
    size_multiplier: float   # 1.0 = full size, 0.7 = reduce 30%, 0.0 = block
    confidence: str          # "strong" | "proceed" | "caution" | "block"
    reason: str
    n_cases: int
    win_rate: Optional[float]


def evaluate(
    override_type: str,
    regime: str,
    calibration_ic: Optional[float] = None,
    log_path: Optional[Path] = None,
) -> GateResult:
    """
    Evaluate whether the AI PM should proceed with an override of the given type.

    Args:
        override_type:    One of data_quality / regime_macro / news_event /
                          correlation_risk / valuation
        regime:           Current regime label (e.g. 'calm_bull')
        calibration_ic:   Spearman IC from calibration tracker (optional)
        log_path:         Path to decision_memory.jsonl (uses default if None)

    Returns:
        GateResult with proceed flag, size multiplier, and reasoning.
    """
    from ascent.memory.decision_memory import get_statistics

    kwargs = {} if log_path is None else {"log_path": log_path}
    stats = get_statistics(override_type=override_type, regime=regime, **kwargs)
    n     = stats["n_cases"]
    wr    = stats["win_rate"]
    aw    = stats["avg_wedge"]

    # data_quality and correlation_risk are always approved — these are the AI PM's
    # clearest edges and hardest to have bad historical track records on
    if override_type in ("data_quality", "correlation_risk", "news_event"):
        return GateResult(
            proceed=True, size_multiplier=1.0, confidence="proceed",
            reason=f"{override_type} overrides are always approved — structural AI PM edge",
            n_cases=n, win_rate=wr,
        )

    # valuation overrides: require calibration IC AND strong historical win rate
    if override_type == "valuation":
        if calibration_ic is not None and calibration_ic < 0.10:
            return GateResult(
                proceed=False, size_multiplier=0.0, confidence="block",
                reason=f"Valuation overrides blocked: calibration IC={calibration_ic:.3f} < 0.10 threshold",
                n_cases=n, win_rate=wr,
            )
        if n >= MIN_BLOCK_CASES and wr is not None and wr < BLOCK_WIN_RATE:
            return GateResult(
                proceed=False, size_multiplier=0.0, confidence="block",
                reason=f"Valuation overrides blocked: historical win rate {wr:.0%} across {n} cases",
                n_cases=n, win_rate=wr,
            )

    # Insufficient data — proceed with caution
    if n < 5:
        return GateResult(
            proceed=True, size_multiplier=0.85, confidence="caution",
            reason=f"Only {n} historical case(s) for [{override_type}/{regime}] — building track record, reduce 15%",
            n_cases=n, win_rate=wr,
        )

    # Strong track record
    if wr >= STRONG_WIN_RATE and aw is not None and aw > 0:
        return GateResult(
            proceed=True, size_multiplier=1.0, confidence="strong",
            reason=f"Strong track record: {wr:.0%} win rate, {aw:+.2%} avg wedge across {n} cases",
            n_cases=n, win_rate=wr,
        )

    # Mixed track record
    if wr >= REDUCE_WIN_RATE:
        return GateResult(
            proceed=True, size_multiplier=0.75, confidence="proceed",
            reason=f"Mixed track record: {wr:.0%} win rate across {n} cases — reduce size 25%",
            n_cases=n, win_rate=wr,
        )

    # Poor track record — block if enough evidence, otherwise reduce
    if n >= MIN_BLOCK_CASES and wr < BLOCK_WIN_RATE:
        return GateResult(
            proceed=False, size_multiplier=0.0, confidence="block",
            reason=f"Poor track record: {wr:.0%} win rate across {n} cases — override blocked",
            n_cases=n, win_rate=wr,
        )

    return GateResult(
        proceed=True, size_multiplier=0.60, confidence="caution",
        reason=f"Below-average track record: {wr:.0%} win rate across {n} cases — reduce size 40%",
        n_cases=n, win_rate=wr,
    )


def format_gate_result(result: GateResult) -> str:
    """Return a concise string for the AI PM tool."""
    status = "✓ PROCEED" if result.proceed else "✗ BLOCKED"
    size   = f"  Size multiplier: {result.size_multiplier:.0%}" if result.proceed else ""
    return (
        f"{status} [{result.confidence.upper()}]{size}\n"
        f"  Reason: {result.reason}\n"
        f"  Historical: {result.n_cases} cases, "
        f"win rate {f'{result.win_rate:.0%}' if result.win_rate is not None else 'n/a'}"
    )
