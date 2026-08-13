"""Verdict rule and scorecard I/O. The only place KEEP/CUT/INSUFFICIENT_DATA is decided."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from ascent.analyst.proof_audit.stats import ICResult

SIGNIFICANCE_P = 0.05
DEFAULT_MIN_SAMPLE = 30


def verdict(result: ICResult, min_sample: int = DEFAULT_MIN_SAMPLE) -> str:
    """Three-way, never a silent default.

    INSUFFICIENT_DATA takes priority over the significance test -- a small sample that happens
    to look significant is not trustworthy enough to KEEP or CUT on.
    """
    if result.n < min_sample:
        return "INSUFFICIENT_DATA"
    if result.p_value < SIGNIFICANCE_P and result.ic_mean > 0:
        return "KEEP"
    return "CUT"


@dataclass(frozen=True)
class ScorecardRow:
    component: str
    kind: str
    method: str
    metric: float | None
    p_value: float | None
    sample_size: int
    verdict: str
    # Why this row is not a clean KEEP/CUT off a full, trustworthy measurement: the deferral,
    # the missing input, the approximation, or the guard that fired. None means the number
    # stands on its own. A reader of the JSON alone must never have to guess.
    reason: str | None = None


def write_scorecard(rows: list[ScorecardRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [dataclasses.asdict(r) for r in rows]
    out_path.write_text(json.dumps(payload, indent=2))
