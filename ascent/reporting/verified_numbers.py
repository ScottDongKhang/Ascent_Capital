"""The one path from artifact to a citable number.

Every performance figure this project states in a document, on the published
page, or in a promotion decision must come through here. The rule exists
because it has been broken expensively: a Sharpe of 0.518 was published as
"the rigorous figure" while matching no artifact in the repo, a synthetic
drawdown was reported as fact, and six different max-drawdown values were in
circulation at once for a single book.

Design rules:

- Read the artifact. Never hardcode a metric.
- If the artifact is missing or malformed, raise or return None. Never
  substitute a plausible default -- a fabricated baseline that silently gates a
  promotion decision is worse than a crash.
- Carry provenance with the value. A number without its artifact path is not
  citable.

Usage:

    from ascent.reporting.verified_numbers import canonical_wf

    wf = canonical_wf()
    print(wf.sharpe, wf.artifact)      # 0.4118  outputs/wf_results/wf_report_clean_2026-06-22.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent

#: The verified walk-forward run. Superseding this pointer is a deliberate act:
#: update it only alongside CURRENT_VERIFIED_NUMBERS.md, and only for a run on a
#: clean single-source cache whose figures have been independently recomputed.
CANONICAL_WF_ARTIFACT = "outputs/wf_results/wf_report_clean_2026-08-15.json"


class MissingArtifact(RuntimeError):
    """Raised when a number is requested but its artifact is unavailable."""


@dataclass(frozen=True)
class WalkForwardRecord:
    """Verified out-of-sample figures, with the artifact they came from."""

    artifact: str
    sharpe: float
    cagr: float
    max_drawdown: float
    beta: float
    alpha: float
    win_rate: float
    wfe: Optional[float]
    volatility: float
    n_folds: int
    n_oos_days: int
    oos_window: str
    n_symbols: Optional[int] = None
    spy_cagr_same_window: Optional[float] = None
    excess_cagr_vs_spy: Optional[float] = None
    caveats: tuple = field(default_factory=tuple)

    @property
    def sharpe_str(self) -> str:
        return f"{self.sharpe:.2f}"

    @property
    def cagr_pct_str(self) -> str:
        return f"{self.cagr * 100:+.1f}%"

    @property
    def max_dd_pct_str(self) -> str:
        return f"{self.max_drawdown * 100:.1f}%"

    def citation(self) -> str:
        """A one-line, self-provenancing summary safe to paste into a doc."""
        return (f"Sharpe {self.sharpe_str}, CAGR {self.cagr_pct_str}, "
                f"max DD {self.max_dd_pct_str}, beta {self.beta:.2f} "
                f"(OOS {self.oos_window}, {self.n_folds} folds, "
                f"{self.n_oos_days} days) [{self.artifact}]")


def load_wf_report(rel: str = CANONICAL_WF_ARTIFACT) -> WalkForwardRecord:
    """Load a walk-forward artifact into a record. Raises if unavailable.

    Deliberately raises rather than returning a default: a caller that cannot
    read the artifact must not be handed a number it could mistake for one.
    """
    path = ROOT / rel
    if not path.exists():
        raise MissingArtifact(f"walk-forward artifact not found: {rel}")
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MissingArtifact(f"{rel} is not valid JSON: {exc}") from exc

    required = ("sharpe", "cagr", "max_drawdown", "beta", "alpha",
                "win_rate", "wfe", "volatility", "n_folds", "n_oos_days")
    missing = [k for k in required if k not in d]
    if missing:
        raise MissingArtifact(f"{rel} is missing fields: {', '.join(missing)}")

    meta = d.get("_meta", {}) or {}

    wfe_raw = d.get("wfe")
    wfe = float(wfe_raw) if wfe_raw is not None else None

    caveats = []
    if wfe is not None and wfe < 0:
        caveats.append("WFE is negative: the in-sample optimizer adds no "
                       "out-of-sample value. Disclose as overfit.")
    elif wfe is None:
        caveats.append("WFE not computed for this artifact: its producing "
                       "framework (ascent/research/walk_forward_runner.py) "
                       "does not track per-fold in-sample Sharpe, unlike the "
                       "retired wf_framework/ pipeline. Do not report a WFE "
                       "figure for this run.")
    if meta.get("llm_disabled"):
        caveats.append("LLM-driven sleeves were zeroed for this run.")

    return WalkForwardRecord(
        artifact=rel,
        sharpe=float(d["sharpe"]),
        cagr=float(d["cagr"]),
        max_drawdown=float(d["max_drawdown"]),
        beta=float(d["beta"]),
        alpha=float(d["alpha"]),
        win_rate=float(d["win_rate"]),
        wfe=wfe,
        volatility=float(d["volatility"]),
        n_folds=int(d["n_folds"]),
        n_oos_days=int(d["n_oos_days"]),
        oos_window=str(meta.get("oos_window", "unknown")),
        n_symbols=meta.get("n_symbols"),
        spy_cagr_same_window=meta.get("spy_cagr_same_window"),
        excess_cagr_vs_spy=meta.get("excess_cagr_vs_spy"),
        caveats=tuple(caveats),
    )


def canonical_wf() -> WalkForwardRecord:
    """The verified walk-forward record. Raises MissingArtifact if unreadable."""
    return load_wf_report(CANONICAL_WF_ARTIFACT)


def canonical_wf_sharpe() -> Optional[float]:
    """Verified OOS Sharpe, or None if the artifact is unavailable.

    Returns None rather than raising for callers that have a legitimate
    no-number path (they must then decline to act, not invent a value).
    """
    try:
        return canonical_wf().sharpe
    except MissingArtifact:
        return None
