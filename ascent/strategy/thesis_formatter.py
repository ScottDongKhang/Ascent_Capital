# ascent/strategy/thesis_formatter.py
"""Thesis formatter — converts AI PM raw output to investment memo JSON + plaintext."""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "ai_pm_theses"

_SCHEMA_DEFAULTS = {
    "market_view": "",
    "regime_assessment": "",
    "quant_baseline_summary": "",
    "ai_pm_portfolio": {},
    "quant_agreement": [],
    "quant_overrides": [],
    "position_rationale": {},
    "key_risks": [],
    "what_could_be_wrong": "",
}


def format_thesis(raw_thesis: dict, as_of_date: Optional[date] = None) -> dict:
    """
    Validate and serialize full investment memo JSON.
    Missing fields are filled with schema defaults.
    Saves to outputs/ai_pm_theses/YYYY-MM-DD-thesis.json.
    Returns the filled thesis dict.
    """
    if as_of_date is None:
        as_of_date = date.today()

    thesis = {**_SCHEMA_DEFAULTS}
    thesis.update({k: v for k, v in raw_thesis.items() if k in _SCHEMA_DEFAULTS})
    thesis["as_of_date"] = str(as_of_date)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{as_of_date}-thesis.json"
    try:
        out_path.write_text(json.dumps(thesis, indent=2, default=str))
    except Exception as exc:
        log.warning("[ThesisFormatter] Could not save thesis to %s: %s", out_path, exc)

    return thesis


def thesis_to_plaintext(thesis: dict) -> str:
    """3-4 sentence narrative summary for investor reports. Never raises."""
    parts = []

    market_view = thesis.get("market_view", "")
    if market_view:
        parts.append(market_view.strip().rstrip(".") + ".")

    regime = thesis.get("regime_assessment", "")
    n_pos = len(thesis.get("ai_pm_portfolio", {}))
    if regime and n_pos:
        parts.append(f"Given {regime}, the AI PM constructed a {n_pos}-position portfolio.")

    agreements = thesis.get("quant_agreement", [])
    overrides = thesis.get("quant_overrides", [])
    if agreements or overrides:
        parts.append(
            f"The AI PM agreed with {len(agreements)} quant recommendations "
            f"and overrode {len(overrides)}."
        )

    risks = thesis.get("key_risks", [])
    if risks:
        parts.append(f"Key risks: {'; '.join(risks[:3])}.")

    return " ".join(parts) if parts else "No thesis available."
