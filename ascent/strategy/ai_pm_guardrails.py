# ascent/strategy/ai_pm_guardrails.py
"""
Level-specific guardrails for AI PM portfolio proposals.

apply_guardrails() used to be called after Phase 2 completes, before
authority_blend(). It was confirmed to have zero non-test callers (the sole
non-test reference in agents/ai_pm_agent.py imports check_conviction_inflation
from this module instead, a different function) and was deleted 2026-08-16
along with its private helpers (_rolling_corr, _apply_tracking_error_cap) and
the level-config table (_LEVEL_CONFIG) that only it consumed.

check_conviction_inflation() and is_valuation_short() below are unrelated,
confirmed-live functions — not touched.
"""
from __future__ import annotations
import logging
from typing import Dict

log = logging.getLogger(__name__)


def check_conviction_inflation(proposals: Dict[str, str]) -> Dict[str, str]:
    """
    If >40% of names are 'high conviction', downgrade the excess to 'medium'.
    A model that marks everything high-conviction has no model of risk.
    """
    if not proposals:
        return proposals
    high_count = sum(1 for v in proposals.values() if v == "high")
    threshold  = max(1, int(len(proposals) * 0.40))
    if high_count <= threshold:
        return proposals
    # Keep the first N high-conviction names, downgrade the rest
    high_syms = [s for s, v in proposals.items() if v == "high"][:threshold]
    return {
        s: v if (v != "high" or s in high_syms) else "medium"
        for s, v in proposals.items()
    }


def is_valuation_short(reason: str) -> bool:
    """
    Detect valuation-based short rationale — explicitly banned per spec constraint 41.
    Returns True if the reason cites P/E, EV/EBITDA, P/B, or price-to-any-multiple.
    """
    banned_patterns = [
        "p/e", "pe ratio", "price-to-earnings", "price to earnings",
        "ev/ebitda", "ev / ebitda", "enterprise value",
        "p/b", "price-to-book", "price to book",
        "overvalued", "too expensive", "stretched valuation",
        "valuation is rich", "trading at a premium",
    ]
    reason_lower = reason.lower()
    return any(p in reason_lower for p in banned_patterns)
