# ascent/strategy/ai_pm_guardrails.py
"""
Level-specific guardrails for AI PM portfolio proposals.

apply_guardrails() is called after Phase 2 completes, before authority_blend().
Violations are logged but never raise — the guardrail always returns a valid portfolio.

Short-selling rules (constraints 38–44 from spec):
- Levels 1–2: no shorts (negative weights) permitted
- Level 3+: shorts allowed only when LONG_SHORT_ENABLED=True and all 5 gates pass
- Valuation-based shorts explicitly rejected
"""
from __future__ import annotations
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Per-level guardrail config
_LEVEL_CONFIG = {
    1: {
        "max_change":      0.02,
        "max_new":         0,
        "allowed_types":   {"AMPLIFY"},
        "max_overrides":   2,
        "max_te":          0.003,   # 0.3% daily tracking error
        "allow_shorts":    False,
        "max_short_single": 0.0,
        "max_short_gross": 0.0,
    },
    2: {
        "max_change":      0.04,
        "max_new":         1,
        "allowed_types":   {"AMPLIFY", "HOLD"},
        "max_overrides":   3,
        "max_te":          0.005,
        "allow_shorts":    False,
        "max_short_single": 0.0,
        "max_short_gross": 0.0,
    },
    3: {
        "max_change":      0.06,
        "max_new":         2,
        "allowed_types":   {"AMPLIFY", "HOLD", "REDUCE"},
        "max_overrides":   4,
        "max_te":          0.008,
        "allow_shorts":    True,
        "max_short_single": 0.03,  # 3% per name
        "max_short_gross":  0.10,  # 10% gross short
    },
    4: {
        "max_change":      0.08,
        "max_new":         3,
        "allowed_types":   {"AMPLIFY", "HOLD", "REDUCE", "NEW"},
        "max_overrides":   5,
        "max_te":          0.012,
        "allow_shorts":    True,
        "max_short_single": 0.05,
        "max_short_gross":  0.20,
    },
    5: {
        "max_change":      0.10,
        "max_new":         5,
        "allowed_types":   {"AMPLIFY", "HOLD", "REDUCE", "NEW"},
        "max_overrides":   999,
        "max_te":          1.0,
        "allow_shorts":    True,
        "max_short_single": 0.07,
        "max_short_gross":  0.30,
    },
}

_CORR_BLOCK_THRESHOLD = 0.65
_ALPHA_QUALITY_PERCENTILE = 0.50   # Level 1: only top 50% alpha names
_VAR_PROXY = 0.0004                 # ~2% daily vol per name, diagonal covariance


def apply_guardrails(
    ai_pm_portfolio: Dict[str, float],
    quant_portfolio: Dict[str, float],
    quant_alpha_scores: Dict[str, float],
    level: int,
    price_returns_63d: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Apply level-specific guardrails to ai_pm_portfolio.
    Returns (filtered_portfolio, violations_log).
    Never raises — always returns a valid portfolio.
    """
    if level not in _LEVEL_CONFIG:
        return dict(quant_portfolio), [f"Unknown level {level} — falling back to quant"]

    cfg        = _LEVEL_CONFIG[level]
    violations: List[str] = []
    result     = dict(quant_portfolio)  # start from quant, apply approved AI PM changes

    # Check if long-short is enabled in execution layer
    long_short_enabled = os.environ.get("LONG_SHORT_ENABLED", "false").lower() == "true"

    # Compute alpha score median for quality gate
    alpha_vals   = sorted(quant_alpha_scores.values()) if quant_alpha_scores else [0.0]
    median_alpha = alpha_vals[len(alpha_vals) // 2] if alpha_vals else 0.0

    quant_syms       = set(quant_portfolio.keys())
    overrides_applied = 0

    for sym, ai_w in ai_pm_portfolio.items():
        quant_w = quant_portfolio.get(sym, 0.0)
        delta   = ai_w - quant_w

        # ── Classify override type ──────────────────────────────────────────
        if ai_w < 0:
            ov_type = "SHORT"
        elif sym not in quant_syms:
            ov_type = "NEW"
        elif delta > 0.001:
            ov_type = "AMPLIFY"
        elif delta < -0.001:
            ov_type = "REDUCE"
        else:
            continue  # HOLD — no meaningful change, skip

        # ── Short-specific gates ────────────────────────────────────────────
        if ov_type == "SHORT":
            if not cfg["allow_shorts"]:
                violations.append(f"{sym}: SHORT not permitted at Level {level} — earn long-only track record first")
                continue
            if not long_short_enabled:
                violations.append(f"{sym}: SHORT blocked — LONG_SHORT_ENABLED=False in execution layer")
                continue
            if sym in quant_syms and quant_portfolio.get(sym, 0.0) > 0:
                violations.append(f"{sym}: SHORT blocked — quant holds this name long; contradictory positions not permitted")
                continue
            short_size = abs(ai_w)
            if short_size > cfg["max_short_single"]:
                ai_w = -cfg["max_short_single"]
                violations.append(f"{sym}: SHORT capped at {cfg['max_short_single']:.0%} (proposed {short_size:.1%})")
            # Check gross short exposure
            current_gross_short = sum(abs(v) for v in result.values() if v < 0)
            if current_gross_short + abs(ai_w) > cfg["max_short_gross"]:
                violations.append(f"{sym}: SHORT blocked — gross short {current_gross_short:.0%} would exceed {cfg['max_short_gross']:.0%} cap")
                continue
            result[sym] = ai_w
            overrides_applied += 1
            continue

        # ── Standard long override gates ────────────────────────────────────
        if ov_type not in cfg["allowed_types"]:
            violations.append(f"{sym}: {ov_type} not allowed at Level {level}")
            continue

        # Level 1–2: amplification quality — only top 50% alpha names
        if level <= 2 and ov_type == "AMPLIFY":
            score = quant_alpha_scores.get(sym, 0.0)
            if score < median_alpha:
                violations.append(f"{sym}: AMPLIFY blocked — alpha {score:.3f} below median {median_alpha:.3f}")
                continue

        # Max weight change cap
        capped_delta = max(-cfg["max_change"], min(cfg["max_change"], delta))
        if abs(capped_delta - delta) > 0.0005:
            violations.append(
                f"{sym}: weight change capped {delta:+.3f}→{capped_delta:+.3f} "
                f"(Level {level} max ±{cfg['max_change']:.0%})"
            )
            delta = capped_delta

        # New symbol cap
        if ov_type == "NEW":
            new_so_far = sum(1 for s in result if s not in quant_syms and result[s] != quant_portfolio.get(s, 0.0))
            if new_so_far >= cfg["max_new"]:
                violations.append(f"{sym}: new symbol blocked (max {cfg['max_new']} at Level {level})")
                continue

        # Override correlation check (if price history available)
        if price_returns_63d and overrides_applied > 0:
            existing = [s for s in result if abs(result[s] - quant_portfolio.get(s, 0.0)) > 0.001]
            corr_blocked = False
            for prev_sym in existing[:3]:
                corr = _rolling_corr(
                    price_returns_63d.get(sym, []),
                    price_returns_63d.get(prev_sym, []),
                )
                if corr > _CORR_BLOCK_THRESHOLD:
                    violations.append(f"{sym}: corr {corr:.2f} with {prev_sym} > {_CORR_BLOCK_THRESHOLD} — blocked")
                    corr_blocked = True
                    break
            if corr_blocked:
                continue

        # Max overrides cap
        if overrides_applied >= cfg["max_overrides"]:
            violations.append(f"{sym}: max overrides ({cfg['max_overrides']}) reached at Level {level}")
            continue

        result[sym] = quant_w + delta
        overrides_applied += 1

    # ── Tracking error cap ──────────────────────────────────────────────────
    result, te_violations = _apply_tracking_error_cap(result, quant_portfolio, cfg["max_te"])
    violations.extend(te_violations)

    return result, violations


def _rolling_corr(a: List[float], b: List[float]) -> float:
    """63-day rolling correlation. Returns 0 if insufficient data."""
    n = min(len(a), len(b), 63)
    if n < 10:
        return 0.0
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da  = math.sqrt(sum((x - ma) ** 2 for x in a))
    db  = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da * db > 0 else 0.0


def _apply_tracking_error_cap(
    blended: Dict[str, float],
    quant: Dict[str, float],
    max_te: float,
) -> Tuple[Dict[str, float], List[str]]:
    """
    Diagonal covariance proxy: TE = sqrt(sum((w_blend - w_quant)^2 * var_i)).
    Uses var_i = 0.0004 (≈2% daily vol) as proxy.
    If TE > cap, scale all AI PM weight changes proportionally.
    """
    diffs = {sym: blended.get(sym, 0.0) - quant.get(sym, 0.0)
             for sym in set(blended) | set(quant)}
    te = math.sqrt(sum(d ** 2 * _VAR_PROXY for d in diffs.values()))

    if te <= max_te:
        return blended, []

    scale  = max_te / te
    result = {}
    for sym in set(blended) | set(quant):
        q_w = quant.get(sym, 0.0)
        b_w = blended.get(sym, 0.0)
        new_w = q_w + (b_w - q_w) * scale
        if new_w < 0 or abs(new_w) >= 0.02:  # keep shorts and meaningful longs
            result[sym] = new_w

    # Renormalize longs only
    long_total = sum(v for v in result.values() if v > 0)
    if long_total > 0:
        result = {s: (v / long_total if v > 0 else v) for s, v in result.items() if abs(v) >= 0.001}

    return result, [
        f"Tracking error {te:.4f} > {max_te:.4f} cap — changes scaled {scale:.2f}x"
    ]


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
