# ascent/monitoring/alpha_wedge_tracker.py
"""
Tracks the performance wedge between AI PM proposals and the pure quant baseline.

After each rebalance, records both portfolios. After 21 trading days, fills in
the realized 21-day return for each and computes the wedge (AI PM - quant).
This is the ground truth for whether the AI PM is adding or destroying value.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOG = _REPO_ROOT / "logs" / "alpha_wedge.jsonl"


def record_rebalance(
    rebalance_date: str,
    ai_pm_weights: Dict[str, float],
    quant_weights: Dict[str, float],
    override_types: Optional[Dict[str, str]] = None,
    log_path: Path = _DEFAULT_LOG,
) -> None:
    """Record AI PM and quant portfolios at a rebalance for future wedge computation.

    override_types: {symbol: override_type} for each AI PM override — used later to
    attribute the wedge by override type (data_quality, regime_macro, etc.).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _norm(w: Dict[str, float]) -> Dict[str, float]:
        total = sum(abs(v) for v in w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else {}

    entry = {
        "rebalance_date": rebalance_date,
        "ai_pm_weights": _norm(ai_pm_weights),
        "quant_weights": _norm(quant_weights),
        "override_types": override_types or {},
        "ai_pm_return_21d": None,
        "quant_return_21d": None,
        "wedge_21d": None,
    }

    # Dedup by date
    try:
        if log_path.exists():
            existing = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
            if any(e.get("rebalance_date") == rebalance_date for e in existing):
                log.debug("[AlphaWedge] Already recorded %s — skipping", rebalance_date)
                return
    except Exception:
        pass

    with log_path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(
        "[AlphaWedge] Recorded rebalance %s — %d AI PM / %d quant positions",
        rebalance_date, len(ai_pm_weights), len(quant_weights),
    )


def update_outcomes(
    price_returns: Dict[str, float],
    as_of_date: str,
    lookback_days: int = 30,
    log_path: Path = _DEFAULT_LOG,
) -> None:
    """Fill 21d realized returns for rebalances that have matured.

    price_returns: {symbol: cumulative_return_since_rebalance} for the period.
    as_of_date: today's date string — rebalances ≥21 calendar days ago are eligible.
    """
    if not log_path.exists():
        return
    try:
        cutoff = date.fromisoformat(as_of_date) - timedelta(days=lookback_days)
        rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        changed = False

        for row in rows:
            if row.get("wedge_21d") is not None:
                continue
            rb_date = date.fromisoformat(row["rebalance_date"])
            if rb_date > cutoff:
                continue

            def _port_return(weights: Dict[str, float]) -> float:
                return sum(w * price_returns.get(sym, 0.0) for sym, w in weights.items())

            ai_ret  = _port_return(row.get("ai_pm_weights", {}))
            qnt_ret = _port_return(row.get("quant_weights", {}))
            row["ai_pm_return_21d"] = round(ai_ret, 6)
            row["quant_return_21d"] = round(qnt_ret, 6)
            row["wedge_21d"] = round(ai_ret - qnt_ret, 6)
            changed = True

        if changed:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", dir=log_path.parent, delete=False, suffix=".tmp"
            )
            for row in rows:
                tmp.write(json.dumps(row) + "\n")
            tmp.flush()
            os.replace(tmp.name, log_path)
            log.info("[AlphaWedge] Updated outcomes as of %s", as_of_date)
    except Exception as exc:
        log.warning("[AlphaWedge] update_outcomes failed: %s", exc)


def get_wedge_summary(n: int = 10, log_path: Path = _DEFAULT_LOG) -> str:
    """Return a text summary of the AI PM vs quant wedge for the last n matured rebalances."""
    if not log_path.exists():
        return "No alpha wedge data yet — tracking begins at first rebalance."

    try:
        rows = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
        total = len(rows)
        matured = [r for r in rows if r.get("wedge_21d") is not None]

        if not matured:
            return (
                f"Alpha wedge tracking active ({total} rebalance(s) recorded, "
                "awaiting 21-day maturation window)."
            )

        recent = matured[-n:]
        wedges = [r["wedge_21d"] for r in recent]
        mean_wedge = sum(wedges) / len(wedges)
        wins = sum(1 for w in wedges if w > 0)

        lines = [f"AI PM vs Quant Wedge — last {len(recent)} matured rebalance(s):"]
        for r in recent:
            w = r["wedge_21d"]
            sign = "✓" if w > 0 else "✗"
            lines.append(
                f"  {sign} {r['rebalance_date']}  "
                f"AI PM={r['ai_pm_return_21d']:+.2%}  "
                f"Quant={r['quant_return_21d']:+.2%}  "
                f"Wedge={w:+.2%}"
            )
        verdict = "AI PM adding value" if mean_wedge > 0 else "AI PM subtracting value"
        lines.append(
            f"\n  Mean wedge: {mean_wedge:+.2%}  "
            f"({wins}/{len(recent)} rebalances positive)  — {verdict}"
        )

        # Attribution by override type
        type_wedges: Dict[str, list] = {}
        for r in recent:
            for sym, otype in r.get("override_types", {}).items():
                ai_w  = r["ai_pm_weights"].get(sym, 0.0)
                qnt_w = r["quant_weights"].get(sym, 0.0)
                if abs(ai_w - qnt_w) < 0.005:
                    continue
                sym_ret = 0.0  # approximate: use portfolio-level wedge as proxy
                type_wedges.setdefault(otype, []).append(r["wedge_21d"])

        if len(type_wedges) > 1:
            lines.append("\n  Wedge by override type (approx):")
            for otype, ws in sorted(type_wedges.items()):
                avg = sum(ws) / len(ws)
                lines.append(f"    {otype}: {avg:+.2%} avg ({len(ws)} obs)")

        return "\n".join(lines)
    except Exception as exc:
        return f"Could not compute wedge summary: {exc}"
