# ascent/integrations/get_mirofish_sentiment.py
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_MIROFISH_BASE_URL = "http://localhost:5001"
_N_ROUNDS = 10
_TIMEOUT_SECS = 1500


def _compute_alignment_score(
    analogue_sentiment: str,
    crowd_sentiment: str,
    crowd_confidence: float,
) -> float:
    """
    Compute how well crowd sentiment aligns with historical analogues expected direction.

    - If analogue and crowd agree: high alignment (confidence-weighted)
    - If they disagree: low alignment
    - Mixed sentiment is a soft signal
    """
    if analogue_sentiment == crowd_sentiment:
        base = 0.60 + crowd_confidence * 0.35
    elif crowd_sentiment == "mixed" or analogue_sentiment == "mixed":
        base = 0.40 + crowd_confidence * 0.20
    else:
        base = max(0.10, 0.50 - crowd_confidence * 0.40)
    return round(min(base, 1.0), 3)


def get_mirofish_sentiment(inputs: dict[str, Any]) -> str:
    """
    Tool executor for get_mirofish_sentiment.

    Runs a MiroFish crowd simulation for the given event and symbols,
    cross-references with historical analogues, and returns a formatted
    string for the AI PM tool loop.

    Returns mirofish_unavailable string if server is unreachable or times out.
    """
    from ascent.integrations.mirofish_client import MiroFishClient
    from ascent.integrations.analogue_matcher import find_analogues
    from ascent.integrations.mirofish_calibration import bootstrap_calibration, get_base_rate

    try:
        bootstrap_calibration()
    except Exception:
        pass

    event_description = str(inputs.get("event_description", "")).strip()
    symbols = [str(s).upper().strip() for s in inputs.get("symbols", []) if s]

    if not event_description:
        return "Error: event_description is required."

    analogues_with_conf = find_analogues(event_description, symbols, top_k=3)
    analogue_ids = [a["event_id"] for a, _ in analogues_with_conf]
    analogue_match_confidence = float(analogues_with_conf[0][1]) if analogues_with_conf else 0.0
    best_analogue_sentiment = analogues_with_conf[0][0].get("sentiment_label", "mixed") if analogues_with_conf else "mixed"
    analogue_sectors = list({
        sector
        for a, _ in analogues_with_conf
        for sector in a.get("affected_sectors", [])
    })[:3]
    primary_sector = analogue_sectors[0] if analogue_sectors else None

    base_rate = get_base_rate(best_analogue_sentiment, primary_sector)

    client = MiroFishClient(base_url=_MIROFISH_BASE_URL)
    raw = client.run_sync(event_description, symbols, n_rounds=_N_ROUNDS, timeout_secs=_TIMEOUT_SECS)

    if raw is None:
        base_rate_str = _format_base_rate(base_rate)
        analogues_str = ", ".join(analogue_ids[:2]) if analogue_ids else "none"
        return (
            f"MIROFISH SENTIMENT: status=timeout\n"
            f"MiroFish did not respond within {_TIMEOUT_SECS}s — proceeding on historical analogues only.\n"
            f"Most similar events: {analogues_str} (match confidence: {analogue_match_confidence:.0%})\n"
            f"Historical base rate: {base_rate_str}\n"
            f"-> Log 'mirofish_unavailable' in thesis. Do not let this block your portfolio submission."
        )

    crowd_sentiment = raw["overall_sentiment"]
    crowd_confidence = raw["confidence"]
    alignment_score = _compute_alignment_score(best_analogue_sentiment, crowd_sentiment, crowd_confidence)
    print(
        f"[MiroFish] alignment_score={alignment_score:.2f} "
        f"sentiment={crowd_sentiment} confidence={crowd_confidence:.2f}"
    )

    return _format_result(
        alignment_score=alignment_score,
        crowd_sentiment=crowd_sentiment,
        crowd_confidence=crowd_confidence,
        base_rate=base_rate,
        top_themes=raw.get("top_themes", []),
        warning_flags=raw.get("warning_flags", []),
        analogue_ids=analogue_ids,
        analogue_match_confidence=analogue_match_confidence,
    )


def _format_base_rate(base_rate: dict) -> str:
    n = base_rate.get("n_events", 0)
    med = base_rate.get("median_21d_return")
    pos = base_rate.get("positive_rate")
    if n == 0 or med is None:
        return "no historical data"
    pos_str = f", positive in {pos:.0%} of cases" if pos is not None else ""
    return f"in {n} similar past events, median 21d return was {med:+.1%}{pos_str}"


def _format_result(
    alignment_score: float,
    crowd_sentiment: str,
    crowd_confidence: float,
    base_rate: dict,
    top_themes: list,
    warning_flags: list,
    analogue_ids: list,
    analogue_match_confidence: float,
) -> str:
    base_rate_str = _format_base_rate(base_rate)
    analogue_str = ", ".join(analogue_ids[:2]) if analogue_ids else "none"
    themes_str = "\n".join(f"  - {t}" for t in top_themes[:5]) if top_themes else "  (none extracted)"
    flags_str = "\n".join(f"  ⚠  {f}" for f in warning_flags[:4]) if warning_flags else "  None"

    if alignment_score > 0.70:
        decision = (
            "CONVICTION AMPLIFIER — crowd confirms thesis. "
            "You may use 10% weight for AMPLIFY picks without needing all 3 standard conditions."
        )
    elif alignment_score < 0.40 and base_rate.get("median_21d_return", 0) is not None and (base_rate.get("median_21d_return") or 0) < 0:
        decision = (
            "SOFT REDUCE SIGNAL — crowd diverges from thesis AND historical base rate is negative. "
            "Apply 25% size reduction to this pick. Log warning_flags in thesis."
        )
    elif alignment_score < 0.40:
        decision = (
            "CAUTION — crowd diverges from thesis. "
            "Consider 25% size reduction if warning_flags are relevant."
        )
    else:
        decision = "NEUTRAL — proceed at standard sizing. No conviction amplifier or reduce signal."

    return (
        f"MIROFISH CROWD SENTIMENT REPORT\n"
        f"{'='*44}\n"
        f"Status: ok\n"
        f"Crowd Sentiment: {crowd_sentiment.upper()} (confidence {crowd_confidence:.0%})\n"
        f"ALIGNMENT Score: {alignment_score:.2f} — {'HIGH' if alignment_score > 0.70 else 'LOW' if alignment_score < 0.40 else 'MODERATE'}\n"
        f"Analogue Match:  {analogue_str} ({analogue_match_confidence:.0%} similarity)\n"
        f"Historical Base Rate: {base_rate_str}\n"
        f"\nTop Crowd Themes:\n{themes_str}\n"
        f"\nWarning Flags:\n{flags_str}\n"
        f"\n-> DECISION RULE: {decision}"
    )
