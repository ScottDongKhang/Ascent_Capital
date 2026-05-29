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
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

MIN_ML_CASES = 30        # train logistic regression once we have this many matured outcomes
BLOCK_WIN_RATE = 0.35    # block if win_rate below this AND n_cases >= MIN_BLOCK_CASES
MIN_BLOCK_CASES = 8      # need at least this many cases to block an override type
REDUCE_WIN_RATE = 0.50   # reduce size if win_rate between BLOCK and REDUCE
STRONG_WIN_RATE = 0.60   # full conviction above this

_MODEL_PATH = Path("data_cache/conviction_model.pkl")
# momentum_exhaustion replaces valuation — requires crowding evidence before reaching the gate
_OVERRIDE_TYPES = ["momentum_exhaustion", "data_quality", "regime_macro", "news_event",
                   "correlation_risk", "valuation"]
_REGIMES       = ["calm_bull", "stressed", "crisis", "neutral", "uncertain"]

_model_cache: dict = {"model": None, "n_trained": 0}


def _build_feature_vector(record) -> list:
    ot_vec  = [1.0 if record.override_type == t else 0.0 for t in _OVERRIDE_TYPES]
    reg_vec = [1.0 if record.regime == r else 0.0 for r in _REGIMES]
    num = [
        record.momentum_252d or 0.0,
        record.sec_tone or 0.0,
        record.transcript_sentiment or 0.0,
        record.reddit_buzz or 0.0,
        record.trends_direction or 0.0,
    ]
    return ot_vec + reg_vec + num


def _get_ml_model(matured_records):
    """Return trained LogisticRegression if n >= MIN_ML_CASES, else None."""
    n = len(matured_records)
    if n < MIN_ML_CASES:
        return None

    if _model_cache["model"] is not None and _model_cache["n_trained"] == n:
        return _model_cache["model"]

    if _MODEL_PATH.exists():
        try:
            with open(_MODEL_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_trained") == n:
                _model_cache["model"] = cached["model"]
                _model_cache["n_trained"] = n
                return _model_cache["model"]
        except Exception:
            pass

    try:
        from sklearn.linear_model import LogisticRegression
        X = [_build_feature_vector(r) for r in matured_records]
        y = [1 if r.wedge_21d > 0 else 0 for r in matured_records]
        model = LogisticRegression(C=1.0, max_iter=500)
        model.fit(X, y)
        _model_cache["model"] = model
        _model_cache["n_trained"] = n
        try:
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_MODEL_PATH, "wb") as f:
                pickle.dump({"model": model, "n_trained": n}, f)
        except Exception:
            pass
        return model
    except Exception as exc:
        log.warning("[ConvictionGate] ML model training failed: %s", exc)
        return None


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
    symbol: Optional[str] = None,
) -> GateResult:
    """
    Evaluate whether the AI PM should proceed with an override of the given type.

    Args:
        override_type:    One of momentum_exhaustion / data_quality / regime_macro /
                          news_event / correlation_risk / valuation
        regime:           Current regime label (e.g. 'calm_bull')
        calibration_ic:   Spearman IC from calibration tracker (optional)
        log_path:         Path to decision_memory.jsonl (uses default if None)
        symbol:           Symbol for ML feature lookup at inference time (optional)

    Returns:
        GateResult with proceed flag, size multiplier, and reasoning.
    """
    from ascent.memory.decision_memory import get_statistics, query, OverrideRecord, _read_altdata_context

    kwargs = {} if log_path is None else {"log_path": log_path}
    stats = get_statistics(override_type=override_type, regime=regime, **kwargs)
    n     = stats["n_cases"]
    wr    = stats["win_rate"]
    aw    = stats["avg_wedge"]

    # ML path — only for non-structural override types
    if override_type not in ("data_quality", "correlation_risk", "news_event"):
        all_matured = [r for r in query(override_type=None, regime=None, n=1000, **kwargs)
                       if r.wedge_21d is not None]
        model = _get_ml_model(all_matured)
        if model is not None:
            ctx = _read_altdata_context(symbol) if symbol else {
                "sec_tone": None, "transcript_sentiment": None,
                "reddit_buzz": None, "trends_direction": None,
            }
            probe = OverrideRecord(
                entry_id="probe", rebalance_date="", symbol=symbol or "",
                override_type=override_type, regime=regime,
                ai_action="", ai_weight=0.0, quant_weight=0.0, weight_delta=0.0,
                momentum_252d=None, wedge_21d=None,
                sec_tone=ctx["sec_tone"],
                transcript_sentiment=ctx["transcript_sentiment"],
                reddit_buzz=ctx["reddit_buzz"],
                trends_direction=ctx["trends_direction"],
            )
            prob = float(model.predict_proba([_build_feature_vector(probe)])[0][1])
            n_total = len(all_matured)
            if abs(prob - 0.5) >= 0.05:
                if prob >= 0.65:
                    return GateResult(
                        proceed=True, size_multiplier=1.0, confidence="strong",
                        reason=f"ML model: {prob:.0%} win probability ({n_total} cases)",
                        n_cases=n, win_rate=wr,
                    )
                elif prob >= 0.50:
                    return GateResult(
                        proceed=True, size_multiplier=0.80, confidence="proceed",
                        reason=f"ML model: {prob:.0%} win probability ({n_total} cases)",
                        n_cases=n, win_rate=wr,
                    )
                else:
                    return GateResult(
                        proceed=False, size_multiplier=0.0, confidence="block",
                        reason=f"ML model: {prob:.0%} win probability ({n_total} cases) — blocked",
                        n_cases=n, win_rate=wr,
                    )
            # model not confident enough — fall through to rules

    # ── Rules-based logic ─────────────────────────────────────────────────────
    # correlation_risk and news_event are structural edges — approved, no historical bar needed.
    # news_event must be within 10 days to be credible (enforced by prompt, not here).
    if override_type in ("correlation_risk", "news_event"):
        return GateResult(
            proceed=True, size_multiplier=1.0, confidence="proceed",
            reason=f"{override_type} overrides approved — structural AI PM edge",
            n_cases=n, win_rate=wr,
        )

    # data_quality: only approved when the AI PM cites a confirmed corporate action.
    # We cannot inspect the text here, so we apply a 0.85 multiplier as a friction cost —
    # the AI PM absorbs a small penalty unless historical track record is strong.
    if override_type == "data_quality":
        if n >= MIN_BLOCK_CASES and wr is not None and wr >= STRONG_WIN_RATE:
            return GateResult(
                proceed=True, size_multiplier=1.0, confidence="strong",
                reason=f"data_quality: strong track record {wr:.0%} over {n} cases",
                n_cases=n, win_rate=wr,
            )
        if n >= MIN_BLOCK_CASES and wr is not None and wr < BLOCK_WIN_RATE:
            return GateResult(
                proceed=False, size_multiplier=0.0, confidence="block",
                reason=f"data_quality overrides blocked: win rate {wr:.0%} — review what counts as a real corporate action",
                n_cases=n, win_rate=wr,
            )
        return GateResult(
            proceed=True, size_multiplier=0.85, confidence="caution",
            reason=f"data_quality: {n} cases, building track record — 15% size friction applied",
            n_cases=n, win_rate=wr,
        )

    # momentum_exhaustion: requires crowding evidence to reach the gate at all (enforced by prompt).
    # Here we apply the same calibration + win-rate rules as other types.
    if override_type == "momentum_exhaustion":
        if calibration_ic is not None and calibration_ic < 0.05:
            return GateResult(
                proceed=False, size_multiplier=0.0, confidence="block",
                reason=f"momentum_exhaustion blocked: calibration IC={calibration_ic:.3f} — your signal is noise",
                n_cases=n, win_rate=wr,
            )
        # If we have data showing this type consistently fails, block it
        if n >= MIN_BLOCK_CASES and wr is not None and wr < BLOCK_WIN_RATE:
            return GateResult(
                proceed=False, size_multiplier=0.0, confidence="block",
                reason=f"momentum_exhaustion blocked: {wr:.0%} win rate over {n} cases",
                n_cases=n, win_rate=wr,
            )

    # valuation: blocked in all regimes unless both calibration IC and track record are strong.
    # This type should rarely be submitted — the prompt now requires crowding evidence instead.
    if override_type == "valuation":
        if calibration_ic is not None and calibration_ic < 0.10:
            return GateResult(
                proceed=False, size_multiplier=0.0, confidence="block",
                reason=f"Valuation overrides blocked: calibration IC={calibration_ic:.3f} < 0.10. Use momentum_exhaustion type with crowding evidence instead.",
                n_cases=n, win_rate=wr,
            )
        if n >= MIN_BLOCK_CASES and wr is not None and wr < BLOCK_WIN_RATE:
            return GateResult(
                proceed=False, size_multiplier=0.0, confidence="block",
                reason=f"Valuation overrides blocked: {wr:.0%} win rate across {n} cases. Use momentum_exhaustion type with crowding evidence instead.",
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
    if wr is not None and wr >= STRONG_WIN_RATE and aw is not None and aw > 0:
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
    if n >= MIN_BLOCK_CASES and wr is not None and wr < BLOCK_WIN_RATE:
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
