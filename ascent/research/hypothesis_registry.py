"""Append-only registry of tested signal/variant hypotheses and their
verdicts -- prevents re-testing an already-falsified idea. Reads the
existing self_improve_log.jsonl as its data source rather than duplicating
what evaluate_variant() already logs; adds a lookup index on top."""

import json
from datetime import date
from pathlib import Path
from hashlib import sha256

SELF_IMPROVE_LOG = Path("logs/self_improve_log.jsonl")   # existing, read-only here
REGISTRY_PATH     = Path("logs/hypothesis_registry.jsonl")  # new


def _config_hash(variant_config: dict) -> str:
    """Stable hash of a variant config so re-tests of the identical
    configuration are detectable regardless of key ordering."""
    return sha256(json.dumps(variant_config, sort_keys=True).encode()).hexdigest()[:12]


def was_previously_rejected(variant_config: dict) -> dict | None:
    """Returns the prior rejection record if this exact config was already
    tested and NOT promoted, else None. Call this BEFORE evaluate_variant()
    spends compute re-testing something already known to fail."""
    h = _config_hash(variant_config)
    if not REGISTRY_PATH.exists():
        return None
    for line in reversed(REGISTRY_PATH.read_text().splitlines()):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if entry.get("config_hash") == h and not entry.get("promoted"):
            return entry
    return None


def record_verdict(variant_config: dict, variant_id: str, oos_sharpe: float,
                    edge: float, promoted: bool, reason: str = "",
                    oos_calmar: float = None) -> None:
    """Append one hypothesis verdict. Called once per variant from inside
    evaluate_variant()'s existing logging block (self_improve.py:296-309),
    not as a separate pass -- keep the two logs in sync by construction.

    `oos_calmar` (added 2026-08-23, alongside the Calmar-based rework of
    self_improve.py's ranking objective) is optional and defaults to None so
    older callers/tests that only pass oos_sharpe keep working. Entries
    written before this change have no `oos_calmar` key at all; callers
    reading this field back must treat a missing key as "no real Calmar
    on record", never substitute oos_sharpe for it (different scale)."""
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    entry = {
        "config_hash": _config_hash(variant_config),
        "variant_id":  variant_id,
        "config":      variant_config,
        "oos_sharpe":  oos_sharpe,
        "oos_calmar":  oos_calmar,
        "edge":        edge,
        "promoted":    promoted,
        "reason":      reason,
        "date":        date.today().isoformat(),
    }
    with open(REGISTRY_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
