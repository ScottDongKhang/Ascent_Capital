"""
compliance/audit_trail.py
Immutable append-only decision log with SHA-256 hash chain.

Every entry links to the previous via prev_hash, forming a tamper-evident chain.
Any modification to a historical entry invalidates all subsequent hashes.

Usage:
    from compliance.audit_trail import AuditTrail
    trail = AuditTrail()
    trail.record("order_submitted", {"symbol": "AAPL", "side": "buy", "shares": 10})
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict

log = logging.getLogger(__name__)

AUDIT_LOG_PATH = Path("logs/audit_trail.jsonl")

VALID_EVENT_TYPES = {
    "signal_generated",
    "portfolio_constructed",
    "debate_verdict",
    "order_submitted",
    "order_filled",
    "order_cancelled",
    "approval_granted",
    "approval_denied",
    "kill_switch_triggered",
    "regime_changed",
    "config_modified",
    "audit_integrity_check",
    "ai_pm_proposal",
}

_GENESIS_HASH = "0" * 64  # sentinel for first entry


class AuditTrail:
    """
    Append-only decision log with hash chain integrity.

    Thread-safe via a per-instance lock.
    """

    def __init__(self, log_path: Path = AUDIT_LOG_PATH):
        self._path = log_path
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        """Read the entry_hash of the last record, or genesis hash if empty."""
        if not self._path.exists():
            return _GENESIS_HASH
        last_line = ""
        with open(self._path, "rb") as f:
            # Scan to last non-empty line efficiently
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                f.seek(0)
            last_line = f.readline().decode("utf-8", errors="replace").strip()
        if not last_line:
            return _GENESIS_HASH
        try:
            return json.loads(last_line).get("entry_hash", _GENESIS_HASH)
        except Exception:
            return _GENESIS_HASH

    def _sequence_number(self) -> int:
        if not self._path.exists():
            return 1
        count = 0
        with open(self._path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count + 1

    @staticmethod
    def _hash_entry(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def record(self, event_type: str, payload: Dict[str, Any]) -> str:
        """
        Append an event to the audit trail.

        Args:
            event_type: One of VALID_EVENT_TYPES.
            payload:    Serializable dict with event details.

        Returns:
            The entry_hash of the new entry.
        """
        if event_type not in VALID_EVENT_TYPES:
            log.warning("[AuditTrail] Unknown event_type '%s' — recording anyway", event_type)

        with self._lock:
            prev_hash = self._last_hash()
            seq       = self._sequence_number()
            ts        = datetime.now(tz=timezone.utc).isoformat()

            entry_body = {
                "sequence_number": seq,
                "timestamp":       ts,
                "event_type":      event_type,
                "payload":         payload,
                "prev_hash":       prev_hash,
            }
            body_str    = json.dumps(entry_body, sort_keys=True)
            entry_hash  = self._hash_entry(prev_hash + body_str)
            entry_body["entry_hash"] = entry_hash

            with open(self._path, "a") as f:
                f.write(json.dumps(entry_body) + "\n")

        return entry_hash

    def verify_integrity(self) -> dict:
        """
        Re-compute hash chain and verify integrity.

        Returns:
            {
              "valid":          bool,
              "total_entries":  int,
              "date_range":     [first_ts, last_ts],
              "broken_at":      int | None,   # sequence number where chain breaks
              "message":        str,
            }
        """
        if not self._path.exists():
            return {
                "valid": True, "total_entries": 0,
                "date_range": [], "broken_at": None,
                "message": "No audit trail found.",
            }

        prev_hash = _GENESIS_HASH
        count     = 0
        first_ts  = None
        last_ts   = None

        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue

                count += 1
                ts = entry.get("timestamp", "")
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

                stored_hash = entry.get("entry_hash", "")
                stored_prev = entry.get("prev_hash", "")

                if stored_prev != prev_hash:
                    return {
                        "valid":         False,
                        "total_entries": count,
                        "date_range":    [first_ts, last_ts],
                        "broken_at":     entry.get("sequence_number", count),
                        "message":       f"Chain broken at entry {count}: prev_hash mismatch.",
                    }

                # Recompute entry_hash
                check_body = {k: v for k, v in entry.items() if k != "entry_hash"}
                body_str   = json.dumps(check_body, sort_keys=True)
                expected   = self._hash_entry(prev_hash + body_str)

                if expected != stored_hash:
                    return {
                        "valid":         False,
                        "total_entries": count,
                        "date_range":    [first_ts, last_ts],
                        "broken_at":     entry.get("sequence_number", count),
                        "message":       f"Entry {count} hash mismatch — possible tampering.",
                    }

                prev_hash = stored_hash

        return {
            "valid":         True,
            "total_entries": count,
            "date_range":    [first_ts, last_ts],
            "broken_at":     None,
            "message":       f"Chain intact — {count} entries verified.",
        }


# Module-level singleton
_trail: AuditTrail | None = None


def get_trail() -> AuditTrail:
    global _trail
    if _trail is None:
        _trail = AuditTrail()
    return _trail


def record_event(event_type: str, payload: Dict[str, Any]) -> str:
    """Convenience wrapper around the singleton trail."""
    return get_trail().record(event_type, payload)
