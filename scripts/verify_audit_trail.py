#!/usr/bin/env python3
"""
scripts/verify_audit_trail.py
Verify audit trail hash chain integrity.

Exits 0 on success, 1 on failure.
Run monthly or call via run_all_agents.py monthly block.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

INTEGRITY_LOG = Path("logs/audit_integrity.jsonl")


def main() -> int:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    try:
        from compliance.audit_trail import AuditTrail
    except ImportError as e:
        log.error("Cannot import AuditTrail: %s", e)
        return 1

    trail  = AuditTrail()
    result = trail.verify_integrity()

    ts = datetime.utcnow().isoformat()
    result["checked_at"] = ts

    INTEGRITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(INTEGRITY_LOG, "a") as f:
        f.write(json.dumps(result) + "\n")

    if result["valid"]:
        log.info("[AuditIntegrity] PASS — %s", result["message"])
        log.info("[AuditIntegrity] Entries: %d | Range: %s",
                 result["total_entries"],
                 " → ".join(result["date_range"]) if result["date_range"] else "none")
        return 0
    else:
        log.critical("[AuditIntegrity] FAIL — %s", result["message"])
        log.critical("[AuditIntegrity] Broken at sequence: %s", result.get("broken_at"))

        # Alert via alert system (non-fatal — do not halt trading)
        try:
            from ascent.monitoring.alert_system import send_alert
            send_alert({
                "type":      "audit_integrity_failure",
                "severity":  "CRITICAL",
                "message":   f"Audit trail integrity check failed: {result['message']}",
                "timestamp": ts,
            })
        except Exception:
            pass

        return 1


if __name__ == "__main__":
    sys.exit(main())
