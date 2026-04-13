"""
ascent/execution/approval_server.py
Lightweight approval server for large trades.

Serves approval.html and handles approve/reject POST requests.
Reads/writes execution/pending_approvals.json.

Usage:
    python3 -m ascent.execution.approval_server          # default port 8099
    python3 -m ascent.execution.approval_server --port 9000
"""

import json
import os
import sys
from datetime import datetime, timedelta
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


PENDING_PATH      = Path("execution/pending_approvals.json")
APPROVAL_HTML_PATH = Path("ascent/dashboard/approval.html")
DEFAULT_PORT      = 8099
TIMEOUT_MINUTES   = 30


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _read_pending() -> dict:
    if not PENDING_PATH.exists():
        return {"trades": [], "created_at": None, "status": "empty"}
    try:
        with open(PENDING_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"trades": [], "created_at": None, "status": "empty"}


def _write_pending(data: dict):
    os.makedirs(PENDING_PATH.parent, exist_ok=True)
    with open(PENDING_PATH, "w") as f:
        json.dump(data, f, indent=2)


# ── Public API (called by eod_runner.py) ─────────────────────────────────────

def write_pending_trades(trades: list, run_date: str):
    """
    Called by eod_runner.py when trades exceed the large-trade threshold.

    Args:
        trades:   List of dicts with {symbol, side, qty, notional, weight_pct}
        run_date: ISO date string (YYYY-MM-DD)
    """
    data = {
        "trades":     trades,
        "created_at": datetime.now().isoformat(),
        "run_date":   run_date,
        "status":     "pending",
        "expires_at": (datetime.now() + timedelta(minutes=TIMEOUT_MINUTES)).isoformat(),
    }
    _write_pending(data)
    print(f"[Approval] {len(trades)} trades pending approval (expires in {TIMEOUT_MINUTES}min)")
    print(f"[Approval] Open http://localhost:{DEFAULT_PORT} to review")


def check_approval_status() -> str:
    """
    Check current approval status.
    Returns: "approved", "rejected", "pending", "expired", or "empty"

    Auto-transitions "pending" → "expired" when the expiry time is past.
    """
    data   = _read_pending()
    status = data.get("status", "empty")

    if status == "pending" and data.get("expires_at"):
        try:
            expires = datetime.fromisoformat(data["expires_at"])
            if datetime.now() > expires:
                data["status"] = "expired"
                _write_pending(data)
                return "expired"
        except (ValueError, TypeError):
            pass

    return status


# ── HTTP handler ──────────────────────────────────────────────────────────────

class ApprovalHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/approval"):
            self._serve_html()
        elif self.path == "/api/pending":
            self._serve_pending()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/approve":
            self._handle_decision("approved")
        elif self.path == "/api/reject":
            self._handle_decision("rejected")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        # Allow CORS pre-flight (useful when testing from browser console)
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _serve_html(self):
        if not APPROVAL_HTML_PATH.exists():
            self.send_error(500, "approval.html not found")
            return
        content = APPROVAL_HTML_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_pending(self):
        data = _read_pending()
        # Auto-expire
        if data.get("status") == "pending" and data.get("expires_at"):
            try:
                if datetime.now() > datetime.fromisoformat(data["expires_at"]):
                    data["status"] = "expired"
                    _write_pending(data)
            except (ValueError, TypeError):
                pass
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_decision(self, decision: str):
        data = _read_pending()
        data["status"]     = decision
        data["decided_at"] = datetime.now().isoformat()
        _write_pending(data)
        body = json.dumps({"status": decision}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"[Approval] Decision: {decision.upper()}")

    def log_message(self, format, *args):
        # Suppress noisy default request logging
        pass


# ── Server runner ─────────────────────────────────────────────────────────────

def run_server(port: int = DEFAULT_PORT):
    server = HTTPServer(("0.0.0.0", port), ApprovalHandler)
    print(f"[Approval] Server running on http://0.0.0.0:{port}")
    print(f"[Approval] Open http://localhost:{port} to view pending trades")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Approval] Server stopped")
        server.server_close()


if __name__ == "__main__":
    port = DEFAULT_PORT
    if "--port" in sys.argv:
        idx  = sys.argv.index("--port")
        port = int(sys.argv[idx + 1])
    run_server(port)


# ── Async approval gate ───────────────────────────────────────────────────────

import threading
from dataclasses import dataclass

APPROVAL_PENDING_PATH = Path("execution/approval_pending.json")
HEARTBEAT_LOG_PATH    = Path("logs/approval_heartbeat.jsonl")


@dataclass
class ApprovalResult:
    status: str   # "approved" | "rejected" | "timeout" | "expired"
    trades: list


def wait_for_approval_async(
    pending_trades: list,
    timeout_seconds: int = 1800,
    poll_interval: int = 30,
    resume: bool = False,
    pending: dict = None,
) -> ApprovalResult:
    """
    Non-blocking approval gate using threading.Event.

    Normal flow (resume=False):
    1. Write persistent state to execution/approval_pending.json
    2. Spawn watcher thread that polls check_approval_status() every poll_interval seconds
    3. Log heartbeat every 5 minutes
    4. When approved/rejected/timeout: set event, join thread, clear state file, return

    Resume flow (resume=True, pending=dict):
    - Recalculate remaining timeout from pending["expires_at"]
    - If already expired: return ApprovalResult(status="expired", ...)
    - Otherwise: poll for remaining time

    Args:
        pending_trades:   list of trade dicts
        timeout_seconds:  max wait (default 30 min)
        poll_interval:    seconds between checks (default 30)
        resume:           if True, resuming after restart
        pending:          previously-persisted pending dict (required when resume=True)
    """
    import time as _time
    import logging as _logging
    log = _logging.getLogger(__name__)

    if resume and pending is not None:
        # Recalculate remaining timeout from persisted expires_at
        expires_at = datetime.fromisoformat(pending["expires_at"])
        remaining = int((expires_at - datetime.now()).total_seconds())
        if remaining <= 0:
            log.warning("[Approval] Resumed state already expired — clearing")
            APPROVAL_PENDING_PATH.unlink(missing_ok=True)
            return ApprovalResult(status="expired", trades=pending.get("trades", []))
        timeout_seconds = remaining
        log.info(f"[Approval] Resuming with {remaining}s remaining")
    else:
        # Write persistent state before starting the wait
        import os as _os
        _os.makedirs(APPROVAL_PENDING_PATH.parent, exist_ok=True)
        state = {
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(seconds=timeout_seconds)).isoformat(),
            "trades":     pending_trades,
            "status":     "pending",
        }
        APPROVAL_PENDING_PATH.write_text(json.dumps(state, indent=2))
        log.info(f"[Approval] Persistent state written — {len(pending_trades)} trades pending")

    approval_event = threading.Event()
    result: dict = {"status": "pending"}

    def _watcher():
        import time as _t
        elapsed = 0
        while elapsed < timeout_seconds:
            try:
                status = check_approval_status()
                if status in ("approved", "rejected", "expired"):
                    result["status"] = status
                    approval_event.set()
                    return
            except Exception as exc:
                log.warning(f"[Approval] check_approval_status raised: {exc} — retrying")
            _t.sleep(poll_interval)
            elapsed += poll_interval
            # Heartbeat every 5 minutes
            if elapsed % 300 == 0:
                try:
                    import os as _o
                    _o.makedirs(HEARTBEAT_LOG_PATH.parent, exist_ok=True)
                    heartbeat = {
                        "timestamp":         datetime.now().isoformat(),
                        "elapsed_seconds":   elapsed,
                        "remaining_seconds": timeout_seconds - elapsed,
                        "status":            "waiting",
                    }
                    with HEARTBEAT_LOG_PATH.open("a") as _hf:
                        _hf.write(json.dumps(heartbeat) + "\n")
                    log.info(
                        f"[Approval] Waiting... {elapsed // 60}m elapsed, "
                        f"{(timeout_seconds - elapsed) // 60}m remaining"
                    )
                except Exception:
                    pass
        result["status"] = "timeout"
        approval_event.set()

    watcher = threading.Thread(target=_watcher, daemon=True)
    watcher.start()
    approval_event.wait(timeout=timeout_seconds + 10)
    watcher.join(timeout=5)

    final_status = result["status"]

    if final_status in ("approved", "rejected", "timeout", "expired"):
        # Clear persistent state only on confirmed terminal status
        APPROVAL_PENDING_PATH.unlink(missing_ok=True)
    else:
        # Watcher died without setting event — preserve state file for resume/inspection
        log.error(
            f"[Approval] wait_for_approval_async exiting with unexpected status '{final_status}' — "
            f"preserving {APPROVAL_PENDING_PATH} for restart resume"
        )
        # Still treat as timeout from caller's perspective
        final_status = "timeout"

    return ApprovalResult(status=final_status, trades=pending_trades)
