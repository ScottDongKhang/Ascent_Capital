# ascent/integrations/mirofish_client.py
from __future__ import annotations

import logging
import re
import time
from typing import Any

log = logging.getLogger(__name__)

_BULLISH_WORDS = {
    "bullish", "optimistic", "positive", "upside", "rally", "buy", "strong",
    "outperform", "upgrade", "confidence", "growth", "expansion", "momentum",
}
_BEARISH_WORDS = {
    "bearish", "pessimistic", "negative", "downside", "sell", "weak", "crash",
    "underperform", "downgrade", "risk", "recession", "decline", "concern",
}


def _parse_sentiment_from_markdown(md: str) -> dict[str, Any]:
    """Extract structured sentiment from MiroFish report markdown."""
    text = md.lower()
    words = re.findall(r"\b\w+\b", text)
    bull_count = sum(1 for w in words if w in _BULLISH_WORDS)
    bear_count = sum(1 for w in words if w in _BEARISH_WORDS)
    total = bull_count + bear_count + 1e-9

    if bull_count > bear_count * 1.3:
        sentiment = "bullish"
        confidence = min(bull_count / total, 0.95)
    elif bear_count > bull_count * 1.3:
        sentiment = "bearish"
        confidence = min(bear_count / total, 0.95)
    else:
        sentiment = "mixed"
        confidence = 0.50

    themes: list[str] = []
    for m in re.finditer(r"^(?:#{1,3}\s+|[-*]\s+)(.+)$", md, re.MULTILINE):
        line = m.group(1).strip().lower()
        if len(line) > 5 and len(line) < 80:
            themes.append(line)
    themes = themes[:5]

    flags: list[str] = []
    for line in md.split("\n"):
        lower = line.lower()
        if any(w in lower for w in ("risk", "concern", "warning", "caveat", "tariff")):
            clean = line.strip().lstrip("- *").strip()
            if len(clean) > 10:
                flags.append(clean[:100])
    flags = flags[:4]

    return {
        "overall_sentiment": sentiment,
        "confidence": round(confidence, 3),
        "top_themes": themes,
        "warning_flags": flags,
    }


class MiroFishClient:
    """
    Synchronous HTTP client for the MiroFish social simulation API.
    Wraps the multi-step flow: project -> graph -> simulate -> report.
    All calls are via HTTP to a local MiroFish server process.
    """

    def __init__(self, base_url: str = "http://localhost:5001") -> None:
        self._base = base_url.rstrip("/")

    def _post(self, path: str, **kwargs) -> dict:
        import requests
        url = f"{self._base}{path}"
        r = requests.post(url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, **kwargs) -> dict:
        import requests
        url = f"{self._base}{path}"
        r = requests.get(url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()

    def _poll(self, fn, interval: float, deadline: float, success_key: str, success_vals: set) -> dict | None:
        """Generic poller. Returns the data dict when success_key matches success_vals, or None on timeout."""
        while time.monotonic() < deadline:
            try:
                result = fn()
                data = result.get("data", {})
                val = data.get(success_key, "")
                if val in success_vals:
                    return data
                log.debug("[MiroFish] polling %s=%s (want %s)", success_key, val, success_vals)
            except Exception as exc:
                log.debug("[MiroFish] poll error: %s", exc)
            time.sleep(interval)
        return None

    def _create_project(self, event_description: str, symbols: list[str]) -> str:
        """Upload event as text file -> get project_id."""
        sim_requirement = (
            f"Financial market social simulation: Simulate the reaction of diverse market participants "
            f"(retail investors, professional traders, financial analysts, hedge fund managers, media commentators) "
            f"to the following market event: {event_description}. "
            f"Key equities involved: {', '.join(symbols)}. "
            f"Focus on: (1) bullish or bearish sentiment toward these stocks, "
            f"(2) crowd conviction about timing, (3) risk factors the crowd discusses."
        )
        event_text = f"Market Event Analysis\n\n{event_description}\n\nSymbols: {', '.join(symbols)}"
        result = self._post(
            "/api/graph/ontology/generate",
            data={
                "simulation_requirement": sim_requirement,
                "project_name": f"Ascent_{symbols[0] if symbols else 'mkt'}",
            },
            files=[("files", ("event.txt", event_text.encode(), "text/plain"))],
        )
        return result["data"]["project_id"]

    def _build_graph(self, project_id: str, deadline: float) -> str | None:
        """Trigger async graph build, poll until complete. Returns graph_id or None."""
        result = self._post("/api/graph/build", json={"project_id": project_id})
        task_id = result["data"]["task_id"]

        def check():
            return self._get(f"/api/graph/task/{task_id}")

        data = self._poll(check, interval=5.0, deadline=deadline, success_key="status", success_vals={"completed"})
        if data is None:
            return None
        return data.get("result", {}).get("graph_id")

    def _create_simulation(self, project_id: str, graph_id: str) -> str:
        """Create simulation record. Returns simulation_id."""
        result = self._post("/api/simulation/create", json={"project_id": project_id, "graph_id": graph_id})
        return result["data"]["simulation_id"]

    def _prepare_simulation(self, sim_id: str, deadline: float) -> bool:
        """Trigger and wait for simulation preparation. Returns True when ready."""
        result = self._post("/api/simulation/prepare", json={"simulation_id": sim_id})
        data = result.get("data", {})
        if data.get("already_prepared") or data.get("status") == "ready":
            return True

        def check():
            return self._post("/api/simulation/prepare/status", json={"simulation_id": sim_id})

        polled = self._poll(check, interval=4.0, deadline=deadline, success_key="status", success_vals={"ready"})
        return polled is not None

    def _start_simulation(self, sim_id: str, max_rounds: int = 10) -> bool:
        """Start the simulation. Returns True on success."""
        try:
            self._post("/api/simulation/start", json={
                "simulation_id": sim_id,
                "platform": "reddit",
                "max_rounds": max_rounds,
            })
            return True
        except Exception as exc:
            log.debug("[MiroFish] start failed: %s", exc)
            return False

    def _wait_for_simulation(self, sim_id: str, deadline: float) -> bool:
        """Poll run-status until completed or stopped."""
        def check():
            return self._get(f"/api/simulation/{sim_id}/run-status")

        data = self._poll(
            check, interval=6.0, deadline=deadline,
            success_key="runner_status", success_vals={"completed", "stopped"},
        )
        return data is not None

    def _generate_report(self, sim_id: str, deadline: float) -> str | None:
        """Trigger async report generation. Returns report_id or None."""
        result = self._post("/api/report/generate", json={"simulation_id": sim_id})
        data = result.get("data", {})
        if data.get("already_generated"):
            return data.get("report_id")
        report_id = data.get("report_id")
        task_id = data.get("task_id")

        def check():
            return self._post("/api/report/generate/status", json={"task_id": task_id})

        polled = self._poll(check, interval=5.0, deadline=deadline, success_key="status", success_vals={"completed"})
        if polled is None:
            return report_id
        return report_id

    def _get_report(self, report_id: str) -> dict | None:
        """Fetch completed report by ID."""
        try:
            result = self._get(f"/api/report/{report_id}")
            return result.get("data")
        except Exception as exc:
            log.debug("[MiroFish] get_report failed: %s", exc)
            return None

    def run_sync(
        self,
        event_description: str,
        symbols: list[str],
        n_rounds: int = 10,
        timeout_secs: int = 480,
    ) -> dict[str, Any] | None:
        """
        Run the full MiroFish pipeline synchronously.

        Returns parsed sentiment dict on success, None on timeout or any failure.
        The caller is responsible for treating None as 'mirofish_unavailable'.
        """
        import requests as _req
        deadline = time.monotonic() + timeout_secs

        try:
            log.info("[MiroFish] Starting pipeline for: %s (symbols: %s)", event_description[:60], symbols)

            project_id = self._create_project(event_description, symbols)
            log.info("[MiroFish] project_id=%s", project_id)
            if time.monotonic() > deadline:
                return None

            graph_id = self._build_graph(project_id, deadline)
            if not graph_id:
                log.warning("[MiroFish] Graph build timed out or failed")
                return None
            log.info("[MiroFish] graph_id=%s", graph_id)

            sim_id = self._create_simulation(project_id, graph_id)
            log.info("[MiroFish] sim_id=%s", sim_id)
            if time.monotonic() > deadline:
                return None

            if not self._prepare_simulation(sim_id, deadline):
                log.warning("[MiroFish] Simulation preparation timed out")
                return None

            if not self._start_simulation(sim_id, max_rounds=n_rounds):
                log.warning("[MiroFish] Simulation start failed")
                return None

            if not self._wait_for_simulation(sim_id, deadline):
                log.warning("[MiroFish] Simulation run timed out")
                return None

            report_id = self._generate_report(sim_id, deadline)
            if not report_id:
                return None

            report_data = self._get_report(report_id)
            if not report_data:
                return None

            md = report_data.get("markdown_content", "")
            parsed = _parse_sentiment_from_markdown(md)
            parsed["report_id"] = report_id
            parsed["simulation_id"] = sim_id
            log.info("[MiroFish] Done. sentiment=%s confidence=%.2f", parsed["overall_sentiment"], parsed["confidence"])
            return parsed

        except _req.exceptions.ConnectionError:
            log.warning("[MiroFish] Server not reachable at %s", self._base)
            return None
        except _req.exceptions.Timeout:
            log.warning("[MiroFish] HTTP timeout")
            return None
        except Exception as exc:
            log.warning("[MiroFish] Pipeline failed: %s", exc)
            return None
