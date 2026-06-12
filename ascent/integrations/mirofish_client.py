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

    # MiroFish localises server output (incl. report markdown) from the
    # Accept-Language header, defaulting to Chinese. Our sentiment parser uses
    # English keywords, so always request English.
    _HEADERS = {"Accept-Language": "en"}

    def _post(self, path: str, **kwargs) -> dict:
        import requests
        url = f"{self._base}{path}"
        headers = {**self._HEADERS, **kwargs.pop("headers", {})}
        r = requests.post(url, timeout=30, headers=headers, **kwargs)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, **kwargs) -> dict:
        import requests
        url = f"{self._base}{path}"
        headers = {**self._HEADERS, **kwargs.pop("headers", {})}
        r = requests.get(url, timeout=30, headers=headers, **kwargs)
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
        # Zep's entity extraction/classification only labels entities it can
        # ground in the document text. A bare event description often yields
        # only abstract concept nodes (0 classified entities -> prepare fails),
        # so name the participant archetypes and companies explicitly.
        company_lines = "\n".join(
            f"- {s} is a publicly traded company whose stock is directly affected by this event."
            for s in symbols
        )
        event_text = (
            f"Market Event Analysis\n\n{event_description}\n\n"
            f"Symbols: {', '.join(symbols)}\n\n"
            f"Companies involved:\n{company_lines}\n\n"
            "Market participants reacting to this event:\n"
            "- Retail investors discuss the event on social media and decide whether to buy or sell.\n"
            "- Professional traders adjust their positions in the affected stocks.\n"
            "- Financial analysts at investment firms publish research notes on the affected companies.\n"
            "- Hedge fund managers evaluate the event's impact on their portfolios.\n"
            "- Media commentators cover the story for financial news outlets.\n"
        )
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

    def _get_sim_status(self, sim_id: str) -> str:
        """Fetch the simulation state status string ('preparing', 'ready', 'failed', ...)."""
        try:
            result = self._get(f"/api/simulation/{sim_id}")
            return str(result.get("data", {}).get("status", ""))
        except Exception as exc:
            log.debug("[MiroFish] sim status fetch failed: %s", exc)
            return ""

    def _prepare_simulation(self, sim_id: str, deadline: float) -> str:
        """
        Trigger and wait for simulation preparation.

        Returns one of:
          'ready'       — preparation complete
          'no_entities' — graph has 0 classified entities; prepare cannot succeed
          'failed'      — server-side preparation failure
          'timeout'     — deadline expired while still preparing
        """
        result = self._post("/api/simulation/prepare", json={"simulation_id": sim_id})
        data = result.get("data", {})
        if data.get("already_prepared") or data.get("status") == "ready":
            return "ready"

        # The /prepare response carries the synchronously computed entity count.
        # 0 entities means the server-side prepare task fails immediately, but
        # /prepare/status (simulation_id-only) never surfaces that failure —
        # without this check we would poll until the deadline.
        if data.get("expected_entities_count") == 0:
            log.warning("[MiroFish] Graph yielded 0 classified entities — preparation cannot proceed")
            return "no_entities"

        while time.monotonic() < deadline:
            try:
                polled = self._post("/api/simulation/prepare/status", json={"simulation_id": sim_id})
                status = polled.get("data", {}).get("status", "")
                if status == "ready":
                    return "ready"
                if self._get_sim_status(sim_id) == "failed":
                    log.warning("[MiroFish] Simulation preparation failed server-side")
                    return "failed"
                log.debug("[MiroFish] preparing… status=%s", status)
            except Exception as exc:
                log.debug("[MiroFish] prepare poll error: %s", exc)
            time.sleep(4.0)
        return "timeout"

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
        """
        Poll until the simulation rounds are done.

        The reddit runner idles in wait-for-commands mode after its round loop
        (run-status stays 'running' forever — it never writes the actions.jsonl
        events the server's monitor needs to flip it to 'completed'). The
        reliable rounds-done signal is env_alive=True from /env-status, which
        the runner only sets after the loop finishes; on seeing it we stop the
        simulation to release the process and proceed to the report.
        """
        while time.monotonic() < deadline:
            try:
                result = self._get(f"/api/simulation/{sim_id}/run-status")
                data = result.get("data", {})
                status = data.get("runner_status", "")
                if status in ("completed", "stopped"):
                    return True
                if status == "failed":
                    log.warning("[MiroFish] Simulation failed server-side: %s", data.get("error"))
                    return False
                log.debug("[MiroFish] polling runner_status=%s", status)
            except Exception as exc:
                log.debug("[MiroFish] run-status poll error: %s", exc)

            try:
                env = self._post("/api/simulation/env-status", json={"simulation_id": sim_id})
                if env.get("data", {}).get("env_alive"):
                    log.info("[MiroFish] Rounds complete (env in wait-for-commands mode) — stopping simulation")
                    try:
                        self._post("/api/simulation/stop", json={"simulation_id": sim_id})
                    except Exception as exc:
                        log.debug("[MiroFish] stop request failed: %s", exc)
                    return True
            except Exception as exc:
                log.debug("[MiroFish] env-status poll error: %s", exc)

            time.sleep(6.0)
        return False

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
        timeout_secs: int = 900,
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

            # Zep entity classification is stochastic: the same project text can
            # yield a graph with 0 classified entities, which makes preparation
            # fail instantly. Rebuilding the graph re-runs extraction, so retry
            # the build->create->prepare leg once before giving up.
            sim_id = None
            max_build_attempts = 2
            for attempt in range(1, max_build_attempts + 1):
                graph_id = self._build_graph(project_id, deadline)
                if not graph_id:
                    log.warning("[MiroFish] Graph build timed out or failed")
                    return None
                log.info("[MiroFish] graph_id=%s (attempt %d)", graph_id, attempt)

                candidate = self._create_simulation(project_id, graph_id)
                log.info("[MiroFish] sim_id=%s", candidate)
                if time.monotonic() > deadline:
                    return None

                prep = self._prepare_simulation(candidate, deadline)
                if prep == "ready":
                    sim_id = candidate
                    break
                if prep in ("no_entities", "failed") and attempt < max_build_attempts and time.monotonic() < deadline:
                    log.warning("[MiroFish] Preparation %s on attempt %d — rebuilding graph", prep, attempt)
                    continue
                log.warning("[MiroFish] Simulation preparation %s", prep)
                return None
            if sim_id is None:
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
