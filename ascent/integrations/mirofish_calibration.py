# ascent/integrations/mirofish_calibration.py
from __future__ import annotations

import json
import logging
import statistics
from datetime import date
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAL_PATH = _REPO_ROOT / "data_cache" / "mirofish_calibration.json"
_ANALOGUES_PATH = _REPO_ROOT / "data_cache" / "mirofish_analogues.json"


def _load_calibration() -> dict[str, Any]:
    try:
        if _CAL_PATH.exists():
            return json.loads(_CAL_PATH.read_text())
    except Exception as exc:
        log.debug("[MiroFishCal] Load failed: %s", exc)
    return {"bootstrapped": False, "entries": []}


def _save_calibration(data: dict[str, Any]) -> None:
    try:
        _CAL_PATH.write_text(json.dumps(data, indent=2))
    except Exception as exc:
        log.warning("[MiroFishCal] Save failed: %s", exc)


def bootstrap_calibration() -> None:
    """Populate calibration from the curated analogues library on first run."""
    cal = _load_calibration()
    if cal.get("bootstrapped"):
        return
    try:
        analogues = json.loads(_ANALOGUES_PATH.read_text()) if _ANALOGUES_PATH.exists() else []
    except Exception:
        analogues = []

    for a in analogues:
        label = a.get("sentiment_label", "mixed")
        for sector in a.get("affected_sectors", ["unknown"]):
            for sym, ret in a.get("realized_21d_returns", {}).items():
                if sym == "SPY":
                    continue
                cal["entries"].append({
                    "event_id": a["event_id"],
                    "sentiment_label": label,
                    "sector": sector,
                    "symbol": sym,
                    "realized_21d_return": float(ret),
                    "recorded_at": date.today().isoformat(),
                })
    cal["bootstrapped"] = True
    _save_calibration(cal)
    log.info("[MiroFishCal] Bootstrapped %d calibration entries", len(cal["entries"]))


def get_base_rate(
    sentiment_label: str,
    sector: str | None = None,
) -> dict[str, Any]:
    """
    Return historical base rate for a given sentiment label and optional sector.

    Returns:
        {n_events, median_21d_return, positive_rate, sentiment_label}
    If fewer than 2 matching entries, falls back to all-label entries.
    """
    cal = _load_calibration()
    entries = cal.get("entries", [])

    def _filter(entries_: list, label: str, sec: str | None) -> list[float]:
        filtered = [e for e in entries_ if e.get("sentiment_label") == label]
        if sec:
            sector_filtered = [e for e in filtered if e.get("sector") == sec]
            if len(sector_filtered) >= 2:
                filtered = sector_filtered
        return [e["realized_21d_return"] for e in filtered]

    returns = _filter(entries, sentiment_label, sector)
    if not returns:
        return {
            "n_events": 0,
            "median_21d_return": None,
            "positive_rate": None,
            "sentiment_label": sentiment_label,
        }
    return {
        "n_events": len(returns),
        "median_21d_return": statistics.median(returns),
        "positive_rate": sum(1 for r in returns if r > 0) / len(returns),
        "sentiment_label": sentiment_label,
    }


def record_entry(
    event_id: str,
    sentiment_label: str,
    sector: str,
    realized_21d_return: float,
) -> None:
    """Append a new realized-return entry after a rebalance cycle closes."""
    cal = _load_calibration()
    cal.setdefault("entries", []).append({
        "event_id": event_id,
        "sentiment_label": sentiment_label,
        "sector": sector,
        "realized_21d_return": float(realized_21d_return),
        "recorded_at": date.today().isoformat(),
    })
    _save_calibration(cal)
