import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

log = logging.getLogger(__name__)

_FOMC_2026 = [
    "2026-01-29", "2026-03-19", "2026-05-07", "2026-06-18",
    "2026-07-30", "2026-09-17", "2026-11-05", "2026-12-17",
]
_CPI_2026 = [
    "2026-01-14", "2026-02-11", "2026-03-11", "2026-04-10",
    "2026-05-13", "2026-06-11", "2026-07-14", "2026-08-12",
    "2026-09-11", "2026-10-13", "2026-11-12", "2026-12-11",
]
_NFP_2026 = [
    "2026-01-09", "2026-02-06", "2026-03-06", "2026-04-03",
    "2026-05-01", "2026-06-05", "2026-07-10", "2026-08-07",
    "2026-09-04", "2026-10-02", "2026-11-06", "2026-12-04",
]

_SYSTEM = (
    "You are a macro risk analyst at an institutional fund. "
    "For each upcoming event, rate the portfolio's sensitivity from 1 (minimal impact) "
    "to 5 (high impact) and explain in one sentence which positions are most exposed. "
    "Return a JSON array: [{\"event\": str, \"date\": str, \"days_away\": int, "
    "\"sensitivity\": int, \"impact\": str}]"
)


def _upcoming_macro_events(today: date, horizon_days: int = 10) -> List[Dict]:
    events = []
    end = today + timedelta(days=horizon_days)
    for d_str in _FOMC_2026 + _CPI_2026 + _NFP_2026:
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        if today < d <= end:
            label = ("FOMC Decision" if d_str in _FOMC_2026
                     else "CPI Release" if d_str in _CPI_2026 else "NFP Release")
            events.append({"event": label, "date": d_str, "days_away": (d - today).days})
    return sorted(events, key=lambda x: x["days_away"])


def _load_earnings_events(today: date, merged_weights: Dict, horizon_days: int = 10) -> List[Dict]:
    cache_path = Path("data_cache/earnings_cache.json")
    if not cache_path.exists():
        return []
    try:
        data = json.loads(cache_path.read_text())
    except Exception:
        return []
    end = today + timedelta(days=horizon_days)
    events = []
    for sym in merged_weights:
        entry = data.get(sym, {})
        d_str = entry.get("report_date") or entry.get("next_earnings_date")
        if not d_str:
            continue
        try:
            d = date.fromisoformat(str(d_str)[:10])
        except ValueError:
            continue
        if today < d <= end:
            events.append({"event": f"{sym} Earnings", "date": str(d), "days_away": (d - today).days})
    return events


def build_event_calendar(
    date_str: str,
    merged_weights: Dict[str, float],
    agent_outputs: list,
) -> List[Dict]:
    """
    Returns list of upcoming macro/earnings events with Haiku-scored sensitivity.
    Returns raw event list (without scores) on LLM failure. Returns [] on date parse failure.
    """
    try:
        today = date.fromisoformat(date_str)
    except ValueError:
        return []

    events = _upcoming_macro_events(today) + _load_earnings_events(today, merged_weights)
    if not events:
        return events

    try:
        from ascent.llm.client import generate_structured, HAIKU_MODEL
    except ImportError:
        return events

    top_positions = sorted(merged_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    pos_str = ", ".join(f"{s} ({w:.1%})" for s, w in top_positions)
    events_str = "\n".join(
        f"- {e['event']} on {e['date']} ({e['days_away']} days away)" for e in events
    )

    user_prompt = (
        f"Portfolio top positions: {pos_str}\n\n"
        f"Upcoming events:\n{events_str}\n\n"
        "Score each event's portfolio sensitivity and identify exposed positions."
    )

    try:
        raw = generate_structured(
            system_prompt=_SYSTEM,
            user_prompt=user_prompt,
            model=HAIKU_MODEL,
            max_tokens=800,
            temperature=0.3,
            use_cache=True,
        )
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            return events
        return json.loads(raw[start:end])
    except Exception as e:
        log.warning("[MacroCalendar] Haiku scoring failed (%s) — returning raw events", e)
        return events
