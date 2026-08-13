"""Canonical series registry.

One name, one series, forever. Five separate artifacts came to answer the single
counterfactual question -- ai_pm_counterfactual, counterfactual_tracker,
counterfactual_rebuild, and two scripts -- because the question had no canonical
address. A new reader hand-wrote its own path and column handling every time.

This is a read-only lens over files that already exist. It moves no bytes and
owns no data. Registering a series here does not change how it is written.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

log = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parent.parent.parent.parent
_DAILY = _REPO / "logs" / "counterfactual_daily.jsonl"


@dataclass(frozen=True)
class Series:
    """A named series and everything needed to read and trust it."""
    name: str
    description: str
    source: Path
    column: str
    index_kind: str   # market_trading_day | calendar_day | utc_timestamp
    coverage: str     # the completeness invariant this series must satisfy
    provenance: str   # where the values ultimately come from


_COVERAGE = "every NYSE session from first to last row"

SERIES = {
    s.name: s for s in [
        Series(
            name="counterfactual.track_astar",
            description="Pure quant daily return, zero Phase 1 influence.",
            source=_DAILY, column="track_astar_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="reconstruction: snapshot weights priced on prices_live",
        ),
        Series(
            name="counterfactual.track_a",
            description=(
                "Quant plus Phase 1 sleeve priors. KNOWN DEFECT: structurally "
                "identical to track_astar -- both read the same post-orchestrator "
                "merged_weights, so this series measures nothing. Do not cite it."
            ),
            source=_DAILY, column="track_a_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="reconstruction: snapshot weights priced on prices_live",
        ),
        Series(
            name="counterfactual.track_b",
            description="Actual traded book daily return.",
            source=_DAILY, column="track_b_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="settled Alpaca 1D bars; total-return basis",
        ),
        Series(
            name="counterfactual.track_c",
            description="SPY benchmark daily return.",
            source=_DAILY, column="track_c_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="prices_live closes; split-only basis",
        ),
        Series(
            name="counterfactual.track_d",
            description="Pure AI PM portfolio daily return, pre-blend.",
            source=_DAILY, column="track_d_return",
            index_kind="market_trading_day", coverage=_COVERAGE,
            provenance="reconstruction: snapshot weights priced on prices_live",
        ),
    ]
}


def names() -> List[str]:
    """Every registered canonical name, sorted."""
    return sorted(SERIES)


def describe(name: str) -> Series:
    """The descriptor for one series. Raises KeyError on an unknown name."""
    if name not in SERIES:
        raise KeyError(f"unknown series {name!r}; known: {names()}")
    return SERIES[name]


def load(name: str) -> pd.Series:
    """Read one series as float values indexed by date, sorted, nulls dropped."""
    s = describe(name)
    if not s.source.exists():
        raise FileNotFoundError(f"{name}: source missing at {s.source}")

    values = {}
    for line in s.source.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        raw = row.get(s.column)
        if raw is None:
            continue
        try:
            values[date.fromisoformat(row["date"])] = float(raw)
        except (KeyError, ValueError, TypeError):
            continue

    out = pd.Series(values, dtype="float64", name=name)
    return out.sort_index()
