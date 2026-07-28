"""Rebuilding counterfactual_daily.jsonl from clean sources.

The live log accumulated three defects (audit 2026-07-27):
  - 45 rows for 78 trading days, because backfills could not insert dates
  - Track B keyed one day late, from a tz-less datetime.fromtimestamp() on
    Alpaca's UTC epochs (same-day corr with A* was -0.005, +0.60 at lag 1)
  - a 2026-06-19 row carrying non-zero returns on Juneteenth, a market holiday

The date-shift and insertion bugs are fixed at source, but the existing rows were
written under the old behaviour, so the history has to be re-derived. These tests
pin the rebuild's contract: trading days only, one row per date, every value
traceable to a source, and nothing invented where a source is missing.
"""

import datetime as dt

import pytest

from ascent.monitoring.counterfactual_rebuild import (
    build_price_map,
    rebuild_rows,
    snapshot_asof,
)


# ── snapshot_asof ─────────────────────────────────────────────────────────────

class TestSnapshotAsof:
    SNAPS = [
        {"date": "2026-04-15", "weights": {"AAPL": 1.0}},
        {"date": "2026-05-05", "weights": {"MSFT": 1.0}},
    ]

    def test_picks_the_most_recent_on_or_before(self):
        assert snapshot_asof(self.SNAPS, "2026-05-04") == {"AAPL": 1.0}
        assert snapshot_asof(self.SNAPS, "2026-05-05") == {"MSFT": 1.0}
        assert snapshot_asof(self.SNAPS, "2026-06-01") == {"MSFT": 1.0}

    def test_returns_none_before_the_first_snapshot(self):
        """A track with no snapshot yet must be None, not silently zero — a
        fabricated 0.0 freezes the track while others accrue."""
        assert snapshot_asof(self.SNAPS, "2026-04-14") is None

    def test_empty_snapshots_yield_none(self):
        assert snapshot_asof([], "2026-05-01") is None


# ── build_price_map ───────────────────────────────────────────────────────────

class TestBuildPriceMap:
    CLOSES = {
        "AAPL": {"2026-05-04": 100.0, "2026-05-05": 110.0},
        "MSFT": {"2026-05-04": 200.0, "2026-05-05": 190.0},
    }

    def test_pairs_previous_and_current_close(self):
        pm = build_price_map(self.CLOSES, "2026-05-05", "2026-05-04")
        assert pm["AAPL"] == {"prev": 100.0, "curr": 110.0}
        assert pm["MSFT"] == {"prev": 200.0, "curr": 190.0}

    def test_symbol_missing_a_close_is_omitted(self):
        pm = build_price_map({"AAPL": {"2026-05-05": 110.0}}, "2026-05-05", "2026-05-04")
        assert "AAPL" not in pm


# ── rebuild_rows ──────────────────────────────────────────────────────────────

def _closes(dates, start=100.0, step=1.0):
    """Flat +1/day series for two symbols so returns are easy to reason about."""
    out = {}
    for sym in ("AAPL", "SPY"):
        out[sym] = {d: start + i * step for i, d in enumerate(dates)}
    return out


class TestRebuildRows:
    DAYS = ["2026-06-17", "2026-06-18", "2026-06-19", "2026-06-22", "2026-06-23"]

    def _args(self, **over):
        base = dict(
            trading_days=[d for d in self.DAYS if d != "2026-06-19"],
            closes=_closes(self.DAYS),
            astar_snaps=[{"date": "2026-06-17", "weights": {"AAPL": 1.0}}],
            a_snaps=[],
            d_snaps=[],
            track_b={"2026-06-18": 0.01, "2026-06-22": 0.02},
            spy_symbol="SPY",
        )
        base.update(over)
        return base

    def test_excludes_market_holidays(self):
        """Juneteenth 2026-06-19 carried non-zero returns in the live log."""
        rows = rebuild_rows(**self._args())
        assert "2026-06-19" not in {r["date"] for r in rows}

    def test_one_row_per_date_in_order(self):
        rows = rebuild_rows(**self._args())
        ds = [r["date"] for r in rows]
        assert ds == sorted(ds)
        assert len(ds) == len(set(ds))

    def test_track_b_comes_from_the_settled_series(self):
        rows = {r["date"]: r for r in rebuild_rows(**self._args())}
        assert rows["2026-06-18"]["track_b_return"] == 0.01
        assert rows["2026-06-22"]["track_b_return"] == 0.02

    def test_track_b_is_none_where_alpaca_has_no_bar(self):
        """Absent is not zero."""
        rows = {r["date"]: r for r in rebuild_rows(**self._args())}
        assert rows["2026-06-23"]["track_b_return"] is None

    def test_first_day_of_the_window_is_dropped(self):
        """No prior close exists, so no track has a return — an all-null row
        would make a gap look like data."""
        rows = {r["date"]: r for r in rebuild_rows(**self._args())}
        assert "2026-06-17" not in rows
        assert rows["2026-06-18"]["track_astar_return"] is not None

    def test_astar_is_none_before_its_first_snapshot(self):
        rows = {r["date"]: r for r in rebuild_rows(
            **self._args(astar_snaps=[{"date": "2026-06-22",
                                       "weights": {"AAPL": 1.0}}]))}
        assert rows["2026-06-18"]["track_astar_return"] is None
        assert rows["2026-06-22"]["track_astar_return"] is not None

    def test_astar_return_matches_the_underlying_price_move(self):
        rows = {r["date"]: r for r in rebuild_rows(**self._args())}
        # AAPL 100 -> 101 on the 18th, 100% weight
        assert rows["2026-06-18"]["track_astar_return"] == pytest.approx(0.01, abs=1e-6)

    def test_tracks_without_snapshots_stay_none(self):
        rows = rebuild_rows(**self._args())
        assert all(r["track_a_return"] is None for r in rows)
        assert all(r["track_d_return"] is None for r in rows)

    def test_rows_carry_provenance(self):
        rows = rebuild_rows(**self._args())
        assert all(r.get("source") == "rebuild" for r in rows)

    def test_days_with_no_computable_track_are_dropped(self):
        rows = rebuild_rows(**self._args(track_b={}, astar_snaps=[], closes={}))
        assert rows == []

    def test_spy_drives_track_c(self):
        rows = {r["date"]: r for r in rebuild_rows(**self._args())}
        assert rows["2026-06-18"]["track_c_return"] == pytest.approx(0.01, abs=1e-6)

    def test_no_weekend_dates_ever(self):
        rows = rebuild_rows(**self._args())
        for r in rows:
            assert dt.date.fromisoformat(r["date"]).weekday() < 5
