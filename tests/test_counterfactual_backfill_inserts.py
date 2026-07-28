"""backfill_track_b must be able to INSERT missing dates, not only patch rows.

Audit, 2026-07-27. logs/counterfactual_daily.jsonl held 45 rows for an
81-business-day window — 36 days simply absent, with a 19-day hole from the
outage. Both backfills iterate the existing lines and mutate fields on rows that
are already there:

    for line in DAILY_LOG.read_text().splitlines():
        ...
        if d in history and r.get("track_b_return") != history[d]:

A settled Alpaca date with no corresponding row is silently discarded — there is
no `else: rows.append(...)`. So the holes are permanent by design, and
`_cumret_over` chains straight across them as though those days never happened.
That is what inflated every cumulative figure 2.5-4x (Track C claimed SPY
+16.63% against an actual +5.31%).

Inserting a Track-B-only row is honest: the row carries the settled Alpaca return
and leaves the other tracks None, so `_common_window_diff`, which pairs
non-null observations, simply does not count that day for A*/D comparisons.
"""

import json

import pytest

import ascent.monitoring.ai_pm_counterfactual as cf


@pytest.fixture
def log(tmp_path, monkeypatch):
    p = tmp_path / "counterfactual_daily.jsonl"
    monkeypatch.setattr(cf, "DAILY_LOG", p)
    return p


def _write(p, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _dates(p):
    """Dates of the parseable rows. Unparseable lines are preserved on disk by
    design (dropping them would lose data), so they are skipped here."""
    out = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line)["date"])
        except Exception:
            continue
    return out


class TestPatchingStillWorks:
    def test_existing_row_is_updated(self, log):
        _write(log, [{"date": "2026-07-24", "track_b_return": 0.0}])
        n = cf.backfill_track_b({"2026-07-24": 0.0123})
        assert n == 1
        assert json.loads(log.read_text().splitlines()[0])["track_b_return"] == 0.0123

    def test_idempotent(self, log):
        _write(log, [{"date": "2026-07-24", "track_b_return": 0.0123}])
        assert cf.backfill_track_b({"2026-07-24": 0.0123}) == 0


class TestInsertion:
    def test_missing_date_is_inserted(self, log):
        _write(log, [{"date": "2026-07-24", "track_b_return": 0.01}])
        n = cf.backfill_track_b({"2026-07-24": 0.01, "2026-07-27": 0.02})
        assert n == 1
        assert "2026-07-27" in _dates(log)

    def test_inserted_row_carries_only_track_b(self, log):
        """Other tracks must stay None rather than being invented."""
        _write(log, [{"date": "2026-07-24", "track_b_return": 0.01}])
        cf.backfill_track_b({"2026-07-27": 0.02})
        row = [json.loads(l) for l in log.read_text().splitlines()
               if json.loads(l)["date"] == "2026-07-27"][0]
        assert row["track_b_return"] == 0.02
        for k in ("track_astar_return", "track_a_return", "track_d_return"):
            assert row.get(k) is None

    def test_rows_are_written_in_date_order(self, log):
        _write(log, [{"date": "2026-07-27", "track_b_return": 0.02}])
        cf.backfill_track_b({"2026-07-24": 0.01, "2026-07-27": 0.02})
        assert _dates(log) == ["2026-07-24", "2026-07-27"]

    def test_fills_a_multi_day_hole(self, log):
        """The real case: a 19-day outage gap."""
        _write(log, [{"date": "2026-06-29", "track_b_return": 0.001}])
        hist = {"2026-06-29": 0.001}
        hist.update({f"2026-07-{d:02d}": 0.002 for d in range(1, 25)})
        n = cf.backfill_track_b(hist)
        assert n == 24
        assert len(_dates(log)) == 25
        assert _dates(log) == sorted(_dates(log))

    def test_no_duplicate_dates_ever(self, log):
        _write(log, [{"date": "2026-07-24", "track_b_return": 0.01}])
        cf.backfill_track_b({"2026-07-24": 0.09, "2026-07-27": 0.02})
        cf.backfill_track_b({"2026-07-24": 0.09, "2026-07-27": 0.02})
        d = _dates(log)
        assert len(d) == len(set(d))


class TestGuards:
    def test_empty_history_is_a_noop(self, log):
        _write(log, [{"date": "2026-07-24", "track_b_return": 0.01}])
        assert cf.backfill_track_b({}) == 0
        assert _dates(log) == ["2026-07-24"]

    def test_creates_the_log_when_absent(self, log):
        """Previously returned 0 if the file did not exist, so a first run after
        an outage could never seed it."""
        assert not log.exists()
        n = cf.backfill_track_b({"2026-07-27": 0.02})
        assert n == 1
        assert _dates(log) == ["2026-07-27"]

    def test_malformed_lines_are_preserved_not_dropped(self, log):
        log.write_text('{"date": "2026-07-24", "track_b_return": 0.01}\nnot json\n')
        cf.backfill_track_b({"2026-07-27": 0.02})
        assert "2026-07-27" in _dates(log)
