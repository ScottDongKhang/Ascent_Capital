"""A scheduled run must not re-process a session it has already traded.

The 2026-07-27 audit found duplicate rows for the same date throughout the logs
(logs/ai_pm_decision_log.jsonl carries 9 rows across only 2 distinct dates,
with 2026-06-10 appearing 8 times). Those duplicates inflate win rates and
double-count returns.

The window is real: the job fires at 09:00 local (UTC+7), which resolves to the
*previous* US session. A manual catch-up run earlier the same session therefore
collides with the scheduled one.
"""

import datetime as dt
import json

from run_all_agents import already_ran_for_session


def _write_log(tmp_path, dates):
    p = tmp_path / "eod_log.jsonl"
    with p.open("w") as f:
        for d in dates:
            f.write(json.dumps({"date": d, "source": "multi_agent", "rebalanced": True}) + "\n")
    return p


class TestAlreadyRanForSession:
    def test_true_when_a_row_exists_for_that_session(self, tmp_path):
        log = _write_log(tmp_path, ["2026-07-24", "2026-07-27"])
        assert already_ran_for_session(dt.date(2026, 7, 27), log_path=log) is True

    def test_false_when_no_row_exists_for_that_session(self, tmp_path):
        log = _write_log(tmp_path, ["2026-07-24", "2026-07-27"])
        assert already_ran_for_session(dt.date(2026, 7, 28), log_path=log) is False

    def test_false_when_log_is_missing(self, tmp_path):
        """Fail OPEN: a missing log must not block the first ever run."""
        assert already_ran_for_session(
            dt.date(2026, 7, 27), log_path=tmp_path / "nope.jsonl"
        ) is False

    def test_ignores_malformed_lines(self, tmp_path):
        p = tmp_path / "eod_log.jsonl"
        p.write_text('not json\n{"date": "2026-07-27"}\n\n')
        assert already_ran_for_session(dt.date(2026, 7, 27), log_path=p) is True
        assert already_ran_for_session(dt.date(2026, 7, 28), log_path=p) is False

    def test_accepts_run_date_as_fallback_key(self, tmp_path):
        p = tmp_path / "eod_log.jsonl"
        p.write_text(json.dumps({"run_date": "2026-07-27"}) + "\n")
        assert already_ran_for_session(dt.date(2026, 7, 27), log_path=p) is True

    def test_ignores_discovery_candidate_rows(self, tmp_path):
        """eod_log also holds discovery-candidate objects (keys: symbol/trigger/
        conviction) that are not run records. Those must not count as a run."""
        p = tmp_path / "eod_log.jsonl"
        p.write_text(json.dumps(
            {"date": "2026-07-28", "trigger": "discovery", "symbol": "PDBC", "conviction": 0.82}
        ) + "\n")
        assert already_ran_for_session(dt.date(2026, 7, 28), log_path=p) is False
