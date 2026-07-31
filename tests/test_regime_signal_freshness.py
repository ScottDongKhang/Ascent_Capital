"""dashboard/regime_signal.json must expire before it steers reporting output.

Three regime sources can disagree: dashboard/regime_labels.csv (the regime
engine's own live output), dashboard/regime_signal.json, and
data_cache/ai_regime_assessment.json. In one audit regime_signal.json sat 5
weeks stale while four reporting/monitoring call sites
(investor_report.py, investor_letter.py, weekly_debrief.py,
scenario_planner.py) read it with no freshness check at all and silently
published its label as current.

`ascent.utils.freshness.fresh_regime_label()` closes that gap: it applies the
same fail-closed gate as `ai_prior_is_fresh` to regime_signal.json's `as_of`,
and on staleness/absence falls back to `regime_label_from_csv()` — the
regime engine's own regime_labels.csv, which needs no freshness question
since it is always as current as the last pipeline run.
"""

import csv
import datetime as dt
import json

import pytest

from ascent.utils.freshness import (
    AI_PRIOR_MAX_AGE_DAYS,
    fresh_regime_label,
    regime_label_from_csv,
)

TODAY = dt.date(2026, 7, 31)


def _write_signal(path, as_of, regime="calm_bull", key="as_of"):
    path.write_text(json.dumps({"regime": regime, key: as_of}))


def _write_labels_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "label", "confidence"])
        for d, label in rows:
            w.writerow([d, label, 0.99])


class TestRegimeLabelFromCsv:
    def test_reads_last_row_label(self, tmp_path):
        csv_path = tmp_path / "regime_labels.csv"
        _write_labels_csv(csv_path, [("2026-07-01", "stressed"), ("2026-07-27", "calm_bull")])
        assert regime_label_from_csv(csv_path) == "calm_bull"

    def test_missing_file_returns_none(self, tmp_path):
        assert regime_label_from_csv(tmp_path / "does_not_exist.csv") is None

    def test_empty_file_returns_none(self, tmp_path):
        csv_path = tmp_path / "regime_labels.csv"
        _write_labels_csv(csv_path, [])
        assert regime_label_from_csv(csv_path) is None


class TestFreshRegimeLabel:
    def test_fresh_signal_is_trusted_as_is(self, tmp_path):
        """Reproduces the non-bug case: a current regime_signal.json is used
        directly and never marked stale."""
        signal = tmp_path / "regime_signal.json"
        csv_path = tmp_path / "regime_labels.csv"
        _write_signal(signal, "2026-07-31", regime="stressed")
        _write_labels_csv(csv_path, [("2026-07-31", "calm_bull")])

        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)
        assert result == {
            "label": "stressed",
            "stale": False,
            "age_days": 0,
            "source": "signal",
        }

    def test_the_audit_incident_5_weeks_stale_falls_back_to_csv(self, tmp_path):
        """Reproduces the bug: a regime_signal.json stamped 35 days old (the
        audited '5 weeks stale' incident) must NOT be silently trusted. Before
        this fix, every call site did `json.loads(...).get('regime')` with no
        date check at all, so a stale label would have been returned here as
        if it were current. The fix must fall back to the engine's live CSV
        and flag the result as stale.
        """
        signal = tmp_path / "regime_signal.json"
        csv_path = tmp_path / "regime_labels.csv"
        stale_date = (TODAY - dt.timedelta(days=35)).isoformat()
        _write_signal(signal, stale_date, regime="stressed")
        _write_labels_csv(csv_path, [(TODAY.isoformat(), "calm_bull")])

        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)

        assert result["stale"] is True
        assert result["age_days"] == 35
        assert result["source"] == "csv_fallback"
        # The fallback label is the engine's live truth, NOT the stale signal's label.
        assert result["label"] == "calm_bull"
        assert result["label"] != "stressed"

    def test_boundary_is_inclusive(self, tmp_path):
        signal = tmp_path / "regime_signal.json"
        csv_path = tmp_path / "regime_labels.csv"
        edge_date = (TODAY - dt.timedelta(days=AI_PRIOR_MAX_AGE_DAYS)).isoformat()
        _write_signal(signal, edge_date, regime="stressed")
        _write_labels_csv(csv_path, [(TODAY.isoformat(), "calm_bull")])
        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)
        assert result["stale"] is False
        assert result["label"] == "stressed"

    def test_missing_signal_file_falls_back_to_csv(self, tmp_path):
        signal = tmp_path / "does_not_exist.json"
        csv_path = tmp_path / "regime_labels.csv"
        _write_labels_csv(csv_path, [(TODAY.isoformat(), "crisis")])
        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)
        assert result["stale"] is True
        assert result["source"] == "csv_fallback"
        assert result["label"] == "crisis"

    def test_unparseable_signal_falls_back_to_csv(self, tmp_path):
        signal = tmp_path / "regime_signal.json"
        signal.write_text("{not valid json")
        csv_path = tmp_path / "regime_labels.csv"
        _write_labels_csv(csv_path, [(TODAY.isoformat(), "crisis")])
        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)
        assert result["stale"] is True
        assert result["label"] == "crisis"

    def test_both_signal_and_csv_missing_reports_unknown_stale(self, tmp_path):
        signal = tmp_path / "does_not_exist.json"
        csv_path = tmp_path / "also_missing.csv"
        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)
        assert result == {"label": "unknown", "stale": True, "age_days": None, "source": "unknown"}

    def test_future_dated_signal_is_treated_as_stale(self, tmp_path):
        """Fails CLOSED, matching ai_prior_is_fresh: a future as_of is a clock
        or override bug, not signal, and must not be trusted."""
        signal = tmp_path / "regime_signal.json"
        csv_path = tmp_path / "regime_labels.csv"
        future = (TODAY + dt.timedelta(days=1)).isoformat()
        _write_signal(signal, future, regime="stressed")
        _write_labels_csv(csv_path, [(TODAY.isoformat(), "calm_bull")])
        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)
        assert result["stale"] is True
        assert result["label"] == "calm_bull"

    def test_falls_back_to_last_refit_date_key(self, tmp_path):
        """regime_signal.json carries both as_of and last_refit_date; the
        loader must accept last_refit_date when as_of is absent."""
        signal = tmp_path / "regime_signal.json"
        csv_path = tmp_path / "regime_labels.csv"
        _write_signal(signal, TODAY.isoformat(), regime="calm_bull", key="last_refit_date")
        _write_labels_csv(csv_path, [(TODAY.isoformat(), "stressed")])
        result = fresh_regime_label(signal_path=signal, csv_fallback_path=csv_path, today=TODAY)
        assert result["stale"] is False
        assert result["label"] == "calm_bull"


class TestWiredIntoReportingCallSites:
    """The helper is only useful if the four reporting/monitoring call sites
    actually consult it instead of reading regime_signal.json raw."""

    def _src(self, rel):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, rel)) as f:
            return f.read()

    @pytest.mark.parametrize("rel", [
        "ascent/reporting/investor_report.py",
        "ascent/reporting/investor_letter.py",
        "ascent/monitoring/weekly_debrief.py",
        "ascent/monitoring/scenario_planner.py",
    ])
    def test_call_site_uses_fresh_regime_label(self, rel):
        src = self._src(rel)
        assert "fresh_regime_label" in src, (
            f"{rel} must read regime_signal.json through fresh_regime_label(), "
            "not a raw json.loads with no freshness check"
        )
