"""Regression tests for ascent/reporting/investor_letter.py.

Two audit findings, both fixed here:

  §1.7 — `_get_regime()` read dashboard/regime_signal.json with no freshness
  check, so a stale label would be published in the investor letter as if
  current. Fixed via `ascent.utils.freshness.fresh_regime_label()`.

  §1.3 — `WF_SHARPE = 0.52` / `WF_PERIOD = "Jan 2020-Apr 2026"` were hardcoded
  constants matching no artifact (CURRENT_VERIFIED_NUMBERS.md cites a
  different, canonical figure). Fixed by reading
  `ascent.reporting.verified_numbers.canonical_wf()` at generation time.
"""
import datetime as dt
import json

import pytest

import ascent.reporting.investor_letter as investor_letter
import ascent.utils.freshness as freshness


# ── §1.7: regime freshness ────────────────────────────────────────────────

class TestGetRegime:
    def test_fresh_signal_returns_plain_label(self, tmp_path, monkeypatch):
        signal = tmp_path / "regime_signal.json"
        today = dt.date(2026, 7, 31)
        signal.write_text(json.dumps({"regime": "calm_bull", "as_of": today.isoformat()}))
        monkeypatch.setattr(investor_letter, "REGIME_PATH", signal)
        monkeypatch.setattr(freshness, "market_today", lambda now=None: today)

        assert investor_letter._get_regime() == "calm_bull"

    def test_stale_signal_falls_back_and_is_annotated(self, tmp_path, monkeypatch):
        """Reproduces the bug: before this fix, _get_regime() did
        `json.loads(REGIME_PATH.read_text()).get('regime', 'unknown')` with no
        date check, so a 5-week-stale label would be returned unmarked. Now it
        must fall back to regime_labels.csv and say so in the string, so the
        investor letter never presents a stale label as current fact."""
        signal = tmp_path / "regime_signal.json"
        csv_path = tmp_path / "regime_labels.csv"
        today = dt.date(2026, 7, 31)
        stale_date = (today - dt.timedelta(days=35)).isoformat()
        signal.write_text(json.dumps({"regime": "stressed", "as_of": stale_date}))
        with open(csv_path, "w") as f:
            f.write("date,label,confidence\n")
            f.write(f"{today.isoformat()},calm_bull,0.99\n")

        monkeypatch.setattr(investor_letter, "REGIME_PATH", signal)
        monkeypatch.setattr(freshness, "REGIME_LABELS_CSV", csv_path)
        monkeypatch.setattr(freshness, "market_today", lambda now=None: today)

        result = investor_letter._get_regime()
        assert "calm_bull" in result
        assert "stale" in result
        assert "stressed" not in result  # the stale label must not leak through

    def test_missing_signal_and_csv_reports_unknown_stale(self, tmp_path, monkeypatch):
        monkeypatch.setattr(investor_letter, "REGIME_PATH", tmp_path / "nope.json")
        monkeypatch.setattr(freshness, "REGIME_LABELS_CSV", tmp_path / "also_nope.csv")

        result = investor_letter._get_regime()
        assert result.startswith("unknown")
        assert "stale" in result


# ── §1.3: unsourced Sharpe ─────────────────────────────────────────────────

class TestCanonicalSharpeSourcing:
    def test_no_hardcoded_wf_sharpe_constant_remains(self):
        assert not hasattr(investor_letter, "WF_SHARPE"), (
            "WF_SHARPE was a hardcoded, unsourced constant (0.52) matching no "
            "wf_report artifact — it must be removed, not just unused"
        )
        assert not hasattr(investor_letter, "WF_PERIOD")

    def test_reads_canonical_wf_sharpe(self, monkeypatch):
        """The letter must source its Sharpe from the canonical loader, not a
        module constant."""
        from ascent.reporting.verified_numbers import WalkForwardRecord

        fake = WalkForwardRecord(
            artifact="outputs/wf_results/wf_report_clean_2026-06-22.json",
            sharpe=0.4118, cagr=0.103, max_drawdown=-0.329, beta=0.733,
            alpha=0.0224, win_rate=0.5, wfe=-0.65, volatility=0.1,
            n_folds=21, n_oos_days=1134, oos_window="2021-01-08 -> 2026-01-14",
        )
        monkeypatch.setattr(
            "ascent.reporting.verified_numbers.canonical_wf", lambda: fake
        )
        sharpe_str, period = investor_letter._canonical_wf_sharpe_and_period()
        assert sharpe_str == "0.41"
        assert period == "2021-01-08 -> 2026-01-14"

    def test_missing_artifact_reports_unavailable_not_a_fabricated_default(self, monkeypatch):
        from ascent.reporting.verified_numbers import MissingArtifact

        def _raise():
            raise MissingArtifact("no artifact")

        monkeypatch.setattr(
            "ascent.reporting.verified_numbers.canonical_wf", _raise
        )
        sharpe_str, period = investor_letter._canonical_wf_sharpe_and_period()
        assert sharpe_str == "unavailable"
        assert period == "unavailable"

    def test_prompt_contains_canonical_sharpe_not_old_hardcoded_value(self, monkeypatch):
        """End-to-end: the built user prompt must carry the canonical figure,
        and must NOT contain the old dead constant 0.52."""
        from ascent.reporting.verified_numbers import WalkForwardRecord

        fake = WalkForwardRecord(
            artifact="outputs/wf_results/wf_report_clean_2026-06-22.json",
            sharpe=0.4118, cagr=0.103, max_drawdown=-0.329, beta=0.733,
            alpha=0.0224, win_rate=0.5, wfe=-0.65, volatility=0.1,
            n_folds=21, n_oos_days=1134, oos_window="2021-01-08 -> 2026-01-14",
        )
        monkeypatch.setattr(
            "ascent.reporting.verified_numbers.canonical_wf", lambda: fake
        )
        prompt = investor_letter._build_user_prompt(
            2026, 6,
            monthly_results=[], itd_results=[],
            attribution={}, events=[], regime="calm_bull", kill_sw={},
            current_weights={},
        )
        assert "0.41" in prompt
        assert "0.52" not in prompt
