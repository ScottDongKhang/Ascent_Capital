"""
tests/debate/test_outcome_tracker_w4.py

W4 (adversarial layer measurement integrity) coverage for
debate/outcome_tracker.py:

  - min-sample guard: n below MIN_SAMPLES_FOR_TRACK_RECORD must produce an
    explicit insufficient-data string, never a bare accuracy percentage.
  - regime-key normalization: "RegimeLabel.STRESSED" and "stressed" must
    land in the same credibility bucket, on both read and write.

All tests use tmp_path + monkeypatch — never touch the real
outputs/debate_log/ directory.
"""
import json

import debate.outcome_tracker as ot


# ── load_recent_verdict_outcomes: min-sample guard ──────────────────────────

def _write_verdict(dir_path, name, *, outcome_scored, recommendation="proceed",
                    outcome_score=1.0, nav_change=0.01, regime="calm_bull",
                    vdate="2026-04-15"):
    rec = {
        "date": vdate,
        "verdict": {"recommendation": recommendation, "confidence": 0.7},
        "portfolio_state": {"us_regime": regime},
        "outcome_scored": outcome_scored,
    }
    if outcome_scored:
        rec["outcome_score"] = outcome_score
        rec["outcome_nav_change"] = nav_change
        rec["outcome_window_end"] = "2026-04-29"
    (dir_path / name).write_text(json.dumps(rec))


def test_recent_verdict_outcomes_n1_is_insufficient(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "DEBATE_LOG_DIR", tmp_path)
    _write_verdict(tmp_path, "verdict_2026-04-15.json", outcome_scored=True)

    text = ot.load_recent_verdict_outcomes(n=5)

    assert "n=1" in text
    assert "insufficient" in text.lower()
    # Must NOT present this as a track record with a CORRECT/WRONG label.
    assert "CORRECT" not in text
    assert "WRONG" not in text


def test_recent_verdict_outcomes_shows_detail_once_over_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "DEBATE_LOG_DIR", tmp_path)
    for i in range(10):
        _write_verdict(tmp_path, f"verdict_2026-05-{i+1:02d}.json",
                        outcome_scored=True, vdate=f"2026-05-{i+1:02d}")

    text = ot.load_recent_verdict_outcomes(n=5)

    assert "insufficient" not in text.lower()
    assert "JUDGE'S OWN TRACK RECORD" in text
    assert "CORRECT" in text


def test_recent_verdict_outcomes_empty_dir_is_insufficient_not_blank(tmp_path, monkeypatch):
    """
    An empty debate_log dir (n=0 scored) must still surface the explicit
    insufficient-data note, not silently return "" — silence here previously
    meant the judge simply got no track-record context at all, indistinguishable
    from "the mechanism isn't wired up."
    """
    monkeypatch.setattr(ot, "DEBATE_LOG_DIR", tmp_path)
    text = ot.load_recent_verdict_outcomes(n=5)
    assert "n=0" in text
    assert "insufficient" in text.lower()


def test_recent_verdict_outcomes_missing_dir_returns_blank(tmp_path, monkeypatch):
    monkeypatch.setattr(ot, "DEBATE_LOG_DIR", tmp_path / "does_not_exist")
    assert ot.load_recent_verdict_outcomes(n=5) == ""


# ── load_credibility_context: min-sample guard ──────────────────────────────

def _write_credibility(path, *, by_regime=None, sample_counts=None,
                        overall=None, overall_sample_counts=None):
    cred = {
        "by_regime": by_regime or {},
        "sample_counts": sample_counts or {},
        "overall": overall or {},
        "overall_sample_counts": overall_sample_counts or {},
    }
    path.write_text(json.dumps(cred))


def test_credibility_context_below_min_samples_is_insufficient(tmp_path, monkeypatch):
    cred_path = tmp_path / "agent_credibility.json"
    monkeypatch.setattr(ot, "CREDIBILITY_PATH", cred_path)
    _write_credibility(
        cred_path,
        by_regime={"stressed": {"bull": 0.1, "bear": 0.0, "devils_advocate": 0.0}},
        sample_counts={"stressed": {"bull": 1, "bear": 1, "devils_advocate": 1}},
        overall={"bull": 0.1, "bear": 0.0, "devils_advocate": 0.0},
        overall_sample_counts={"bull": 1, "bear": 1, "devils_advocate": 1},
    )

    text = ot.load_credibility_context("stressed")

    assert "insufficient" in text.lower()
    # A single scored debate must never be shown as a bare percentage.
    assert "0%" not in text
    assert "10%" not in text


def test_credibility_context_above_min_samples_shows_percentage(tmp_path, monkeypatch):
    cred_path = tmp_path / "agent_credibility.json"
    monkeypatch.setattr(ot, "CREDIBILITY_PATH", cred_path)
    _write_credibility(
        cred_path,
        by_regime={"stressed": {"bull": 0.7, "bear": 0.8, "devils_advocate": 0.6}},
        sample_counts={"stressed": {"bull": 12, "bear": 12, "devils_advocate": 12}},
        overall={"bull": 0.7, "bear": 0.8, "devils_advocate": 0.6},
        overall_sample_counts={"bull": 12, "bear": 12, "devils_advocate": 12},
    )

    text = ot.load_credibility_context("stressed")

    assert "insufficient" not in text.lower()
    assert "70%" in text or "80%" in text


# ── Regime-key normalization ─────────────────────────────────────────────────

def test_normalize_regime_key_strips_prefix_and_lowercases():
    assert ot._normalize_regime_key("RegimeLabel.STRESSED") == "stressed"
    assert ot._normalize_regime_key("stressed") == "stressed"
    assert ot._normalize_regime_key("  Calm_Bull ") == "calm_bull"


def test_credibility_context_reads_regimelabel_key_via_normalization(tmp_path, monkeypatch):
    """
    A record whose stored key is 'RegimeLabel.STRESSED' must still be found
    when the judge asks for regime='RegimeLabel.STRESSED' (the raw str() of
    the enum) or 'stressed' — same bucket, not two.
    """
    cred_path = tmp_path / "agent_credibility.json"
    monkeypatch.setattr(ot, "CREDIBILITY_PATH", cred_path)
    _write_credibility(
        cred_path,
        by_regime={"stressed": {"bull": 0.7, "bear": 0.8, "devils_advocate": 0.6}},
        sample_counts={"stressed": {"bull": 12, "bear": 12, "devils_advocate": 12}},
    )

    text_raw_enum = ot.load_credibility_context("RegimeLabel.STRESSED")
    text_plain     = ot.load_credibility_context("stressed")

    assert "In stressed regime" in text_raw_enum
    assert "In stressed regime" in text_plain
    assert text_raw_enum == text_plain


def test_rebuild_credibility_merges_regimelabel_and_plain_keys(tmp_path, monkeypatch):
    """
    The write-side bug: two verdict files, one stored 'stressed', the other
    'RegimeLabel.STRESSED' — must merge into ONE bucket with n=2, not split
    into two buckets of n=1 each (the exact 2026-04-15 failure mode).
    """
    debate_dir = tmp_path / "debate_log"
    debate_dir.mkdir()
    cred_path = tmp_path / "agent_credibility.json"
    monkeypatch.setattr(ot, "DEBATE_LOG_DIR", debate_dir)
    monkeypatch.setattr(ot, "CREDIBILITY_PATH", cred_path)

    rec1 = {
        "date": "2026-04-15",
        "portfolio_state": {"us_regime": "RegimeLabel.STRESSED"},
        "outcome_scored": True,
        "agent_scores": {"bull": 0.1, "bear": 0.0, "devils_advocate": 0.0},
    }
    rec2 = {
        "date": "2026-04-20",
        "portfolio_state": {"us_regime": "stressed"},
        "outcome_scored": True,
        "agent_scores": {"bull": 0.9, "bear": 1.0, "devils_advocate": 1.0},
    }
    (debate_dir / "verdict_2026-04-15.json").write_text(json.dumps(rec1))
    (debate_dir / "verdict_2026-04-20.json").write_text(json.dumps(rec2))

    ot._rebuild_credibility()

    cred = json.loads(cred_path.read_text())
    assert "stressed" in cred["by_regime"]
    assert "regimelabel.stressed" not in cred["by_regime"]
    assert cred["sample_counts"]["stressed"]["bull"] == 2
