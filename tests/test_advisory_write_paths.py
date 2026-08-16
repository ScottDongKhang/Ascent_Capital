"""Every unproven intervention path is advisory-only — no live order, no lost trail.

Three live-write mechanisms were removed in this rebuild because their measured
value-add scored CUT (regime overlay, hedge overlay, the debate judge's position
change, the AI PM's earned-authority blend). The final whole-branch review found
a fourth that was never measured at all and survived untouched: the falsifier
trim (`_apply_falsifier_trim`), which submitted real orders via
`run_eod_with_weights(..., dry_run=False, force=True)` off unmeasured AI PM
output. Same policy, applied consistently: detect, log, record for future
measurement — do not touch live capital.

The same review found that removing the judge's write path silently killed the
measurement trail it was supposed to preserve: `record_intervention()` and
`add_judge_falsifier()` lived inside `apply_judge_position_change`, which is now
never called. Both are re-hoisted into the advisory paths so the ladder keeps
accumulating proposals to score.
"""

import datetime as dt
import json
import os
import sys

import pytest

import run_all_agents as R


TODAY = dt.date(2026, 8, 14)


# ── Fix 1: falsifier trim must not submit orders ──────────────────────────────

@pytest.fixture
def falsifier_env(tmp_path, monkeypatch):
    """Neutralize every gate and side effect around `_apply_falsifier_trim`."""
    import debate.adversarial_authority as auth
    import ascent.strategy.falsifier_registry as reg
    import ascent.execution.eod_runner as eod

    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    calls = {"eod": [], "recorded": [], "marked": []}

    monkeypatch.setattr(R, "_check_mini_rebalance_cooldown", lambda *a, **k: False)
    monkeypatch.setattr(R, "_live_book_or", lambda w: dict(w))
    monkeypatch.setattr(R, "_get_current_regime", lambda *a, **k: "calm_bull")
    monkeypatch.setattr(auth, "get_authority", lambda t: {"suspended": False,
                                                          "allowed_change_pct": 0.01})
    monkeypatch.setattr(auth, "record_intervention",
                        lambda **kw: calls["recorded"].append(kw))
    monkeypatch.setattr(reg, "mark_trimmed", lambda i: calls["marked"].append(i))

    def _boom(*a, **k):
        calls["eod"].append((a, k))
        raise AssertionError("run_eod_with_weights must not be called")

    monkeypatch.setattr(eod, "run_eod_with_weights", _boom)
    return calls


def _fired_one(symbol="BAX"):
    return [{
        "id": f"pm-{symbol}-0",
        "symbol": symbol,
        "source": "prethesis",
        "kind": "price",
        "raw_text": "BAX breaks below 200MA",
        "fired_value": -0.12,
        "trimmed": False,
    }]


class TestFalsifierTrimIsAdvisoryOnly:
    def test_fired_falsifier_submits_no_orders(self, falsifier_env):
        R._apply_falsifier_trim(_fired_one(), {"BAX": 0.08, "AAPL": 0.92},
                                TODAY, dry_run=False)
        assert falsifier_env["eod"] == [], (
            "a fired falsifier must not reach run_eod_with_weights — the trim "
            "is unmeasured (never among the proof audit's 23 components) and "
            "is therefore advisory-only"
        )

    def test_the_would_be_trim_is_still_logged(self, falsifier_env, capsys):
        R._apply_falsifier_trim(_fired_one(), {"BAX": 0.08, "AAPL": 0.92},
                                TODAY, dry_run=False)
        out = capsys.readouterr().out
        assert "BAX" in out and "8.0%" in out and "6.0%" in out, (
            "the trim that WOULD have been made must stay visible for future "
            "measurement"
        )

    def test_eod_log_row_is_marked_advisory(self, falsifier_env, tmp_path):
        R._apply_falsifier_trim(_fired_one(), {"BAX": 0.08, "AAPL": 0.92},
                                TODAY, dry_run=False)
        rows = [json.loads(l) for l in
                (tmp_path / "logs" / "eod_log.jsonl").read_text().splitlines() if l.strip()]
        assert rows, "the advisory trim must still append an eod_log row"
        assert rows[-1]["trigger"] == "falsifier_trim_advisory", (
            "the log row must not claim a trade was executed"
        )
        assert rows[-1]["applied"] is False

    def test_intervention_is_still_recorded_for_future_scoring(self, falsifier_env):
        R._apply_falsifier_trim(_fired_one(), {"BAX": 0.08, "AAPL": 0.92},
                                TODAY, dry_run=False)
        assert len(falsifier_env["recorded"]) == 1
        rec = falsifier_env["recorded"][0]
        assert rec["intervention_type"] == "falsifier_trim"
        assert rec["symbol"] == "BAX"
        assert rec["applied"] is False

    def test_shared_mini_rebalance_cooldown_is_not_started(self, falsifier_env, tmp_path):
        """An advisory event must not suppress the live discovery path."""
        R._apply_falsifier_trim(_fired_one(), {"BAX": 0.08, "AAPL": 0.92},
                                TODAY, dry_run=False)
        assert not (tmp_path / "data_cache" / "last_mini_rebalance.json").exists(), (
            "no live order was submitted, so nothing may start the 5-day "
            "cooldown that gates the discovery mini-rebalance"
        )

    def test_source_has_no_live_eod_call_left(self):
        """AST, not text: the docstring legitimately names the removed call."""
        import ast

        tree = ast.parse(_run_all_agents_src())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "_apply_falsifier_trim")
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        imported = {a.name for n in ast.walk(fn)
                    if isinstance(n, ast.ImportFrom) for a in n.names}
        assert "run_eod_with_weights" not in names | imported, (
            "_apply_falsifier_trim must neither import nor call "
            "run_eod_with_weights — it submits real orders"
        )


def _run_all_agents_src():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "run_all_agents.py")) as f:
        return f.read()


# ── Fix 2: the judge's measurement trail survives the write-path removal ──────

@pytest.fixture
def judge_env(monkeypatch):
    import debate.adversarial_authority as auth
    import ascent.strategy.falsifier_registry as reg

    calls = {"recorded": [], "falsifiers": []}
    monkeypatch.setattr(auth, "record_intervention",
                        lambda **kw: calls["recorded"].append(kw))
    monkeypatch.setattr(reg, "add_judge_falsifier",
                        lambda *a: calls["falsifiers"].append(a))
    return calls


def _verdict(symbol="BAX", new_weight=0.065):
    return {"position_changes": [{
        "symbol": symbol,
        "new_weight": new_weight,
        "current_weight": 0.075,
        "intervention_type": "adversarial_thesis",
        "reason": "worst momentum in book",
        "prediction": f"{symbol} underperforms SPY over the next 10 trading days",
    }]}


class TestJudgeProposalIsStillRecorded:
    def test_helper_exists(self):
        assert hasattr(R, "_record_advisory_judge_proposal")

    def test_proposal_is_recorded_without_being_applied(self, judge_env):
        w = {"BAX": 0.075, "AAPL": 0.5, "MSFT": 0.425}
        before = dict(w)
        R._record_advisory_judge_proposal(w, _verdict(), TODAY,
                                          portfolio_state={"us_regime": "calm_bull"})
        assert w == before, "the advisory recorder must not mutate weights"
        assert len(judge_env["recorded"]) == 1
        rec = judge_env["recorded"][0]
        assert rec["symbol"] == "BAX"
        assert rec["from_weight"] == pytest.approx(0.075)
        assert rec["to_weight"] == pytest.approx(0.065)
        assert rec["regime"] == "calm_bull"
        assert rec["applied"] is False

    def test_judge_falsifier_is_still_registered(self, judge_env):
        R._record_advisory_judge_proposal({"BAX": 0.075, "AAPL": 0.925},
                                          _verdict(), TODAY)
        assert len(judge_env["falsifiers"]) == 1
        assert judge_env["falsifiers"][0][0] == "BAX"

    def test_no_position_changes_records_nothing(self, judge_env):
        R._record_advisory_judge_proposal({"BAX": 0.075}, {}, TODAY)
        assert judge_env["recorded"] == []

    def test_inapplicable_proposal_is_not_recorded(self, judge_env):
        """Same gate the removed write path used: an unheld symbol was never
        recorded historically either, so recording it now would pollute the
        ladder with rows the existing trail does not contain."""
        R._record_advisory_judge_proposal({"AAPL": 0.6, "MSFT": 0.4},
                                          _verdict(symbol="NVDA"), TODAY)
        assert judge_env["recorded"] == []

    def test_only_the_first_change_is_recorded(self, judge_env):
        v = _verdict()
        v["position_changes"].append({"symbol": "AAPL", "new_weight": 0.4,
                                      "current_weight": 0.5})
        R._record_advisory_judge_proposal({"BAX": 0.075, "AAPL": 0.5, "MSFT": 0.425},
                                          v, TODAY)
        assert len(judge_env["recorded"]) == 1
        assert judge_env["recorded"][0]["symbol"] == "BAX"


class TestBothAdvisoryCallSitesRecord:
    def test_scheduled_path_records_the_proposal(self):
        src = _run_all_agents_src()
        i = src.index("def main(")
        j = src.index("def _log_run(")
        body = src[i:j]
        assert "_record_advisory_judge_proposal(" in body, (
            "main()'s debate block must keep feeding the authority ladder even "
            "though nothing is applied"
        )

    def test_discovery_path_records_the_proposal(self):
        src = _run_all_agents_src()
        i = src.index("def _trigger_mini_rebalance")
        tail = src[i:i + 9000]
        assert "_record_advisory_judge_proposal(" in tail

    def test_record_intervention_reaches_a_live_call_site(self):
        """Regression: record_intervention() must be reachable from the
        advisory recorder. apply_judge_position_change (the old write path
        that used to hold the only other call site) was deleted 2026-08-15
        after a repo-wide confirm of zero callers."""
        src = _run_all_agents_src()
        assert "def apply_judge_position_change" not in src, (
            "apply_judge_position_change was deleted as dead code"
        )
        i = src.index("def _record_advisory_judge_proposal")
        j = src.index("def already_ran_for_session")
        body = src[i:j]
        assert "record_intervention(" in body


# ── Fix 3: a single agent's failed Sharpe gate must not halt the pipeline ─────

class TestEarlyZerosCannotHaltTheOnlyAgent:
    def test_all_agents_zeroed_falls_through_to_full_allocation(self, monkeypatch):
        import orchestrator.central_intelligence as ci
        monkeypatch.setattr(ci, "_compute_early_sharpe", lambda aid: -0.5)
        zeroed, adjusted = ci._apply_early_zeros(["us_equities"],
                                                 {"us_equities": 1.0},
                                                 current_regime="calm_bull")
        assert zeroed == [], (
            "with no survivors there is nothing to redistribute to — the gate "
            "must not be applied"
        )
        assert adjusted == {"us_equities": 1.0}
        assert sum(adjusted.values()) == pytest.approx(1.0)

    def test_degenerate_case_warns_loudly(self, monkeypatch, caplog):
        import logging
        import orchestrator.central_intelligence as ci
        monkeypatch.setattr(ci, "_compute_early_sharpe", lambda aid: -0.5)
        with caplog.at_level(logging.WARNING):
            ci._apply_early_zeros(["us_equities"], {"us_equities": 1.0},
                                  current_regime="calm_bull")
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            "the only capital-allocating agent having a non-positive rolling "
            "Sharpe is a real event a human must see"
        )

    def test_multi_agent_redistribution_still_works(self, monkeypatch):
        import orchestrator.central_intelligence as ci
        monkeypatch.setattr(ci, "_compute_early_sharpe",
                            lambda aid: -0.5 if aid == "b" else 0.8)
        zeroed, adjusted = ci._apply_early_zeros(["a", "b"], {"a": 0.5, "b": 0.5},
                                                 current_regime="calm_bull")
        assert zeroed == ["b"]
        assert adjusted["b"] == 0.0
        assert adjusted["a"] == pytest.approx(1.0)
