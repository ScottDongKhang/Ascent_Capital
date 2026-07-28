"""AI PM overrides must be derived from the weights, not self-reported.

Audit, 2026-07-27. `overrides_applied` in logs/ai_pm_decision_log.jsonl was
populated from `thesis["quant_overrides"]` — a free-form field the model had to
volunteer. It volunteered none: all 9 rows across 2 dates have `[]`.

That single fact starves the entire measurement apparatus. `n_decisions_evaluated`
is len(_score_decisions(...)), which iterates `overrides_applied`, so it is
permanently 0 — and `min_decisions: n >= 5` is a hard promotion gate in
earned_authority.PROMOTION_CONFIG. Demotion, by contrast, needs only a single
bad day. The ladder is a one-way ratchet: days_stuck was 19 at Level 1.

Both weight vectors were already being logged side by side, so the override set
is recoverable by subtraction and never needed the model's cooperation.

Second defect: even a volunteered override would not have scored. The tool emits
{symbol, ai_action, reason, override_type} but the scorer reads ov["ai_w"],
ov["quant_w"], ov["type"] — so every override scored 0.0 incremental alpha,
forcing profit_factor to exactly 1.0, which fails the > 1.2 promotion gate
forever. The derived records use the field names the scorer actually reads.
"""

import pytest

from ascent.strategy.ai_pm_perf_feedback import derive_overrides


class TestDerivation:
    def test_amplify_is_detected(self):
        out = derive_overrides({"AAPL": 0.05}, {"AAPL": 0.09})
        assert len(out) == 1
        assert out[0]["symbol"] == "AAPL"
        assert out[0]["type"] == "amplify"
        assert out[0]["quant_w"] == pytest.approx(0.05)
        assert out[0]["ai_w"] == pytest.approx(0.09)

    def test_reduce_is_detected(self):
        out = derive_overrides({"BAX": 0.075}, {"BAX": 0.05})
        assert out[0]["type"] == "reduce"

    def test_new_position_is_detected(self):
        out = derive_overrides({"AAPL": 1.0}, {"AAPL": 0.9, "EEM": 0.10})
        eem = [o for o in out if o["symbol"] == "EEM"]
        assert eem and eem[0]["type"] == "new"
        assert eem[0]["quant_w"] == 0.0

    def test_exit_is_detected(self):
        out = derive_overrides({"AAPL": 0.9, "BAX": 0.10}, {"AAPL": 1.0})
        bax = [o for o in out if o["symbol"] == "BAX"]
        assert bax and bax[0]["type"] == "exit"
        assert bax[0]["ai_w"] == 0.0

    def test_uses_the_field_names_the_scorer_reads(self):
        """_score_decisions reads ai_w / quant_w / type — not ai_action /
        override_type, which is what the tool schema emitted."""
        out = derive_overrides({"AAPL": 0.05}, {"AAPL": 0.09})
        assert {"symbol", "type", "ai_w", "quant_w"} <= set(out[0])


class TestThresholding:
    def test_noise_below_threshold_is_ignored(self):
        """Rounding differences are not decisions."""
        assert derive_overrides({"AAPL": 0.0500}, {"AAPL": 0.0503}) == []

    def test_threshold_is_configurable(self):
        assert derive_overrides({"A": 0.05}, {"A": 0.056}, min_delta=0.005) != []
        assert derive_overrides({"A": 0.05}, {"A": 0.056}, min_delta=0.02) == []

    def test_results_are_ordered_by_conviction(self):
        out = derive_overrides(
            {"A": 0.05, "B": 0.05, "C": 0.05},
            {"A": 0.06, "B": 0.11, "C": 0.02},
        )
        deltas = [abs(o["ai_w"] - o["quant_w"]) for o in out]
        assert deltas == sorted(deltas, reverse=True)


class TestDegenerateInputs:
    def test_fallback_empty_ai_book_yields_no_overrides(self):
        """A force-sealed run expressed no original judgment; it must not be
        recorded as having overridden the whole book."""
        assert derive_overrides({"AAPL": 0.5, "MSFT": 0.5}, {}) == []

    def test_identical_books_yield_nothing(self):
        w = {"AAPL": 0.5, "MSFT": 0.5}
        assert derive_overrides(w, dict(w)) == []

    def test_none_inputs_are_safe(self):
        assert derive_overrides(None, None) == []
        assert derive_overrides({"A": 1.0}, None) == []


class TestWiredIntoDecisionLog:
    def test_run_all_agents_derives_instead_of_self_reporting(self):
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "run_all_agents.py")) as f:
            src = f.read()
        i = src.index("def _write_decision_log")
        body = src[i:i + 3000]
        assert "derive_overrides(" in body, (
            "_write_decision_log must derive overrides from the weight vectors"
        )


class TestAuthorityLadderIsReachable:
    """The promotion metrics must actually reach update_authority().

    On rebalance days a bare `update_authority(_ai_ret, _qt_ret)` ran first with
    n_decisions_evaluated defaulted to 0, stamping last_updated=today. The
    informed call later in the same run then hit the
    `if state["last_updated"] == today: return state` short-circuit in
    earned_authority.update_authority, so the promotion gates were never
    evaluated on precisely the days that produce decisions.
    """

    def _call_lines(self):
        """Lines that actually invoke update_authority — not comments, not the
        import. Counting raw occurrences would match prose in the comment that
        explains why the duplicate call was removed."""
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "run_all_agents.py")) as f:
            src = f.read()
        out = []
        for lineno, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("from ") or stripped.startswith("import "):
                continue
            if "update_authority(" in stripped:
                out.append((lineno, stripped))
        return src, out

    def test_only_one_update_authority_call_site(self):
        _, calls = self._call_lines()
        assert len(calls) == 1, (
            f"expected exactly one update_authority() call so it cannot be "
            f"pre-empted by a same-day short-circuit, found {len(calls)}: {calls}"
        )

    def test_the_surviving_call_passes_the_promotion_metrics(self):
        src, calls = self._call_lines()
        lineno = calls[0][0]
        # The call spans several lines; inspect the following few.
        window = "\n".join(src.splitlines()[lineno - 1:lineno + 8])
        for field in ("n_decisions_evaluated", "hit_rate", "profit_factor", "fade_rate"):
            assert field in window, f"{field} not passed to update_authority"
