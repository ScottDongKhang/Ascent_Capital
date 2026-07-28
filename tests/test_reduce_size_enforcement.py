"""`reduce_size` must actually reduce size, and must not sell what the judge protected.

Audit, 2026-07-27. The `reduce_size` channel had four compounding defects:

1. It could not reduce gross exposure. `_apply_verdict_adjustments` told Haiku
   "weights must sum to 1.0" and renormalized; `_enforce_reduce_size` ended by
   renormalizing to exactly 1.0. Measured gross before/after on all three dates
   it ever fired (2026-04-06, 04-15, 07-27): 1.0000 every time. It was a
   rotation channel wearing a de-risking name.

2. The reduction detector ran AFTER renormalization. On 07-27 Haiku correctly
   implemented the judge's VNQ cut of 1.01pp; the renormalize-to-1.0 scaled
   everything back up and shrank that cut to 0.947pp, just under the 1pp
   threshold. The counter read 0 and the forced trim fired.

3. The forced trim sorted by weight and had no idea what the judge said. The
   verdict argued "cutting insurance 48 hours before the catalyst it exists to
   hedge is poor timing" — and the trim sold UUP and TLT, the top two by
   weight, plus BIL (T-bills) which the same verdict called too small.

4. It ignored the authority cap. VNQ moved -2.93pp against a stated 1.0pp limit.
"""

import pytest

from ascent.execution.eod_runner import _enforce_reduce_size


def _book(**kw) -> dict:
    return dict(kw)


class TestGrossExposureActuallyFalls:
    def test_forced_trim_lands_on_the_gross_target(self):
        original = _book(UUP=0.30, TLT=0.30, VNQ=0.20, BIL=0.20)
        haiku = dict(original)  # Haiku changed nothing -> forced trim path
        out = _enforce_reduce_size(original, haiku, target_gross=0.90)
        assert sum(out.values()) == pytest.approx(0.90, abs=1e-6)

    def test_accepted_haiku_reduction_is_never_re_grossed(self):
        """The invariant is gross <= target, not gross == target.

        Haiku here cut to 0.85, below the 0.90 target. Scaling *up* to hit 0.90
        exactly would undo part of its reduction — the same class of bug as the
        old renormalize-to-1.0. It must be left alone.
        """
        original = _book(A=0.25, B=0.25, C=0.25, D=0.25)
        haiku = _book(A=0.20, B=0.20, C=0.20, D=0.25)  # 3 cuts of 5pp, sum 0.85
        out = _enforce_reduce_size(original, haiku, target_gross=0.90)
        assert sum(out.values()) == pytest.approx(0.85, abs=1e-6)
        assert sum(out.values()) <= 0.90 + 1e-9

    def test_default_target_reduces_rather_than_no_opping(self):
        """The default must NOT be 1.0.

        I originally defaulted target_gross to 1.0 for "backwards compatibility",
        which made the function a silent no-op when the argument was omitted:
        needed = total - 1.0 = 0, early return, nothing reduced. That relocates
        the original bug into the signature. A function named
        _enforce_reduce_size must reduce by default; tests/test_plan_b.py caught
        this.
        """
        from ascent.execution.eod_runner import REDUCE_SIZE_GROSS_TARGET
        original = _book(A=0.5, B=0.5)
        out = _enforce_reduce_size(original, dict(original))
        assert sum(out.values()) == pytest.approx(REDUCE_SIZE_GROSS_TARGET, abs=1e-6)
        assert sum(out.values()) < 1.0


class TestDetectionIsPreRenormalization:
    def test_a_genuine_cut_is_not_erased_by_renormalization(self):
        """The 07-27 bug: a 1.01pp cut measured after renorm reads as 0.947pp.

        23 roughly-equal positions, one cut by just over the 1pp threshold. The
        detector must see the cut on a comparable gross basis and accept it
        rather than force a size-sorted trim.
        """
        original = {f"S{i}": 1.0 / 23 for i in range(23)}
        haiku = dict(original)
        for s in ("S0", "S1", "S2"):
            haiku[s] -= 0.0110  # just over the 1pp threshold, as on 07-27
        out = _enforce_reduce_size(original, haiku, target_gross=1.0)
        # Accepted, not force-trimmed: the three cut names stay cut, and the
        # untouched largest names are not trimmed.
        assert out["S0"] < original["S0"]
        assert out["S3"] == pytest.approx(original["S3"], abs=1e-6)


class TestProtectedPositions:
    def test_forced_trim_never_touches_a_protected_symbol(self):
        original = _book(UUP=0.30, TLT=0.28, VNQ=0.22, IFRA=0.12, SCHH=0.08)
        out = _enforce_reduce_size(
            original, dict(original), target_gross=0.90,
            protected=frozenset({"UUP", "TLT"}),
        )
        assert out["UUP"] == pytest.approx(original["UUP"], abs=1e-6)
        assert out["TLT"] == pytest.approx(original["TLT"], abs=1e-6)

    def test_the_2026_07_27_scenario(self):
        """Real weights from verdict_2026-07-27.json. The judge protected the
        UUP/TLT hedge leg ahead of FOMC; the trim sold both."""
        original = {
            "UUP": 0.0870, "TLT": 0.0825, "VNQ": 0.0734, "IFRA": 0.0712,
            "BIL": 0.0543, "SCHH": 0.0460, "DBB": 0.0470, "PDBC": 0.0447,
            "CXT": 0.0444, "DLR": 0.0432, "BRBR": 0.0431, "AES": 0.0418,
            "SGOV": 0.0412, "SLM": 0.0384, "KNF": 0.0382, "DKS": 0.0373,
            "MRNA": 0.0325, "ALGM": 0.0293, "VIXY": 0.0257, "EWJ": 0.0255,
            "EWT": 0.0241, "EEM": 0.0229, "EFA": 0.0225,
        }
        out = _enforce_reduce_size(
            original, dict(original), target_gross=0.90,
            protected=frozenset({"UUP", "TLT"}),
        )
        assert out["UUP"] == pytest.approx(original["UUP"], abs=1e-6)
        assert out["TLT"] == pytest.approx(original["TLT"], abs=1e-6)
        assert sum(out.values()) == pytest.approx(0.90, abs=1e-6)

    def test_protecting_everything_still_reaches_the_gross_target(self):
        """Degenerate case: nothing is trimmable by name, so the book must be
        scaled down uniformly rather than left at full gross."""
        original = _book(A=0.5, B=0.5)
        out = _enforce_reduce_size(
            original, dict(original), target_gross=0.80,
            protected=frozenset({"A", "B"}),
        )
        assert sum(out.values()) == pytest.approx(0.80, abs=1e-6)


class TestAuthorityCapIsNotThisFunctionsJob:
    """Defect 4 (VNQ moved -2.93pp against a 1.0pp cap) is deliberately not
    fixed by capping per-name change here.

    A per-name cap and an honoured protect-list cannot both hold: keeping UUP at
    its full weight while gross falls to 0.90 IS a deviation from UUP's pro-rata
    share of the reduced book, so a 1pp cap would forbid honouring the judge.
    The audit hit the same contradiction from the other side — requiring three
    cuts of >=1pp while capping each intervention at 1.0pp.

    De-grossing is a portfolio-level risk action, not a per-name bet. The
    per-intervention cap governs the judge's `position_changes`, applied and
    capped in run_all_agents. What needs bounding here is total gross.
    """

    def test_gross_target_is_bounded(self):
        from ascent.execution.eod_runner import (
            REDUCE_SIZE_GROSS_TARGET, REDUCE_SIZE_MAX_GROSS, REDUCE_SIZE_MIN_GROSS,
        )
        assert REDUCE_SIZE_MIN_GROSS <= REDUCE_SIZE_GROSS_TARGET <= REDUCE_SIZE_MAX_GROSS
        assert REDUCE_SIZE_MIN_GROSS > 0.5, "a reduce_size must not near-liquidate"
        assert REDUCE_SIZE_MAX_GROSS < 1.0, "a reduce_size must actually reduce"

    def test_reduction_is_spread_not_concentrated_on_the_largest(self):
        """The old trim took 2pp off the top 5 regardless of thesis. A
        proportional cut across the non-protected sleeve keeps relative sizing."""
        original = _book(A=0.40, B=0.30, C=0.20, D=0.10)
        out = _enforce_reduce_size(original, dict(original), target_gross=0.90)
        # every name shrinks, and their ratios are preserved
        assert all(out[s] < original[s] for s in original)
        assert out["A"] / out["B"] == pytest.approx(original["A"] / original["B"], rel=1e-6)


class TestExistingGuardsPreserved:
    def test_empty_haiku_weights_returns_original(self):
        original = _book(A=0.6, B=0.4)
        assert _enforce_reduce_size(original, {}) == original

    def test_no_negative_weights(self):
        original = _book(A=0.01, B=0.99)
        out = _enforce_reduce_size(original, dict(original), target_gross=0.90)
        assert all(w >= 0.0 for w in out.values())


class TestVerdictProtectedSymbols:
    """The judge's protection existed only as English prose inside
    verdict["reasoning"], which no downstream code could read. The verdict
    contract now carries a machine-readable list."""

    def test_reads_the_protected_positions_field(self):
        from ascent.execution.eod_runner import _verdict_protected_symbols
        v = {"protected_positions": [
            {"symbol": "UUP", "reason": "FOMC hedge leg"},
            {"symbol": "TLT", "reason": "FOMC hedge leg"},
        ]}
        assert _verdict_protected_symbols(v) == frozenset({"UUP", "TLT"})

    def test_accepts_a_plain_list_of_strings(self):
        from ascent.execution.eod_runner import _verdict_protected_symbols
        assert _verdict_protected_symbols(
            {"protected_positions": ["UUP", "TLT"]}
        ) == frozenset({"UUP", "TLT"})

    def test_absent_or_malformed_yields_empty_not_error(self):
        from ascent.execution.eod_runner import _verdict_protected_symbols
        for v in ({}, {"protected_positions": None}, {"protected_positions": "UUP"},
                  {"protected_positions": [{"no_symbol": 1}, 42]}, None):
            assert _verdict_protected_symbols(v) == frozenset()

    def test_symbols_are_upper_cased_and_stripped(self):
        from ascent.execution.eod_runner import _verdict_protected_symbols
        assert _verdict_protected_symbols(
            {"protected_positions": [{"symbol": " uup "}]}
        ) == frozenset({"UUP"})


class TestReduceSizeTargetGross:
    def test_defaults_when_verdict_says_nothing(self):
        from ascent.execution.eod_runner import (
            REDUCE_SIZE_GROSS_TARGET, _reduce_size_target_gross)
        assert _reduce_size_target_gross({}) == REDUCE_SIZE_GROSS_TARGET

    def test_honours_a_verdict_supplied_reduction(self):
        from ascent.execution.eod_runner import _reduce_size_target_gross
        # The 2026-04-06 verdict asked to "reduce overall position size by ~30%".
        assert _reduce_size_target_gross({"reduction_pct": 0.20}) == pytest.approx(0.80)

    def test_clamps_an_absurd_reduction(self):
        from ascent.execution.eod_runner import (
            REDUCE_SIZE_MAX_GROSS, REDUCE_SIZE_MIN_GROSS, _reduce_size_target_gross)
        assert _reduce_size_target_gross({"reduction_pct": 0.95}) == REDUCE_SIZE_MIN_GROSS
        assert _reduce_size_target_gross({"reduction_pct": 0.001}) == REDUCE_SIZE_MAX_GROSS

    def test_ignores_garbage(self):
        from ascent.execution.eod_runner import (
            REDUCE_SIZE_GROSS_TARGET, _reduce_size_target_gross)
        for bad in ({"reduction_pct": "lots"}, {"reduction_pct": None},
                    {"reduction_pct": -0.1}, None):
            assert _reduce_size_target_gross(bad) == REDUCE_SIZE_GROSS_TARGET
