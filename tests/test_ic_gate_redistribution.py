"""tests/test_ic_gate_redistribution.py

_get_gated_weights() previously redistributed freed weight (from a sleeve zeroed
by the IC gate) ONLY to a hardcoded "trend" key. With the 2-sleeve meanrev/statarb
stack, "trend" isn't a key in DEFAULT_ALPHA_WEIGHTS at all, so that branch never
fired and freed weight stayed at 0 in the dict.

CORRECTED NARRATIVE (found during final review): this does NOT produce an
under-invested portfolio in the common case. build_alpha_stack()'s blend step
renormalizes by `alpha_weights.get(name) / sum(alpha_weights.get(k) for k in
alphas)` -- a constant scale factor on every surviving weight is exactly
cancelled by that renormalization, so the pre-fix and post-fix composite are
bit-for-bit identical whenever at least one survivor exists. The ORIGINAL
commit message and this file's original docstring overstated the live effect;
this comment corrects the record for anyone diagnosing a future re-validation
run using git blame.

The one case this fix genuinely changes: if EVERY sleeve is gated
simultaneously, the pre-fix code produced an all-zero dict, which renormalizes
to `total_w == 0 -> total_w = 1.0 -> every weight = 0` -- a silent no-signal
composite. Post-fix, the original (ungated) weights are returned instead, so
the portfolio keeps the pre-gate signal rather than going dark. That fail-safe
is the real, load-bearing effect of this commit.

The proportional-redistribution semantics (vs. the old hardcoded "trend" name)
are still worth keeping even though renormalization mostly cancels them out --
they make the returned dict self-consistent for any future consumer that does
NOT renormalize, and they remove a hardcoded dependency on a sleeve name that
no longer exists.
"""
import json

import pytest


def _write_ic_log(path, gated_sleeves, window=5, healthy_ic=0.02, bad_ic=-0.015):
    """Follow the fixture pattern from tests/test_fundamental_alpha.py:
    `window` unique-date entries, each carrying mean_ic per sleeve."""
    lines = []
    for i in range(window):
        entry = {
            "date": f"2026-05-{20 + i:02d}",
            "sleeves": {
                sleeve: {"mean_ic": bad_ic if sleeve in gated_sleeves else healthy_ic, "t_stat": -3.0, "n": 900}
                for sleeve in gated_sleeves
            },
        }
        lines.append(json.dumps(entry))
    path.write_text("\n".join(lines) + "\n")


def test_two_sleeve_gate_freed_weight_goes_entirely_to_survivor(tmp_path):
    """2-sleeve dict, one gated: freed weight must go entirely to the lone
    survivor, not be dropped, and must NOT create a "trend" key."""
    from ascent.alpha.stack import _get_gated_weights

    ic_log = tmp_path / "sleeve_ic_log.jsonl"
    _write_ic_log(ic_log, gated_sleeves=["meanrev"])

    base = {"meanrev": 0.5, "statarb": 0.5}
    result = _get_gated_weights(base, ic_log_path=str(ic_log))

    assert result["meanrev"] == 0.0
    assert result["statarb"] == pytest.approx(1.0)
    assert "trend" not in result


def test_three_sleeve_gate_freed_weight_redistributes_proportionally(tmp_path):
    """3+ sleeve dict, one gated: freed weight splits among survivors in
    proportion to their existing weight share (not evenly)."""
    from ascent.alpha.stack import _get_gated_weights

    ic_log = tmp_path / "sleeve_ic_log.jsonl"
    _write_ic_log(ic_log, gated_sleeves=["fundamental"])

    base = {"fundamental": 0.2, "meanrev": 0.3, "statarb": 0.5}
    result = _get_gated_weights(base, ic_log_path=str(ic_log))

    freed = 0.2
    # survivors' shares of the surviving total (0.3 + 0.5 = 0.8)
    expected_meanrev = 0.3 + freed * (0.3 / 0.8)
    expected_statarb = 0.5 + freed * (0.5 / 0.8)

    assert result["fundamental"] == 0.0
    assert result["meanrev"] == pytest.approx(expected_meanrev, abs=1e-4)
    assert result["statarb"] == pytest.approx(expected_statarb, abs=1e-4)
    assert sum(result.values()) == pytest.approx(sum(base.values()), abs=1e-3)


def test_all_sleeves_gated_returns_original_weights_unchanged(tmp_path):
    """If every sleeve is gated simultaneously (no survivors), the function must
    fail safe and return the ORIGINAL alpha_weights unchanged -- not an
    all-zero dict, which would leave the portfolio in cash silently."""
    from ascent.alpha.stack import _get_gated_weights

    ic_log = tmp_path / "sleeve_ic_log.jsonl"
    _write_ic_log(ic_log, gated_sleeves=["meanrev", "statarb"])

    base = {"meanrev": 0.5, "statarb": 0.5}
    result = _get_gated_weights(base, ic_log_path=str(ic_log))

    assert result == base
    assert result["meanrev"] == 0.5
    assert result["statarb"] == 0.5


def test_legacy_shaped_dict_trend_gets_no_special_treatment(tmp_path):
    """A legacy-shaped dict where 'trend' is present alongside other sleeves:
    trend must get proportional redistribution like everything else, no
    special-cased boost beyond its proportional share."""
    from ascent.alpha.stack import _get_gated_weights

    ic_log = tmp_path / "sleeve_ic_log.jsonl"
    _write_ic_log(ic_log, gated_sleeves=["fundamental"])

    base = {"fundamental": 0.05, "trend": 0.38, "meanrev": 0.05, "statarb": 0.52}
    result = _get_gated_weights(base, ic_log_path=str(ic_log))

    freed = 0.05
    survivors_total = 0.38 + 0.05 + 0.52  # 0.95
    expected_trend = 0.38 + freed * (0.38 / survivors_total)
    expected_meanrev = 0.05 + freed * (0.05 / survivors_total)
    expected_statarb = 0.52 + freed * (0.52 / survivors_total)

    assert result["fundamental"] == 0.0
    assert result["trend"] == pytest.approx(expected_trend, abs=1e-4)
    assert result["meanrev"] == pytest.approx(expected_meanrev, abs=1e-4)
    assert result["statarb"] == pytest.approx(expected_statarb, abs=1e-4)
