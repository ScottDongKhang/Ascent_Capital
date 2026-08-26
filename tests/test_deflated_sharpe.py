# tests/test_deflated_sharpe.py
"""
Tests for ascent/research/deflated_sharpe.py (Bailey & Lopez de Prado 2014,
SSRN 2460551).

No internet access in this environment, so these are NOT reproductions of a
published table from the paper. Instead each test is a hand/independently
computed numeric example: round-number inputs are fed through the same
closed-form equations the module docstring cites (E[max SR] extreme-value
approximation, then the Mertens/PSR formula), computed independently in a
scratch script with `scipy.stats.norm` directly (not by importing the
module under test), and the expected values are pinned here. See the
computation transcript this file's numbers were derived from in the PR/
session notes if reproducing independently.
"""
import math

import pytest
from scipy.stats import norm

from ascent.research.deflated_sharpe import (
    KNOWN_TRIAL_COUNT,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)

_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# expected_max_sharpe()
# ---------------------------------------------------------------------------

def test_expected_max_sharpe_hand_computed():
    """n_trials=10, V=0.01 (sigma=0.1): independently computed via the same
    extreme-value formula using scipy.stats.norm directly in this test."""
    n, V = 10, 0.01
    sigma = math.sqrt(V)
    expected = sigma * (
        (1 - _GAMMA) * norm.ppf(1 - 1.0 / n)
        + _GAMMA * norm.ppf(1 - 1.0 / (n * math.e))
    )
    assert expected == pytest.approx(0.157459830134575, abs=1e-9)
    assert expected_max_sharpe(n, V) == pytest.approx(expected, abs=1e-12)


def test_expected_max_sharpe_n_trials_1_is_zero():
    """No selection effect with a single trial -- degenerate Phi^-1(0) term
    is explicitly special-cased to 0.0 rather than -inf."""
    assert expected_max_sharpe(1, 0.05) == 0.0


def test_expected_max_sharpe_increases_with_n_trials():
    """More trials -> a higher bar of pure-luck Sharpe to beat."""
    small = expected_max_sharpe(5, 0.01)
    large = expected_max_sharpe(500, 0.01)
    assert large > small > 0


def test_expected_max_sharpe_rejects_negative_variance():
    with pytest.raises(ValueError):
        expected_max_sharpe(10, -0.01)


# ---------------------------------------------------------------------------
# probabilistic_sharpe_ratio()
# ---------------------------------------------------------------------------

def test_psr_hand_computed_normal_returns():
    """SR=1.0, benchmark=0, skew=0, RAW kurtosis=3 (normal), n_obs=101.
    denom_inside = 1 - 0*1 + (3-1)/4*1^2 = 1.5
    z = 1.0 * sqrt(100) / sqrt(1.5) = 8.164...
    PSR = Phi(z) ~= 1.0 (essentially certain SR > 0 with this much data)."""
    psr = probabilistic_sharpe_ratio(
        sharpe_observed=1.0, sharpe_benchmark=0.0, skew=0.0, kurtosis=3.0, n_obs=101
    )
    assert psr == pytest.approx(0.9999999999999999, abs=1e-12)


def test_psr_zero_when_observed_equals_benchmark():
    """SR exactly at the benchmark -> PSR = Phi(0) = 0.5 exactly."""
    psr = probabilistic_sharpe_ratio(
        sharpe_observed=0.3, sharpe_benchmark=0.3, skew=0.0, kurtosis=3.0, n_obs=50
    )
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_degenerate_n_obs_returns_uninformative_half():
    assert probabilistic_sharpe_ratio(1.0, 0.0, 0.0, 3.0, n_obs=1) == 0.5
    assert probabilistic_sharpe_ratio(1.0, 0.0, 0.0, 3.0, n_obs=0) == 0.5


def test_psr_negative_denominator_returns_none_not_half():
    """Extreme skew/kurtosis/SR combination can drive the Mertens
    denominator non-positive. This is a FORMULA BREAKDOWN, not the same
    thing as the genuinely uninformative n_obs<=1 case -- it must NOT be
    silently reported as 0.5 (which would misreport a possibly-strong
    result as neutral). The fix returns None instead, verified here by
    first confirming the denominator really is negative for these inputs:
        denom_inside = 1 - skew*SR + ((kurtosis-1)/4)*SR**2
                     = 1 - 5*5   + ((3-1)/4)*25
                     = 1 - 25 + 12.5 = -11.5  (< 0)
    """
    skew, kurtosis, sharpe = 5.0, 3.0, 5.0
    denom_inside = 1 - skew * sharpe + ((kurtosis - 1) / 4.0) * sharpe ** 2
    assert denom_inside < 0  # sanity-check this really is the degenerate case

    psr = probabilistic_sharpe_ratio(
        sharpe_observed=sharpe, sharpe_benchmark=0.0, skew=skew, kurtosis=kurtosis, n_obs=30
    )
    assert psr is None


def test_psr_plausible_positive_skill_denominator_negative_returns_none():
    """Reviewer's own example: skew=+2, SR=3 is not a contrived edge case
    -- it's a plausible real skew/Sharpe combination for a strategy with
    genuinely strong measurable skill, and it still drives the Mertens
    denominator negative. Must return None, never a silently-neutral 0.5.
        denom_inside = 1 - 2*3 + ((3-1)/4)*3**2 = 1 - 6 + 4.5 = -0.5 (< 0)
    """
    skew, kurtosis, sharpe = 2.0, 3.0, 3.0
    denom_inside = 1 - skew * sharpe + ((kurtosis - 1) / 4.0) * sharpe ** 2
    assert denom_inside < 0

    psr = probabilistic_sharpe_ratio(
        sharpe_observed=sharpe, sharpe_benchmark=0.0, skew=skew, kurtosis=kurtosis, n_obs=252
    )
    assert psr is None


def test_psr_n_obs_le_1_still_returns_real_half_not_none():
    """Regression safety: the genuinely uninformative n_obs<=1 case is
    UNCHANGED by this fix -- it is a real, defensible sentinel (there is
    no test statistic to compute at all), distinct from the denominator-
    breakdown case above, and must keep returning exactly 0.5, not None."""
    assert probabilistic_sharpe_ratio(1.0, 0.0, 0.0, 3.0, n_obs=1) == 0.5
    assert probabilistic_sharpe_ratio(1.0, 0.0, 0.0, 3.0, n_obs=0) == 0.5
    # Also confirm a case that would have both n_obs<=1 AND an extreme
    # skew/kurtosis/SR combination still resolves via the n_obs<=1 branch
    # (checked first) and returns 0.5, not None.
    assert probabilistic_sharpe_ratio(5.0, 0.0, 5.0, 3.0, n_obs=1) == 0.5


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio()
# ---------------------------------------------------------------------------

def test_dsr_hand_computed_full_pipeline():
    """SR=1.0, n_trials=10, pandas-style skew=0/kurtosis=0 (i.e. normal:
    raw kurtosis 3), n_obs=101, using the documented default SR-variance
    fallback. Independently re-derived in a scratch computation:
      raw_kurtosis = 0 + 3 = 3
      V_default = (1 - 0*1 + (3-1)/4*1^2) / (101-1) = 1.5/100 = 0.015
      bench = expected_max_sharpe(10, 0.015) ~= 0.19284811940751...
      denom = sqrt(1.5) ~= 1.224745
      z = (1.0 - bench) * sqrt(100) / denom ~= 6.59058...
      DSR = Phi(z) ~= 0.999999999978063
    """
    dsr = deflated_sharpe_ratio(
        sharpe_observed=1.0, n_trials=10, skew=0.0, kurtosis=0.0, n_obs=101
    )
    assert dsr == pytest.approx(0.999999999978063, abs=1e-9)


def test_dsr_n_trials_1_reduces_to_plain_psr():
    """n_trials=1 -> no deflation benchmark (0.0) -> DSR should equal a
    plain PSR(sharpe_observed, benchmark=0, ...) call exactly, since
    deflated_sharpe_ratio's internal expected_max_sharpe(1, ...) is 0.0
    regardless of the SR-variance fallback."""
    sharpe, skew, kurt_excess, n_obs = 0.8, 0.1, 1.0, 300

    dsr = deflated_sharpe_ratio(
        sharpe_observed=sharpe, n_trials=1, skew=skew, kurtosis=kurt_excess, n_obs=n_obs
    )
    psr = probabilistic_sharpe_ratio(
        sharpe_observed=sharpe,
        sharpe_benchmark=0.0,
        skew=skew,
        kurtosis=kurt_excess + 3.0,  # convert pandas-excess -> raw, as DSR does internally
        n_obs=n_obs,
    )
    assert dsr == pytest.approx(psr, abs=1e-12)


def test_dsr_normal_returns_skew_zero_kurtosis_zero_runs_without_error():
    """skew=0, kurtosis=0 (pandas-excess convention -> normal distribution)
    is a supported, non-degenerate input."""
    dsr = deflated_sharpe_ratio(
        sharpe_observed=0.5, n_trials=8, skew=0.0, kurtosis=0.0, n_obs=252
    )
    assert 0.0 <= dsr <= 1.0


def test_dsr_more_trials_deflates_more():
    """Holding everything else fixed, more trials -> a higher pure-luck
    bar -> DSR should not increase (monotonic in the deflation direction)."""
    kwargs = dict(sharpe_observed=0.6, skew=-0.2, kurtosis=1.0, n_obs=252)
    dsr_few = deflated_sharpe_ratio(n_trials=2, **kwargs)
    dsr_many = deflated_sharpe_ratio(n_trials=200, **kwargs)
    assert dsr_many <= dsr_few


def test_dsr_propagates_none_for_degenerate_psr():
    """deflated_sharpe_ratio() must propagate probabilistic_sharpe_ratio()'s
    None sentinel verbatim rather than coercing a formula breakdown back
    into a number. skew=+2.0, sharpe_observed=+3.0 (pandas-excess
    kurtosis=0.0 -> raw kurtosis 3.0 after the internal +3 conversion) is
    the reviewer's own plausible-real-inputs example that drives the
    Mertens denominator negative:
        denom_inside = 1 - 2*3 + ((3-1)/4)*3**2 = 1 - 6 + 4.5 = -0.5 (< 0)
    This must surface as None end-to-end, not 0.5 -- the whole point of
    this fix is that a genuinely strong result never gets silently
    reported as a neutral one.
    """
    dsr = deflated_sharpe_ratio(
        sharpe_observed=3.0, n_trials=10, skew=2.0, kurtosis=0.0, n_obs=252
    )
    assert dsr is None


def test_dsr_none_does_not_crash_caller_style_formatting():
    """Mirrors the guard added at the ascent/research/walk_forward_runner.py
    call site (the same `is not None` idiom already used there for `wfe`):
    a None DSR must be safely distinguishable and printable/JSON-
    serializable without the walk-forward run crashing over a degenerate
    PSR for one fold's moments."""
    import json

    dsr = deflated_sharpe_ratio(
        sharpe_observed=3.0, n_trials=10, skew=2.0, kurtosis=0.0, n_obs=252
    )
    assert dsr is None

    # The runner's print guard: `dsr is not None` -> %.3f, else an "n/a"
    # message. Confirm both branches are safely reachable without a
    # TypeError from feeding None into %.3f.
    if dsr is not None:
        message = "%.3f" % dsr
    else:
        message = "n/a (PSR formula degenerated)"
    assert message == "n/a (PSR formula degenerated)"

    # The runner's JSON write: plain `json.dump` (not `default=float`
    # coercion for this field) must serialize None as `null`, not crash
    # or silently reinterpret it as 0.5.
    serialized = json.dumps({"deflated_sharpe_ratio": dsr})
    assert serialized == '{"deflated_sharpe_ratio": null}'


def test_dsr_le_psr_against_zero():
    """DSR (tested against a positive pure-luck benchmark whenever
    n_trials > 1) must never exceed the plain PSR-against-zero for the same
    strategy -- deflation can only make the bar harder to clear, never
    easier. Uses the same hand-checked scenario as the module's
    'observably deflated' case (SR=0.5, n_trials=100, n_obs=252,
    skew=-0.5, kurtosis(pandas-excess)=2.0)."""
    sharpe, skew, kurt_excess, n_obs = 0.5, -0.5, 2.0, 252

    dsr = deflated_sharpe_ratio(
        sharpe_observed=sharpe, n_trials=100, skew=skew, kurtosis=kurt_excess, n_obs=n_obs
    )
    psr_vs_zero = deflated_sharpe_ratio(
        sharpe_observed=sharpe, n_trials=1, skew=skew, kurtosis=kurt_excess, n_obs=n_obs
    )
    assert dsr <= psr_vs_zero
    assert dsr == pytest.approx(0.99995879245362, abs=1e-8)


def test_dsr_explicit_sr_variance_estimate_overrides_default():
    """When the caller supplies a real per-trial SR variance, it must be
    used verbatim rather than the Mertens-formula fallback."""
    common = dict(sharpe_observed=1.0, n_trials=10, skew=0.0, kurtosis=0.0, n_obs=101)
    dsr_default = deflated_sharpe_ratio(**common)
    dsr_explicit = deflated_sharpe_ratio(**common, sr_variance_estimate=0.5)
    # A much larger variance -> a much higher expected-max-SR benchmark ->
    # a lower (here, still saturating, but strictly non-larger) DSR.
    assert dsr_explicit <= dsr_default


def test_known_trial_count_is_documented_positive_int():
    """KNOWN_TRIAL_COUNT is a curated constant, not derived at import time
    -- just assert it's the sane, documented value so an accidental edit
    is caught."""
    assert isinstance(KNOWN_TRIAL_COUNT, int)
    assert KNOWN_TRIAL_COUNT == 8


def test_dsr_output_always_in_unit_interval():
    for sr in (-1.0, 0.0, 0.5, 1.0, 3.0):
        for nt in (1, 5, 50):
            dsr = deflated_sharpe_ratio(
                sharpe_observed=sr, n_trials=nt, skew=0.3, kurtosis=1.5, n_obs=252
            )
            assert 0.0 <= dsr <= 1.0
