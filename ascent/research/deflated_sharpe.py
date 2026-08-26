"""
ascent/research/deflated_sharpe.py
-----------------------------------
Deflated Sharpe Ratio (DSR).

From Bailey & Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality", Journal of
Portfolio Management, 2014 (SSRN 2460551).

DSR answers a narrower question than the plain Sharpe ratio: given that we
picked the best-looking configuration out of `n_trials` candidates we tried
(sleeve variants, threshold retunes, universe changes, ...), and given that
the underlying return series is not normally distributed (has skew and
excess kurtosis), what is the probability that the *true* Sharpe ratio of
the selected configuration is actually greater than zero? A high naive
Sharpe on a strategy that was the survivor of many trials is expected even
under the null of no real skill (this is exactly what CLAUDE.md integrity
constraint #7 and the fundamental-sleeve kill are about) -- DSR is the
correction for that selection effect, plus a separate correction for
non-normal returns that the plain Sharpe ratio silently assumes away.

DSR is built from two pieces, both implemented below:

1. `expected_max_sharpe()` -- the expected value of the MAXIMUM Sharpe ratio
   you'd see across `n_trials` independent trials, purely from sampling
   noise, if the true Sharpe of every trial were zero. This becomes the
   benchmark DSR tests against, instead of testing against zero directly.
2. `probabilistic_sharpe_ratio()` -- the non-normality-corrected test
   statistic (PSR) from Bailey & Lopez de Prado, "The Sharpe Ratio
   Efficient Frontier" (2012) / Mertens (2002), evaluated against that
   benchmark instead of against zero.

DSR = PSR(SR_observed, SR_benchmark = expected_max_sharpe(...))

------------------------------------------------------------------------
LIMITATION -- the SR-variance-across-trials input (read this before using)
------------------------------------------------------------------------
The paper's E[max SR] formula needs the variance of the Sharpe ratio
*estimator*, taken across the population of trials:

    E[max_SR] ~= sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N)
                            + gamma * Phi^-1(1 - 1/(N * e)))

where V = Var(SR_hat) across the N trial Sharpe estimates, gamma is the
Euler-Mascheroni constant (~0.5772), and Phi^-1 is the inverse standard
normal CDF.

This codebase does not log a per-trial Sharpe ratio for every historical
config that was tried (`ascent/research/hypothesis_registry.py` reads
`logs/self_improve_log.jsonl`, which holds only ~a few dozen self-improve
variant entries -- not the full set of trials enumerated in
`KNOWN_TRIAL_COUNT` below, most of which predate that logger or were never
run through `self_improve.py` at all: universe restriction, cost-model
changes, and the AI-PM/debate layer removal are not "variants" in that
loop's sense). So V cannot be estimated empirically from trial dispersion
here, and the function cannot silently substitute 0 or 1 for it without
hand-waving the correction the paper is built around.

Chosen fallback (the `sr_variance_estimate` parameter, default `None`):
when not supplied by the caller, this module falls back to the asymptotic
sampling variance of the Sharpe ratio ESTIMATOR ITSELF -- the same
Mertens (2002) formula the paper already uses for the PSR denominator,
evaluated at `sharpe_observed`, `skew`, `kurtosis`, `n_obs`:

    V_default = (1 - skew * SR + ((kurtosis - 1) / 4) * SR**2) / (n_obs - 1)

Justification: this is the variance of *one* Sharpe estimate drawn from a
return series with these moments and this sample size. Using it as a proxy
for "variance across trial Sharpes" is the standard simplifying assumption
in public reproductions of this paper (e.g. the widely-cited `mlfinlab`
`deflated_sharpe_ratio` implementation, and Lopez de Prado's own course
material) when the trials' individual Sharpes/variances were not
separately tracked: it assumes every trial was measured on a comparably
sized, comparably shaped return series to the one actually reported, so
its own sampling variance is a reasonable stand-in for the population
variance across trials. It is a documented approximation, not a measured
quantity -- callers who *do* have per-trial Sharpe ratios on hand should
compute `np.var(trial_sharpes, ddof=1)` and pass it explicitly via
`sr_variance_estimate`, which this function will always prefer over the
fallback.

Do not treat `deflated_sharpe_ratio()`'s output as more precise than this
input allows: it is exactly as good as the `n_trials` count and the
variance assumption behind it, both of which are documented judgment calls
(see `KNOWN_TRIAL_COUNT` below), not mechanically derived facts.
"""
from __future__ import annotations

import math

from scipy.stats import norm

# Euler-Mascheroni constant.
_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# KNOWN_TRIAL_COUNT -- human-curated, NOT mechanically derived
# ---------------------------------------------------------------------------
# This is a curated judgment call, not a count read out of a log. The two
# places that *could* mechanically derive a trial count were checked and are
# both insufficient on their own:
#   - `ascent/research/hypothesis_registry.py` reads
#     `logs/self_improve_log.jsonl` / writes `logs/hypothesis_registry.jsonl`.
#     As of this writing `logs/self_improve_log.jsonl` holds only a handful of
#     self-improve variant-evaluation entries and `logs/hypothesis_registry.jsonl`
#     does not exist yet on disk -- neither captures the broader set of
#     distinct strategy/config trials this project has actually run and
#     compared against the OOS window (threshold retunes, universe changes,
#     cost-model changes, the entire AI-PM/debate layer build-and-removal,
#     and this statistical-rigor batch itself were never routed through that
#     logger).
#   - `logs/sleeve_ic_log.jsonl` (read by `ascent/alpha/stack.py`
#     `_get_gated_weights()`) tracks per-sleeve rolling IC, not distinct
#     backtest configurations.
#
# So this constant is a deliberately curated list of distinct
# strategy/config trials whose OOS performance was evaluated and compared
# against this project's walk-forward window, drawn from
# `docs/session_log_archive.md` and this worktree's own commit history.
# Reviewer: treat this as a starting estimate to sanity-check, not ground
# truth -- it is exactly the kind of number this file's own docstring says
# not to over-trust.
#
#   1. Fundamental sleeve       -- tried, killed (measured anti-signal;
#                                   CLAUDE.md integrity constraint #7).
#   2. Trend sleeve              -- tried, killed (see 2026-06-24 session log
#                                   entry: trend IC driven negative by a data
#                                   error, subsequently not restored to the
#                                   active 2-sleeve set; DEFAULT_ALPHA_WEIGHTS
#                                   is {meanrev, statarb} only).
#   3. IC-gate threshold retune  -- constant changed -0.010 -> -0.005
#                                   (2026-06-03 session log entry;
#                                   ascent/alpha/stack.py IC_GATE_THRESHOLD).
#   4. Universe restriction      -- sp500_only / strict=True fix aligning
#                                   live, WF, and self-improve universes
#                                   (commit 62c91a8, this worktree).
#   5. Cost-model change         -- liquidity-scaled impact cost model added
#                                   to ascent/backtest/costs.py
#                                   (commit 62c91a8, this worktree).
#   6. AI-PM / debate / earned-authority / falsifier layer -- built, run
#                                   advisory-only from 2026-08-14, measured
#                                   negative-or-insignificant on every axis,
#                                   removed outright 2026-08-23 (CLAUDE.md
#                                   integrity constraint #5 history).
#   7. The 5 walk-forward-accuracy bugfixes -- universe re-cap gap, live/WF
#                                   universe alignment, walk_forward_lightweight
#                                   universe fix, WFE tracking, cost-model
#                                   duplication review (commit 62c91a8, this
#                                   worktree; counted as one bundled trial
#                                   since they landed as a single evaluated
#                                   commit, not five independently-run
#                                   configurations).
#   8. This statistical-rigor batch -- the 8-item batch this module is part
#                                   of (Lo-adjusted Sharpe, this DSR module,
#                                   CPCV, and related additions on
#                                   feat/wf-statistical-rigor).
#
# => KNOWN_TRIAL_COUNT = 8
#
# This almost certainly UNDERCOUNTS the true number of configurations ever
# informally tried (e.g. individual alpha-weight tweaks inside
# self_improve.py's own search loop are each technically a "trial" but are
# deliberately NOT broken out here as separate top-level items -- they are
# folded into item 7/8's bundled commits since they were not independently
# reported against the OOS window the way items 1-6 were). Bump this
# constant, with an inline citation added above, the next time a genuinely
# new, independently-evaluated configuration is run -- do not silently
# reuse 8 forever.
KNOWN_TRIAL_COUNT = 8


def expected_max_sharpe(n_trials: int, sr_variance_estimate: float) -> float:
    """Expected value of the maximum Sharpe ratio observed across
    `n_trials` independent trials, under the null that every trial's true
    Sharpe ratio is zero (pure sampling noise). Bailey & Lopez de Prado
    (2014), eq. 8 (standard extreme-value approximation for the max of N
    correlated-or-independent approximately-Normal variables):

        E[max_SR] ~= sqrt(V) * ((1 - gamma) * Phi^-1(1 - 1/N)
                                 + gamma * Phi^-1(1 - 1/(N * e)))

    Args:
        n_trials: number of independent trials (N). Must be >= 1.
        sr_variance_estimate: V, the variance of the Sharpe ratio estimator
            across trials. See module docstring for how this codebase
            estimates it when true per-trial dispersion isn't tracked.

    Returns:
        Expected maximum Sharpe ratio under the zero-skill null. For
        `n_trials <= 1` there is no selection effect to correct for (a
        single trial's expected max IS just that trial, and the formula's
        `Phi^-1(1 - 1/1) = Phi^-1(0) = -inf` term is degenerate), so this
        returns 0.0 -- i.e. "test against zero," which is the correct
        single-trial special case.
    """
    if n_trials <= 1:
        return 0.0
    if sr_variance_estimate < 0:
        raise ValueError(f"sr_variance_estimate must be >= 0, got {sr_variance_estimate}")

    sigma = math.sqrt(sr_variance_estimate)
    term1 = (1 - _GAMMA) * norm.ppf(1 - 1.0 / n_trials)
    term2 = _GAMMA * norm.ppf(1 - 1.0 / (n_trials * math.e))
    return sigma * (term1 + term2)


def probabilistic_sharpe_ratio(
    sharpe_observed: float,
    sharpe_benchmark: float,
    skew: float,
    kurtosis: float,
    n_obs: int,
) -> float | None:
    """Probabilistic Sharpe Ratio (PSR): P(true SR > sharpe_benchmark),
    correcting for non-normal returns (skew, excess kurtosis) and finite
    sample size. Bailey & Lopez de Prado / Mertens (2002):

        PSR = Phi( (SR_obs - SR_bench) * sqrt(n_obs - 1)
                   / sqrt(1 - skew*SR_obs + ((kurtosis - 1) / 4) * SR_obs**2) )

    `kurtosis` here is EXCESS kurtosis convention as used in the paper's
    formula (normal distribution -> kurtosis term should be passed such
    that (kurtosis - 1)/4 vanishes for normal-with-kurtosis=3, i.e. this
    function expects pandas' `.kurtosis()` non-excess-adjusted-by-3
    convention to be pre-adjusted by the caller if needed -- see
    `deflated_sharpe_ratio()` below, which passes pandas `.kurtosis()`
    output straight through: pandas' `.kurtosis()` already returns EXCESS
    kurtosis (0 for a normal distribution), matching this formula's
    `(kurtosis - 1)/4` term only if 1 is subtracted from a *raw* kurtosis.
    To avoid this ambiguity `deflated_sharpe_ratio()` documents exactly
    which convention it feeds in.

    Args:
        sharpe_observed: observed (non-annualized-vs-annualized both fine
            as long as consistent) Sharpe ratio of the strategy.
        sharpe_benchmark: the Sharpe ratio to test against (0 for a plain
            PSR test; `expected_max_sharpe(...)` for the deflated version).
        skew: sample skewness of the return series.
        kurtosis: sample RAW kurtosis of the return series (i.e. 3.0 for a
            normal distribution) -- see `deflated_sharpe_ratio()` for the
            excess-kurtosis conversion from pandas.
        n_obs: number of return observations.

    Returns:
        PSR in [0, 1] normally. Two DISTINCT degenerate cases are handled
        differently on purpose -- collapsing them to the same sentinel was
        the bug this docstring now documents:

        - `n_obs <= 1`: there is genuinely no defensible test statistic
          (can't take sqrt(n_obs - 1) meaningfully), so this returns the
          documented `0.5` sentinel -- "uninformative, no evidence either
          way." This is a real, if boring, answer and is safe to persist
          as-is.
        - The Mertens (2002) denominator
          (`1 - skew*SR + ((kurtosis-1)/4)*SR**2`) going non-positive (or
          NaN): this is NOT the same as "uninformative." It means the PSR
          formula itself has broken down for this particular
          skew/kurtosis/sharpe_observed combination -- which happens for
          plausible real inputs, not just contrived ones (e.g.
          skew=+2.0, sharpe_observed=+3.0 already drives it negative) --
          and can occur even with a strategy that has genuinely strong,
          measurable skill. Silently returning `0.5` here would misreport
          a formula breakdown as a neutral "no skill detected" result, and
          that number can then get persisted into an artifact (e.g.
          `wf_report.json`'s `deflated_sharpe_ratio` field) with no way
          for a reader to tell the two cases apart. So this case returns
          `None` instead -- a value a caller cannot mistake for a real
          probability, and that a JSON writer will happily serialize as
          `null` rather than a plausible-looking `0.5`. Callers (see
          `deflated_sharpe_ratio()` below and
          `ascent/research/walk_forward_runner.py`'s `wf_report` write)
          must propagate `None` rather than coercing it back to a number.
    """
    if n_obs <= 1:
        return 0.5

    denom_inside = 1 - skew * sharpe_observed + ((kurtosis - 1) / 4.0) * sharpe_observed ** 2
    if denom_inside <= 0 or math.isnan(denom_inside):
        # Formula breakdown, NOT "uninformative" -- see docstring. Do not
        # collapse this to 0.5; that would misreport a possibly-strong
        # result as a neutral one.
        return None

    denom = math.sqrt(denom_inside)
    z = (sharpe_observed - sharpe_benchmark) * math.sqrt(n_obs - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    skew: float,
    kurtosis: float,
    n_obs: int,
    sr_variance_estimate: float | None = None,
) -> float | None:
    """Deflated Sharpe Ratio (DSR): probability the true Sharpe ratio of
    the SELECTED (best-of-n_trials, i.e. survivorship-biased) configuration
    exceeds zero, after correcting for (a) selection bias across
    `n_trials` and (b) non-normal returns.

    DSR = PSR(sharpe_observed, sharpe_benchmark=expected_max_sharpe(...))

    Args:
        sharpe_observed: observed Sharpe ratio of the strategy actually
            reported (e.g. `wf_summary["sharpe"]`).
        n_trials: number of distinct configurations whose OOS performance
            was evaluated and compared before this one was selected/
            reported. See `KNOWN_TRIAL_COUNT` in this module for this
            project's curated estimate and its citations -- pass that
            constant, or a more precise count if you have one, don't
            invent a number here.
        skew: sample skewness of the return series (`returns.skew()`).
        kurtosis: sample kurtosis of the return series. Pass pandas'
            `.kurtosis()` output DIRECTLY -- pandas already returns EXCESS
            kurtosis (0.0 for normal), and this function internally adds
            back the +3 to match `probabilistic_sharpe_ratio()`'s raw-
            kurtosis convention. Do not pre-adjust.
        n_obs: number of return observations (`len(returns)`).
        sr_variance_estimate: variance of the Sharpe ratio estimator across
            trials (V in `expected_max_sharpe()`). If None (default), falls
            back to the Mertens (2002) asymptotic sampling variance of
            `sharpe_observed` itself -- see the module docstring's
            "LIMITATION" section for the full justification. Callers with
            real per-trial Sharpe dispersion should pass
            `np.var(trial_sharpes, ddof=1)` explicitly.

    Returns:
        DSR in [0, 1] normally. Interpretation: probability the true Sharpe
        ratio exceeds the expected maximum Sharpe achievable by pure luck
        across `n_trials` trials, given the observed return distribution's
        non-normality. DSR near 1.0 = strong evidence of genuine,
        non-overfit, non-normality-adjusted skill. DSR near 0.5 = the
        result is statistically indistinguishable from what `n_trials` of
        random search would produce.

        Can also return `None` -- see `probabilistic_sharpe_ratio()`'s
        docstring for the two degenerate cases it distinguishes. `n_obs <=
        1` still returns the real `0.5` sentinel (genuinely uninformative,
        safe to persist). But if the Mertens (2002) denominator this
        function derives its default `sr_variance_estimate` from -- and
        the one `probabilistic_sharpe_ratio()` independently recomputes
        using `sharpe_benchmark` instead of 0 -- goes non-positive for the
        given skew/kurtosis/sharpe_observed combination, this propagates
        `None` straight through rather than silently returning `0.5`.
        `None` is NOT "no skill detected"; it means the PSR formula
        degenerated for these inputs and no probability was computed.
        Callers (notably `ascent/research/walk_forward_runner.py`'s
        `wf_report_<date>.json` writer) must check for `None` explicitly
        and record it as such (e.g. JSON `null` plus a `_meta` note) rather
        than defaulting it back to a number.

    Edge cases:
        - `n_trials <= 1`: no selection-bias correction is applied
          (`expected_max_sharpe` returns 0.0), so DSR reduces to a plain
          PSR test against zero.
        - `skew == 0, kurtosis == 0` (i.e. raw kurtosis 3.0 after the +3
          adjustment -- a normal distribution): the PSR non-normality
          correction term collapses to `1 - 0 + (2/4)*SR**2` ... note it
          does NOT collapse all the way to the textbook
          `PSR = Phi((SR_obs - SR_bench) * sqrt(n_obs - 1))` form used by
          some simplified expositions, because the `(kurtosis-1)/4 * SR**2`
          term does not vanish at kurtosis=3 unless SR itself is small;
          this is the actual Mertens (2002) formula, not the further-
          simplified textbook special case, and is the one the paper cites.
        - Denominator degeneration (extreme skew/kurtosis/SR combination):
          returns `None`. See `probabilistic_sharpe_ratio()` docstring.
    """
    if sr_variance_estimate is None:
        raw_kurtosis_default = kurtosis + 3.0
        denom_inside = 1 - skew * sharpe_observed + \
            ((raw_kurtosis_default - 1) / 4.0) * sharpe_observed ** 2
        n_eff = max(n_obs, 2)
        sr_variance_estimate = max(denom_inside, 0.0) / (n_eff - 1)

    sharpe_benchmark = expected_max_sharpe(n_trials, sr_variance_estimate)

    # probabilistic_sharpe_ratio() expects RAW kurtosis; pandas-style
    # `.kurtosis()` (excess) is converted here, once, in one place.
    raw_kurtosis = kurtosis + 3.0

    return probabilistic_sharpe_ratio(
        sharpe_observed=sharpe_observed,
        sharpe_benchmark=sharpe_benchmark,
        skew=skew,
        kurtosis=raw_kurtosis,
        n_obs=n_obs,
    )
