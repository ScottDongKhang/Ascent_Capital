#!/usr/bin/env python
"""
Package the 2026-08-15 walk_forward_runner.py validation into the
wf_report_clean_*.json schema that ascent/reporting/verified_numbers.py reads.

Why this exists
----------------
CANONICAL_WF_ARTIFACT pointed at outputs/wf_results/wf_report_clean_2026-06-22.json,
produced by scripts/run_ascent_wf.py / ascent/research/wf_framework/. That framework
has a confirmed, still-open bug: ascent_strategy.py::_make_alpha_weights force-injects
the `trend` sleeve (measured CUT/negative signal, not in DEFAULT_ALPHA_WEIGHTS) and
bypasses the IC gate entirely, because ascent/alpha/stack.py's gate only runs when
alpha_weights is None, and wf_framework always passes an explicit override. So the
project's own canonical, citable number did not reflect the actual shipped 2-sleeve
(meanrev + statarb) system.

This script does not re-run anything. It reads the already-verified summary numbers
out of the 2026-08-15 walk_forward_runner.py log (independently re-grepped from the
raw log twice this session) and writes them into the same JSON shape the old artifact
used, so canonical_wf() can be repointed at a run of the real shipped system.

Source of the numbers
----------------------
outputs/wf_results/wf_run_target_architecture_2026-08-15_post_phantom_fix.log
  (grep for "PERFORMANCE REPORT" / "FOLD SUMMARY" — the file is ~33k lines, don't
  read it whole)
outputs/wf_results/vc-task-4-post-phantom-fix-report.md
  (human-readable summary of the same run, independently cross-checked)

Known gap: this artifact has no `wfe` (Walk-Forward Efficiency). WFE is defined in
ascent/research/wf_framework/metrics.py as mean(OOS_Sharpe_fold / IS_Sharpe_fold)
across folds -- a metric specific to that framework's per-fold in-sample refit.
ascent/research/walk_forward_runner.py (the successor, and the one that actually
produced this run) does not track per-fold in-sample Sharpe and has no equivalent
metric. Rather than fabricate a WFE figure, this script writes "wfe": null.
ascent/reporting/verified_numbers.py treats wfe as optional for exactly this reason
and attaches an explicit caveat instead of a number.

Usage
-----
    .venv/bin/python scripts/generate_wf_report_from_runner.py
    .venv/bin/python scripts/generate_wf_report_from_runner.py --check   # print only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "outputs" / "wf_results" / "wf_report_clean_2026-08-15.json"

SOURCE_LOG = "outputs/wf_results/wf_run_target_architecture_2026-08-15_post_phantom_fix.log"
SOURCE_REPORT = "outputs/wf_results/vc-task-4-post-phantom-fix-report.md"

# Verbatim from the PERFORMANCE REPORT block in SOURCE_LOG (grepped independently
# twice this session; also tabulated in SOURCE_REPORT's "post_phantom_fix" column).
REPORT = {
    "cagr": 0.1020,
    "volatility": 0.2458,
    "sharpe": 0.415,
    "sortino": 0.551,
    "max_drawdown": -0.4565,
    "win_rate": 0.523,
    # WFE is not computed by walk_forward_runner.py -- see module docstring.
    # Do not fill this with a number; verified_numbers.py handles None explicitly.
    "wfe": None,
    "n_folds": 165,
    "n_oos_days": 1641,
    # Log labels this "Alpha: -3.62%" in the PERFORMANCE REPORT block. It is the
    # strategy's raw excess return over the benchmark for this window, not a
    # regression-fit Jensen's alpha -- same field name and slot as the old
    # artifact's "alpha", but confirm the definition before comparing the two
    # numbers directly.
    "alpha": -0.0362,
    "beta": 0.947,
    "_meta": {
        "source_log": SOURCE_LOG,
        "source_report": SOURCE_REPORT,
        "framework": "ascent/research/walk_forward_runner.py (walk_forward_pipeline)",
        "llm_disabled": None,  # not applicable to this framework; not asserted
        # The shipped 2-sleeve alpha stack per CLAUDE.md's integrity constraint 6.
        # This run's [alpha_stack] log line lists every sleeve *loaded* into the
        # stack object (trend, meanrev, volatility, statarb, fundamental, earnings,
        # analyst, options_flow, insider, earnings_tone, narrative) -- that is not
        # the same as which sleeves carry nonzero weight. The active, weighted set
        # is the 2-sleeve DEFAULT_ALPHA_WEIGHTS (meanrev/statarb) per
        # ascent/alpha/stack.py and ascent/research/self_improve.py.
        "alpha_overrides": {"meanrev": 0.5, "statarb": 0.5},
        "oos_window": "2020-01-02 -> 2026-07-15",
        "oos_years": None,
        "spy_cagr_same_window": 0.1382,
        "strat_cagr_recomputed": None,
        "excess_cagr_vs_spy": -0.0362,
        "n_symbols": None,
        "avg_universe_size": 452.9,
        "total_return": 0.8826,
        "calmar_ratio": 0.223,
        "profit_factor": 1.11,
        "avg_turnover_per_day": 0.1001,
        "avg_positions": 10.9,
        "excess_sharpe": -0.222,
        "notes": (
            "Replaces CANONICAL_WF_ARTIFACT's prior pointer at "
            "wf_report_clean_2026-06-22.json, which was produced by "
            "scripts/run_ascent_wf.py / ascent/research/wf_framework/ -- a "
            "framework with a confirmed, still-open bug: it force-injects the "
            "CUT `trend` sleeve and bypasses the IC gate entirely (see "
            "ascent_strategy.py::_make_alpha_weights). This artifact instead "
            "packages a clean run of the actual shipped 2-sleeve system via "
            "ascent/research/walk_forward_runner.py, independently re-verified "
            "twice in this session. Sortino here is NOT annualization-corrected "
            "the way ascent/research/wf_framework/metrics.py's sortino() is -- "
            "it is read verbatim from walk_forward_runner.py's own report, a "
            "different implementation. verified_numbers.py does not surface "
            "sortino at all (WalkForwardRecord has no sortino field), so this "
            "does not affect canonical_wf() output; it is included here only "
            "for completeness against the old artifact's schema."
        ),
    },
}


def build() -> dict:
    return REPORT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                     help="print the artifact without writing it")
    args = ap.parse_args()

    report = build()
    text = json.dumps(report, indent=2) + "\n"

    if args.check:
        print(text)
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
