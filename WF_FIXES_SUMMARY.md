# Risk-engineering pass — 2026-08-24 (commit `b905973`)

Goal: turn the system's stated target — steady, low-drawdown monthly returns,
capital preservation first — into actual code, not just docs. Return stayed
roughly where it was; the drawdown profile is what changed.

## What changed

**Volatility target: 15% → 12% annualized**
`ascent/config/settings.py`, `ascent/portfolio/exposure.py`. The
walk-forward framework had its own disconnected hardcoded 15% default —
fixed to inherit the shared constant so research and live can't silently
diverge again.

**Kill switch tightened + new monthly circuit breaker**
`ascent/execution/kill_switch.py`. Hard stop 15% → 12%, soft warn 8% → 5%.
Added an independent monthly circuit breaker at 6% month-to-date drawdown —
previously the switch only watched whole-book peak-to-trough, so a bad month
near an all-time high wouldn't trip anything. Trip-reason-aware audit
logging so the compliance trail records which check actually fired.

**Selection objective: Sharpe → Calmar**
`ascent/research/self_improve.py`, `ascent/research/shadow_promoter.py`.
The self-improvement loop and shadow-promotion graduation both used to rank
candidate configs on Sharpe (penalizes upside and downside vol equally).
Now they rank on Calmar (return / max drawdown) — directly rewards "avoid
the big loss" instead of "good on average." Both files now share one
scoring helper (`score_variant()`) instead of two independently-drifting
copies of the same formula.

**Wipeout scoring bug**
`ascent/research/evaluation.py`. A total account wipeout (-100% in one day)
used to score Calmar = 0.0 — the same as a flat, no-drawdown variant. Fixed
to score -1.0 annualized, so a catastrophic variant can no longer outrank a
merely mediocre one in the promotion ranking.

**Fold-gap corruption in the lightweight OOS path**
`ascent/research/walk_forward_lightweight.py`. The walk-forward folds used
by self-improve aren't contiguous (there's an untested embargo gap between
them); the return series feeding Calmar used to concatenate folds directly,
which could hide a real drawdown or fabricate a fake recovery across the
gap. Now the gap is explicitly zero-filled. Turnover penalty rescaled to
match Calmar's smaller typical scale.

## Verification

- 3 independent code-review passes (2 before an unrelated worktree
  incident, re-verified byte-identical after recovery)
- 231/232 relevant tests pass — the one failure depends on a sibling
  branch's in-progress universe-restriction change landing first, not a
  bug in this commit
- `scripts/verify_docs.py`: 24/24

## Not in this commit

Universe-restriction consistency across live/backtest/self-improve, a
duplicated cost-model implementation, and a couple of walk-forward
provenance/bias findings — all real, all flagged to the session doing that
work, landing separately.
