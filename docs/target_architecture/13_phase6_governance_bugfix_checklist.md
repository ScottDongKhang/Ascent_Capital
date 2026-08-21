# Phase 6 Governance Bug Fix — Verified Root Cause and Exact Diff

Re-derived from source, not from the earlier blueprint's paraphrase. The root
cause is narrower and more specific than `06_judgment_governance.md` stated —
worth reading this before touching code.

## The single call site

`run_all_agents.py:2261` is the only place `update_authority()` is called
outside tests (confirmed via repo-wide grep). Its gating logic:

```python
# run_all_agents.py:2258-2270
_d_ret_today  = _cf_record.get("track_d_return")      if _cf_record else None
_as_ret_today = _cf_record.get("track_astar_return")   if _cf_record else None
if _d_ret_today is not None and _as_ret_today is not None:
    update_authority(
        track_d_return=_d_ret_today,
        track_astar_return=_as_ret_today,
        n_decisions_evaluated=_fb.get("n_decisions_evaluated", 0),
        hit_rate=_fb.get("hit_rate_21d"),
        profit_factor=_fb.get("profit_factor"),
        fade_rate=_fb.get("fade_rate"),
    )
else:
    print("[Runner] Authority update skipped — no Track D snapshot yet")
```

`earned_authority.py:166` has a second, redundant guard (`if track_d_return is
None or track_astar_return is None: skip`), so the buffer-append failure mode
is enforced twice — the caller already prevents the call, and the callee would
refuse it anyway if called.

## Verified root cause: NOT "AI PM doesn't run daily" — it's price-fetch fragility

The earlier blueprint (`06`) assumed the gap was structural — Track D only
gets a fresh value on rebalance days. **That assumption is wrong.**
`ai_pm_counterfactual.py::load_snapshots()` (line 370-378) returns weights
**from the last rebalance, forward-filled** — `_load_last_snapshot()` doesn't
require today's date, it returns the most recent snapshot ever written. So
`_d_w` (and `_as_w`) are non-empty on almost every day *after* the first
rebalance has ever happened, not just on rebalance days themselves.

The actual failure mode is upstream, in the price-fetch block that computes
`_cf_prices` (`run_all_agents.py:2166-2197`):

```python
# run_all_agents.py:2192-2197
# Visible warning: empty prices → Track A★/D record None (skipped),
# not a fabricated 0.0. Silent freeze here is what produced the
# fictional -11.6pp 'AI PM cost'.
if _as_w and not _cf_prices:
    print(f"[Runner] WARNING: counterfactual priced 0/{len(_cf_syms)} "
          f"snapshot symbols — Track A★/D will record None for {today}")
```

This block calls `yfinance.download()` once here (line 2176) for the
counterfactual snapshot symbols, and the code has **already been patched once**
to guard against `yfinance` returning a trailing all-NaN row for today's
unpublished bar (comment at lines 2181-2183: "yfinance returns a trailing
all-NaN row for today's unpublished bar, which otherwise makes every track
NaN/0.0 and freezes A★/D" — filtered via `.dropna()` and a `len(_ser) >= 2`
check at line 2185). **This is documented, in-repo evidence that this exact
fetch has already caused silent freezes once before** — the current code only
half-fixes it (drops NaN rows) but any *other* `yfinance` failure mode (rate
limit, network error, a symbol delisted/renamed, the whole download returning
empty) still produces `_cf_prices = {}`, which produces
`_d_ret_today = None`, which skips the buffer append entirely for that day —
silently, with only a `print()` (not durable-logged, not counted anywhere).

**This means the "buffers empty after 19 update cycles" symptom is most
plausibly explained by: the yfinance fetch at `run_all_agents.py:2176` failing
or returning insufficient data on a meaningful fraction of trading days,** not
by AI PM cadence. The 63-day (or 21/42-day) window counts consecutive
*successful* scoring days, and every failed fetch resets the effective clock
further out without ever being tallied as a distinct, trackable failure mode.

## Exact fix

1. **Instrument the failure, don't just print it.** At
   `run_all_agents.py:2195-2197`, in addition to the existing `print()`,
   append to a small counter file (or reuse `compliance/audit_trail.py`,
   already imported project-wide — see `12_phase0_execution_checklist.md`
   item 2) so "how many of the last N trading days had a counterfactual
   price-fetch failure" becomes a queryable number, not something only visible
   by grepping historical stdout logs.
2. **Add the `PROMOTION_PATH_STALLED` alarm** (as `06` originally specified,
   still correct in spirit even though the root cause differs): in
   `earned_authority.py`, track `days_since_last_buffer_append` in state
   (increment whenever `update_authority()` returns early at line 166-170;
   reset to 0 on a real append at line 172-176). If
   `days_since_last_buffer_append >= 2 * win` for the current promotion
   window, log a distinct `PROMOTION_PATH_STALLED` event — this is diagnosable
   today from `earned_authority.json`'s `last_updated` field plus buffer
   length, but isn't currently surfaced as its own signal distinct from
   `is_stuck()` (`earned_authority.py:55`, which only checks 63 days at
   current level with no cause breakdown).
3. **Consider a fallback price source or retry** for the `run_all_agents.
   py:2176` fetch specifically — it's a small, fixed symbol set (the union of
   three snapshot weight dicts), so a single retry with backoff, or falling
   back to the `prices_live` cache already maintained elsewhere in the
   pipeline instead of a fresh `yfinance.download()` call, would likely close
   most of the gap without new infrastructure.

## What NOT to do

Do not "fix" this by relaxing the `is None` guards at
`run_all_agents.py:2260` or `earned_authority.py:166` — those guards are
correct and intentional (per the in-code comment: skipping is what prevents
"a fabricated duplicate," referencing a real prior incident where a silent
0.0-fill produced a fictional -11.6pp AI PM cost figure). The fix is to reduce
*how often* the upstream fetch fails, and to make failures visible/countable,
not to weaken the safety guard that correctly refuses to buffer a fabricated
number.
