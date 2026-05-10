# Plan 5 — Execution Excellence

> **For agentic workers:** Use superpowers:subagent-driven-development to execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Transform execution from single market-order-at-close into institutional-quality execution with TWAP scheduling, implementation shortfall (IS) measurement, capacity modeling, and intraday rebalance triggers. As AUM grows, execution quality is alpha — a $10M portfolio eating 30bps of unnecessary slippage loses $30,000 per rebalance. This plan closes that gap.

**Architecture:** Three layers added on top of the existing execution stack. (1) TWAP executor: large orders are split into child orders executed over a scheduled window. (2) IS measurement: three-part decomposition (decision price → arrival price → fill price) replaces the current single-delta slippage measurement. (3) Capacity model: computes the maximum AUM at which each strategy component's signal still exceeds its own market impact cost. Intraday triggers from Plan 3 (event agent) and emergency regime refits already exist — this plan provides the execution infrastructure they call into.

**Tech Stack:** Python 3.12, `alpaca-trade-api`, existing `ascent/execution/order_engine.py`, `ascent/execution/cost_model.py` (Almgren-Chriss, already partially built), `ascent/execution/slippage_tracker.py`.

**Prerequisites:** Plan 3 (Event-Driven Architecture) should be built first — the event runner uses the TWAP executor. But the TWAP executor can be built independently and Plan 3 can be retrofitted.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `ascent/execution/twap_executor.py` | TWAP schedule builder + child order submitter |
| Create | `ascent/execution/implementation_shortfall.py` | IS decomposition — decision / arrival / fill |
| Create | `ascent/execution/capacity_model.py` | Max AUM before signal decay from market impact |
| Create | `ascent/execution/intraday_trigger.py` | Intraday rebalance trigger logic |
| Modify | `ascent/execution/order_engine.py` | Route large orders through TWAP executor |
| Modify | `ascent/execution/slippage_tracker.py` | Add IS decomposition to fill log |
| Modify | `ascent/execution/cost_model.py` | Productionize Almgren-Chriss; connect to TWAP sizing |
| Modify | `ascent/execution/eod_runner.py` | Call intraday trigger check; log IS per rebalance |
| Create | `tests/test_execution_excellence.py` | Full test suite — 16 tests |

---

## Task 1: TWAP Executor

**File:** `ascent/execution/twap_executor.py`

### Steps
- [ ] 1.1 Write `build_twap_schedule(symbol, total_shares, side, start_time, window_minutes=60, n_slices=6) -> list[dict]`. Returns a list of `{"symbol": str, "shares": int, "side": str, "submit_at": datetime}`. Slices are equally spaced over `window_minutes`. Minimum slice size is 1 share — if total_shares / n_slices < 1, use n_slices = total_shares (one-share-per-slice).
- [ ] 1.2 Write `compute_twap_window(adv, trade_size_shares, urgency="normal") -> int`. Returns execution window in minutes. Uses Almgren-Chriss optimal execution time formula: `T* = sqrt(trade_size / (2 · η · σ²))` where η is the temporary impact coefficient and σ is daily volatility. Practical bounds: min 5 minutes (urgent), max 90 minutes (routine). Urgency override: "high" → max 15 minutes, "low" → max 90 minutes.
- [ ] 1.3 Write `execute_twap(symbol, total_dollars, side, adv, price, urgency="normal") -> list[dict]`. Converts dollars to shares, builds schedule, submits child orders via Alpaca as limit orders (bid+1tick for buys, ask-1tick for sells, refreshed each slice). Returns list of fill dicts. If a slice expires unfilled, rolls the remaining shares into the next slice (catching-up).
- [ ] 1.4 Write `should_use_twap(trade_size_dollars, adv_dollars, threshold_pct=0.05) -> bool`. Returns True when trade size exceeds 5% of ADV. Below this threshold, market order is fine — TWAP would add unnecessary complexity and time risk.
- [ ] 1.5 Wire into `order_engine.py`: in `submit_order()`, check `should_use_twap()`. If True and `TWAP_ENABLED = True`, call `execute_twap()`. If False or disabled, use existing market order path. TWAP_ENABLED defaults to False — paper-mode testing first.

---

## Task 2: Implementation Shortfall Measurement

**File:** `ascent/execution/implementation_shortfall.py`

### Steps
- [ ] 2.1 Define IS decomposition. Implementation shortfall = total cost vs. paper portfolio that traded at decision price. Three components:
  - **Delay cost**: price movement from signal generation (1:45 PM) to order submission (close-ish). = (arrival_price − decision_price) × direction.
  - **Market impact**: price movement from order submission to fill. = (fill_price − arrival_price) × direction.
  - **Opportunity cost**: shares not filled × (end-of-day price − decision_price) × direction.
- [ ] 2.2 Write `record_decision_price(symbol, decision_price, decision_time) -> None`. Called at signal generation time (before orders are built). Stores to a temporary in-memory dict keyed by symbol. Decision price = last trade price at time of signal generation.
- [ ] 2.3 Write `record_arrival_price(symbol, arrival_price, arrival_time) -> None`. Called when the first child order is submitted (for TWAP) or when the order is submitted (for market orders).
- [ ] 2.4 Write `compute_is(symbol, fill_records, eod_price) -> dict`. Called after all fills are complete. Returns `{"symbol", "decision_price", "arrival_price", "avg_fill_price", "eod_price", "delay_cost_bps", "market_impact_bps", "opportunity_cost_bps", "total_is_bps"}`. Costs in basis points.
- [ ] 2.5 Write `log_is_record(is_dict, rebalance_date) -> None`. Appends to `logs/slippage_log.jsonl` (existing file) with `"is_breakdown"` sub-dict. Backwards-compatible — old entries without IS decomposition are not modified.
- [ ] 2.6 Write `is_summary(lookback_days=63) -> dict`. Reads `logs/slippage_log.jsonl`, computes mean and std of each IS component over trailing period. Returns summary dict. Used by dashboard and investor reporting.

---

## Task 3: Capacity Model

**File:** `ascent/execution/capacity_model.py`

### Steps
- [ ] 3.1 Write `estimate_market_impact(trade_size_pct_adv, volatility, participation_rate=0.1) -> float`. Uses Almgren-Chriss: temporary impact = η · σ · (trade_size / ADV)^0.6, where η ≈ 0.142 (empirically estimated). Returns impact in basis points. This is already partially implemented in `cost_model.py` — productionize it here.
- [ ] 3.2 Write `compute_signal_breakeven_adv(signal_ic, signal_ir, holding_period_days=5) -> float`. Returns the minimum ADV (as a fraction of position size) at which the signal's expected gross alpha exceeds its own market impact. Formula: `alpha_per_period = IC · σ · (2/π)^0.5 · holding_days^0.5`. `breakeven_adv = (position_size / (alpha / impact_coefficient))^(1/0.6)`.
- [ ] 3.3 Write `compute_strategy_capacity(portfolio_weights, signal_ic_by_sleeve) -> dict`. For each sleeve, estimates the maximum NAV at which the sleeve's signal survives its own market impact given the current portfolio weights and universe ADV distribution. Returns `{"by_sleeve": {sleeve: max_nav}, "binding_constraint": str, "overall_capacity": float}`.
- [ ] 3.4 Write `capacity_report(prices_df, portfolio_weights) -> dict`. Runs the capacity model against current holdings. Logs to `logs/capacity_log.jsonl`. Called weekly (Sunday). Output includes the binding capacity constraint — the sleeve or position that limits total strategy capacity first.
- [ ] 3.5 The capacity model is informational only — it does not block trades or modify weights. Its output is visible in the live dashboard and investor report.

---

## Task 4: Intraday Trigger

**File:** `ascent/execution/intraday_trigger.py`

### Steps
- [ ] 4.1 Write `check_intraday_triggers(portfolio_state, market_data) -> list[dict]`. Evaluates three triggers: (a) regime emergency — SPY −3% intraday AND VIX > 30 → partial de-risk (multiply all weights by 0.70, same as SPY 200MA overlay); (b) kill switch approach — drawdown exceeds 12% (soft warn threshold) → reduce gross exposure by 20% preemptively; (c) event agent urgency — if any "high" urgency event fired in the last 60 minutes for a top-5 position → flag for immediate partial trim. Returns list of trigger dicts or empty list.
- [ ] 4.2 Write `execute_intraday_adjustment(trigger, current_positions, alpaca_broker) -> dict`. Submits the minimal set of orders to implement the triggered adjustment. For de-risk: proportionally trim all positions to the new target. For event trim: trim just the flagged position. All intraday adjustments use TWAP with urgency="high" (15-minute window).
- [ ] 4.3 Wire into `eod_runner.py`: call `check_intraday_triggers()` at 12:00 PM ET and 14:30 PM ET daily. If triggers fire, execute adjustments before the 1:45 PM daily rebalance runs. The daily rebalance then starts from the adjusted portfolio state.
- [ ] 4.4 All intraday adjustments logged to `logs/intraday_adjustments.jsonl`. Post-hoc analysis shows whether intraday adjustments improved or worsened outcome vs. waiting for daily rebalance.

---

## Task 5: Fill Quality Analytics

### Steps
- [ ] 5.1 In `ascent/execution/slippage_tracker.py` (modify existing): after computing existing slippage (signal price vs. fill price), also call `compute_is()` to add the three-part IS decomposition to the log entry.
- [ ] 5.2 Write `fill_quality_report(lookback_days=63) -> dict`. Returns: `{"mean_is_bps": float, "mean_delay_cost_bps": float, "mean_market_impact_bps": float, "mean_opportunity_cost_bps": float, "twap_vs_market_order_comparison": dict, "by_sleeve": dict}`. Breaks down IS by sleeve (which sleeves have higher execution cost?), by order type (TWAP vs market order), and by regime.
- [ ] 5.3 Add fill quality summary to the weekly IC brief (`ascent/reporting/ic_brief_generator.py`). One paragraph: average IS this week, dominant cost component, any outliers.

---

## Task 6: Tests

**File:** `tests/test_execution_excellence.py` — 16 tests

- [ ] `test_twap_schedule_correct_slice_count` — n_slices orders at equal intervals
- [ ] `test_twap_schedule_shares_sum_to_total` — sum of child order shares = total
- [ ] `test_twap_schedule_minimum_slice_size` — shares_per_slice ≥ 1
- [ ] `test_compute_twap_window_respects_urgency` — high urgency ≤ 15 min
- [ ] `test_should_use_twap_true_above_threshold` — 6% ADV → True
- [ ] `test_should_use_twap_false_below_threshold` — 3% ADV → False
- [ ] `test_twap_disabled_uses_market_order` — TWAP_ENABLED=False → no TWAP call
- [ ] `test_is_delay_cost_positive_on_adverse_move` — price moves against you before submission → delay cost > 0
- [ ] `test_is_market_impact_positive_on_adverse_fill` — fill worse than arrival → impact > 0
- [ ] `test_is_components_sum_to_total_is` — delay + impact + opportunity ≈ total IS
- [ ] `test_market_impact_increases_with_size` — larger trade → higher impact bps
- [ ] `test_signal_breakeven_adv_positive` — returns positive value for any valid IC
- [ ] `test_capacity_report_returns_binding_constraint` — result has "binding_constraint" key
- [ ] `test_intraday_trigger_spy_crash` — SPY −3.5% + VIX 35 → de-risk trigger fires
- [ ] `test_intraday_trigger_no_trigger_on_normal_day` — no extreme conditions → empty list
- [ ] `test_fill_quality_report_by_sleeve_present` — report has "by_sleeve" key with sleeve names

---

## Acceptance Criteria

1. TWAP executor routes all trades > 5% ADV correctly when `TWAP_ENABLED = True`
2. IS decomposition present in `logs/slippage_log.jsonl` for all fills after deployment
3. Capacity report runs weekly without error; result logged
4. Intraday triggers checked twice daily; at least one trigger successfully tested in paper mode
5. Fill quality report generated and included in weekly IC brief
6. All 16 tests passing; full suite passing
