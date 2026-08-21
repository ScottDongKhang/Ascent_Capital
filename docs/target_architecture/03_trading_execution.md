# Trading & Execution Department — Target Architecture

## Layer 1 — Department Mandate

**Purpose**: translate risk-approved target weights (produced upstream by alpha →
portfolio → `us_equities` agent → orchestrator merge) into filled orders at Alpaca
at the best achievable price/cost, without ever exceeding risk-approved size, and to
reconcile the resulting broker state against the internal book every day.

**Authority boundaries** — this department does **not** decide *what* to buy. It
has zero write access to `target_weights`. It can only:
- reject, delay, or resize an order on cost/compliance/liquidity grounds (never on
  alpha grounds)
- choose *how* (market vs. TWAP) and *when* (now vs. sliced) to execute an
  already-approved delta
- halt submission of the *entire* order batch (kill switch, reconciliation break) —
  never selectively override the target

This mirrors integrity constraint #5 in CLAUDE.md: like the debate judge,
earned-authority blend, and falsifier trim, execution is advisory/mechanical, not
alpha-generating. If a role here ever gains authority to change *what* is held, it
becomes a fourth judgment layer and must go through the same advisory-only
discipline.

**Reports**: a daily Trading Desk Report (fills, slippage, TCA, reconciliation
status) attached to the existing rebalance recap in CLAUDE.md; an immediate
escalation (not batched) on any halt condition — kill switch trip, compliance
rejection of >X% of NAV, or a reconciliation break — to the same channel that
currently receives `kill_switch_triggered` audit events.

## Layer 2 — Roles

| Role | Function |
|---|---|
| **Order Manager (OM)** | Diffs target vs. current weights into an order list, batches, sequences (sells before buys), assigns an `order_id`, owns the state machine from "proposed" to "closed." |
| **Pre-Trade Compliance Checker (PTCC)** | Final gate before any order reaches the broker: restricted list, position limits, buying power, sector/name concentration re-check against *actual* broker state (not the stale weights snapshot). |
| **Execution Trader / Algo Selector (ETAS)** | Chooses market vs. TWAP (or, later, VWAP/POV) per order based on size relative to ADV and urgency; owns the TWAP scheduler. |
| **Transaction Cost Analyst (TCA)** | Post-trade: measures slippage vs. arrival price and vs. a VWAP benchmark, flags outliers, feeds the cost model's calibration. |
| **Position Reconciliation (PR)** | End-of-day (and pre-open) comparison of Alpaca (broker of record) vs. the internal book; halts next-day submission on unresolved breaks. |

## Layer 3 — Per-Role Responsibilities and Decision Logic

**Order Manager**
- Consumes the diff (`weight_delta` per symbol) already computed by
  `compute_orders()`; nothing new needed here except making its output an explicit
  `Order` object with a `state` field (`proposed -> compliance_checked -> routed ->
  submitted -> filled/partial/rejected -> reconciled`).
- Sequencing rule (already implicit in `order_engine.py:102`, should be made a
  named policy): **all sells execute before any buys**, to avoid buying-power
  shortfalls that force a mid-batch compliance rejection.
- Batches orders below `MIN_TRADE_THRESHOLD` (0.5% NAV, `order_engine.py:21`) into
  "hold" — unchanged.
- New: an order that PTCC rejects is **not silently dropped** — it is re-queued
  with a `deferred_reason` and retried on the next rebalance, distinct from
  `skipped_no_price` (price-unavailable, already exists at `order_engine.py:76-84`).

**Pre-Trade Compliance Checker** — the layer that does not currently exist as a
distinct gate. Today `kill_switch.check()` is the *only* pre-submission check, and
it is portfolio-level drawdown only — it has no concept of a restricted list, a
single-name limit, or buying power. Concretely, PTCC evaluates each order against:
- **Restricted list**: symbol in a `restricted_symbols.json` (earnings blackout,
  legal hold, wash-sale conflict with a recent tax-loss sale) → reject with reason
  `restricted`.
- **Position limit**: post-trade weight for any symbol must respect the existing
  `_water_fill_cap()` output — PTCC re-checks it against *live* broker positions
  (`get_positions()`), not the stale `target_weights` snapshot the pipeline computed
  hours earlier, because prices/positions can have moved since. Breach → reject, do
  not resize (resizing is Order Manager/ETAS's job on retry, not PTCC's).
- **Buying power**: sum of pending buy dollar amounts must not exceed
  `get_account()["buying_power"]`. If short, PTCC rejects the smallest-conviction
  buys first (lowest `abs(weight_delta)`) until the batch fits — an explicit,
  auditable tie-break rule, not "give up."
- **Large-order approval**: the existing `LARGE_TRADE_THRESHOLD_PCT = 2.0`
  (`eod_runner.py:48`) constant is currently defined but never actually gates
  anything in the code — no call site enforces it. PTCC becomes that enforcement
  point: any single order > 2% NAV requires a synchronous approval step (today: log
  + wait for `--approve` flag in dry-run mode; later: a human-in-the-loop
  Slack/webhook ack) before routing.
- Output is **approve/reject with a reason string per order**, never a silent
  pass-through.

**Execution Trader / Algo Selector**
- Reuses `should_use_twap()` (`twap_executor.py:32-40`): route to TWAP when
  `trade_size_dollars / adv_dollars > 0.05` (5% of 21-day ADV, computed from
  `dollar_vol_21d`). This logic already exists in `order_engine.py:112-141` but is
  currently gated behind `TWAP_ENABLED = False` — per CLAUDE.md's kill-switch list,
  it stays off pending validation, but the *routing decision logic* should run and
  log its counterfactual (`"would have TWAP'd $X"`) even while disabled, exactly as
  `execute_twap()` already does when off (`twap_executor.py:136-142`).
- Window sizing: `compute_twap_window()`'s Almgren-Chriss approximation
  (`twap_executor.py:43-74`), bounded 5-90 minutes by urgency — unchanged, just
  needs a real urgency input from the Order Manager (today it defaults to
  `"normal"`).
- New decision: orders ≤ 5% ADV go straight to market order via
  `submit_order()`; orders > 5% ADV but < some hard ceiling (e.g., 15% ADV) go
  TWAP; orders > 15% ADV are rejected back to PTCC/OM as "too large to execute
  safely in one session" — split across multiple rebalance days instead. This
  ceiling does not exist today and should.

**Transaction Cost Analyst** — does not exist today as a formal role;
`slippage_tracker.py` exists but only computes signal-price-vs-execution-price
after a 30-second sleep (`eod_runner.py:372-378`). TCA formalizes this:
- **Arrival-price slippage**: (fill price - price at order-submission time) /
  arrival price, in bps, per order.
- **Threshold**: slippage > 50 bps on a single order (or > 25 bps aggregate
  NAV-weighted across the batch) triggers a `tca_flag` reviewed before the *next*
  rebalance — not blocking the current one (TCA is inherently post-trade).
- **Benchmark**: also compute realized cost vs. a same-day VWAP proxy (from
  `dollar_volume`/`volume` already in the price cache) where available, to
  distinguish "the market moved against us" from "we executed badly."
- Feeds `cost_model.py`'s `CostModelParams` — repeated flags on the same symbol
  should raise its estimated impact coefficient for future sizing, closing the loop
  between measured and predicted cost.

**Position Reconciliation** — does not exist today; nothing compares Alpaca
positions to an internal ledger. Concretely:
- After EOD fills settle, pull `get_positions()` from Alpaca and diff against the
  internal fill ledger (sum of every `order_submitted` audit event since the last
  clean reconciliation).
- **Threshold**: discrepancy beyond **$250 or 1 share** (whichever is looser, to
  tolerate fractional-share rounding already handled by `close_position()`) on any
  single symbol, or **$1,000 aggregate**, halts next-day order submission (a new
  gate parallel to `kill_switch.check()`, tripping the same halt path) until a
  human resolves it. This is deliberately stricter than the kill switch's
  portfolio-level 8%/15% drawdown — reconciliation breaks indicate a
  bookkeeping/systems fault, not a market outcome, and should never self-heal by
  waiting.
- State persisted in `logs/reconciliation_state.json`, mirroring
  `kill_switch_state.json`'s pattern (`tripped`, `tripped_at`, `discrepancies`,
  `reset_at`), with the same manual-reset discipline.

## Layer 4 — Interfaces / Data Contracts

| Producer -> Consumer | Payload |
|---|---|
| Portfolio/orchestrator -> **Order Manager** | `target_weights: pd.Series[symbol->weight]`, already risk-approved (post `_water_fill_cap`, post sector constraint). OM must not re-derive it. |
| **Order Manager** -> `compute_orders()` | current Alpaca positions (`get_positions()`), `portfolio_value` (`get_portfolio_value()`), `min_threshold`. Emits `List[Order]` + `diff_df` (unchanged, `order_engine.py:35-41`). |
| **Order Manager** -> **PTCC** | one `Order` at a time, plus live `get_account()` (buying power), live `get_positions()`, and the restricted-list file. Emits **per-order** `{order_id, decision: approve\|reject, reason: str\|None}`. |
| **PTCC** -> **ETAS** | approved orders only, each carrying `dollar_amount`, `adv_dollars` (from `extract_cost_features`), and an `urgency` field (new -- default `"normal"`, `"high"` on kill-switch-adjacent or event-driven trades). |
| **ETAS** -> `alpaca_broker.submit_order()` / `twap_executor.execute_twap()` | `symbol, qty, side, order_type`. Emits `{status, order_id, fill_price\|None}` per order (or per TWAP slice). |
| Fills -> **TCA** | `{symbol, side, qty, fill_price, arrival_price, submitted_at, filled_at}` from the audit trail (`compliance.audit_trail.record_event`, already called at `eod_runner.py:352-356`) joined against `signal_prices` (`price_df.iloc[-1]`, already built at `eod_runner.py:377`). Emits `{symbol, slippage_bps, vwap_slippage_bps, flag: bool}` appended to a new `logs/tca_log.jsonl`. |
| Broker (Alpaca) + internal ledger -> **Position Reconciliation** | `get_positions()` (broker of record) vs. cumulative `order_submitted` audit events since last clean check. Emits `{status: clean\|break, discrepancies: [{symbol, broker_qty, internal_qty, delta_usd}], halt: bool}` to `logs/reconciliation_state.json`, read by `run_eod_with_weights()` at the same point it currently reads `kill_switch` state. |

## Layer 5 — Concrete Implementation Mapping

**Current call sequence** (both `run_eod()` and `run_eod_with_weights()`, which are
near-duplicate paths — `eod_runner.py:103` and `:766`):

```
run_pipeline() -> target_weights
  -> compute_orders()                          [order_engine.py:35]
  -> kill_switch.check()                       [kill_switch.py:130]  <- only pre-trade gate today
  -> cancel_all_orders()
  -> for order in orders: submit_order()       [alpaca_broker.py:118]
  -> slippage_tracker.track_slippage()         [post-hoc, after 30s sleep]
```

**Target call sequence** — new modules inserted between `compute_orders()` and
`submit_order()`:

```
run_pipeline() -> target_weights
  -> compute_orders()                                    [existing, unchanged]
  -> ascent/execution/reconciliation.py::check()          [NEW] -- halts here if prior-day break unresolved
  -> kill_switch.check()                                  [existing, unchanged -- portfolio-level]
  -> ascent/execution/compliance_gate.py::check_batch()    [NEW] -- per-order approve/reject:
        restricted list, position limits vs live positions,
        buying power, LARGE_TRADE_THRESHOLD_PCT approval
  -> ascent/execution/order_manager.py::sequence()         [NEW] -- sells-before-buys, re-queue rejects
  -> ascent/execution/algo_selector.py::route()            [NEW, wraps existing should_use_twap/execute_twap]
        market -> alpaca_broker.submit_order()
        TWAP   -> twap_executor.execute_twap()             [existing, still gated by TWAP_ENABLED]
  -> ascent/execution/tca.py::analyze()                     [NEW, replaces ad-hoc slippage_tracker call]
        writes logs/tca_log.jsonl, flags > 50bps
  -> ascent/execution/reconciliation.py::end_of_day()       [NEW] -- writes reconciliation_state.json for next run
```

**What already exists and is reused as-is**: `kill_switch.py` (portfolio drawdown,
unchanged — it stays the *last-resort* circuit breaker, not the compliance gate),
`order_engine.compute_orders()` (diffing), `twap_executor.py` (TWAP mechanics,
still behind `TWAP_ENABLED=False`), `cost_model.py` (`extract_cost_features`,
`apply_cost_filter` — becomes ETAS's ADV/impact input), `alpaca_broker.py`
(unchanged — it is the broker adapter, not a decision layer), `slippage_tracker.py`
(subsumed into the new `tca.py` rather than duplicated).

**What is net-new**: `compliance_gate.py` (the actual pre-trade compliance role —
today's gap, since `LARGE_TRADE_THRESHOLD_PCT` is a dead constant with no
enforcement site, and there is no restricted-list or buying-power check anywhere),
`reconciliation.py` (does not exist at all today — no code compares Alpaca
positions to an internal ledger), `order_manager.py` (currently the
sequencing/state-machine logic is inlined and duplicated across `run_eod()` and
`run_eod_with_weights()`; consolidating it also removes that duplication),
`algo_selector.py` (currently the TWAP-routing decision lives inline inside
`order_engine.py:104-141`; pulling it out makes it independently testable and
gives it room for a future VWAP/POV algo without further bloating
`order_engine.py`), `tca.py` (formalizes and extends `slippage_tracker.py` with an
explicit flag/threshold and a VWAP benchmark it doesn't currently compute).

**Two duplicate code paths should collapse into one.** `run_eod()`
(`eod_runner.py:103`) and `run_eod_with_weights()` (`:766`) independently
reimplement compliance-adjacent logic (kill switch call, order loop,
`close_position` vs `submit_order` branching) with subtly different behavior — e.g.
`run_eod_with_weights` swallows a non-`KillSwitchTriggered` exception and continues
(`:1020-1021`), while `run_eod` does not have that branch at all. Once
`order_manager.py` / `compliance_gate.py` exist, both entrypoints should call the
same shared pipeline object so gate logic can't silently diverge between the
daily-run path and the multi-agent-merge path.
