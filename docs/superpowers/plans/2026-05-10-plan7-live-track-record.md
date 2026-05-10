# Plan 7 — Live Track Record & Compliance Framework

> **For agentic workers:** Use superpowers:subagent-driven-development to execute task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the audit trail, compliance infrastructure, and investor-facing documentation required to present Ascent as a fundable system with a real track record. This plan has two distinct phases: (A) infrastructure — immutable audit trail, GIPS-compliant performance presentation, risk disclosure generation, methodology documentation; and (B) time — 12 months of live operation with real capital. Phase A is buildable now. Phase B requires waiting. Nothing in this plan changes how the system trades.

**What "YC-ready" actually requires:**
- A written strategy description that explains every alpha source, risk control, and execution decision at a level a technical investor can evaluate
- A 12-month live track record with auditable trade-by-trade records
- A clean compliance audit trail showing every decision was made by a documented process
- Real capital at risk (even a small amount — $5,000–$25,000 is enough to establish that you skin-in-the-game)
- A risk disclosure document that is legally accurate

**What it does NOT require:**
- SEC registration (required only if managing third-party money over $150M, or any amount in most states without an exemption — consult a lawyer before accepting outside capital)
- A prime brokerage relationship
- A CISO or compliance officer
- Audited financials (required only after raising outside capital)

**Prerequisites:** Plans 1–6 should be complete or in progress. The audit trail is most valuable when the full system is operating. However, the compliance infrastructure can be built independently and should be built now — every day of operation without an audit trail is a day of track record you cannot fully document.

---

## Files

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `compliance/audit_trail.py` | Immutable append-only decision log |
| Create | `compliance/performance_report.py` | GIPS-compliant performance presentation |
| Create | `compliance/risk_disclosure.py` | Risk disclosure document generator |
| Create | `compliance/methodology_index.py` | Machine-readable index of all strategy components |
| Create | `docs/methodology.md` | Human-readable strategy methodology document |
| Create | `docs/risk_disclosures.md` | Standard risk disclosures |
| Create | `docs/track_record_template.md` | Template for performance presentation to investors |
| Modify | `ascent/execution/eod_runner.py` | Wire audit trail into every decision point |
| Modify | `ascent/execution/order_engine.py` | Log every order decision with rationale |
| Modify | `debate/debate_runner.py` | Log full debate record to audit trail |
| Create | `scripts/verify_audit_trail.py` | Script to verify audit trail integrity |
| Create | `tests/test_compliance.py` | Full test suite — 12 tests |

---

## Task 1: Immutable Audit Trail

**File:** `compliance/audit_trail.py`

### Steps
- [ ] 1.1 Write `AuditTrail` class. Uses `logs/audit_trail.jsonl` as its backing store. Append-only — no delete, no overwrite methods. Every entry is signed with a SHA-256 hash of (previous_hash + entry_content) to form a hash chain. Tampering with any entry invalidates all subsequent entries.
- [ ] 1.2 Write `AuditTrail.record(event_type, payload) -> str`. `event_type` values: `"signal_generated"`, `"portfolio_constructed"`, `"debate_verdict"`, `"order_submitted"`, `"order_filled"`, `"order_cancelled"`, `"approval_granted"`, `"approval_denied"`, `"kill_switch_triggered"`, `"regime_changed"`, `"config_modified"`. `payload` is a serializable dict. Returns the entry hash.
- [ ] 1.3 Entry structure:
  ```json
  {
    "sequence_number": 1042,
    "timestamp": "2026-05-10T14:32:15.123456Z",
    "event_type": "order_submitted",
    "payload": {"symbol": "AAPL", "side": "buy", "shares": 10, "rationale": "trend+earnings alpha"},
    "prev_hash": "a3f9...",
    "entry_hash": "b7c2..."
  }
  ```
- [ ] 1.4 Wire into `eod_runner.py`: record `"signal_generated"` (alpha scores), `"portfolio_constructed"` (target weights + regime + alpha source weights), `"debate_verdict"` (full verdict dict), `"order_submitted"` per order, `"order_filled"` per fill.
- [ ] 1.5 Wire into `order_engine.py`: `"approval_granted"` / `"approval_denied"` for approval gate events.
- [ ] 1.6 Wire into `kill_switch.py`: `"kill_switch_triggered"` with drawdown amount and threshold.
- [ ] 1.7 Wire into `debate_runner.py`: `"debate_verdict"` with full transcript summary, all agent positions, judge rationale, and disagreement score.

---

## Task 2: Audit Trail Integrity Verification

**File:** `scripts/verify_audit_trail.py`

### Steps
- [ ] 2.1 Write a script that reads `logs/audit_trail.jsonl`, recomputes each entry's hash, and verifies the hash chain is intact. Reports: total entries, date range, any broken links.
- [ ] 2.2 Run automatically on the first day of each month as part of the `run_all_agents.py` monthly block. Log result to `logs/audit_integrity.jsonl`.
- [ ] 2.3 If integrity check fails: log CRITICAL error, send alert (via alert system from Plan 6), do NOT halt trading (chain breaks are informational — they indicate tampering or data corruption but the trade records themselves are still useful).

---

## Task 3: GIPS-Compliant Performance Presentation

**File:** `compliance/performance_report.py`

### Steps
- [ ] 3.1 Write `compute_gips_performance(start_date, end_date, agent_id="us_equities") -> dict`. Reads from TimescaleDB `nav_series` (Plan 6) or `logs/us_equities_pnl.jsonl` as fallback. Computes:
  - **Time-weighted return (TWR)**: chains daily returns. This is the GIPS standard — not money-weighted. Formula: `TWR = ∏(1 + r_t) - 1`.
  - **Annualized return**: `(1 + TWR)^(252/n_days) - 1`.
  - **Annualized volatility**: `std(daily_returns) × sqrt(252)`.
  - **Sharpe ratio**: `(annualized_return - risk_free_rate) / annualized_volatility`. Use the 3-month T-bill rate (from FRED `TB3MS`) as the risk-free rate.
  - **Max drawdown**: maximum peak-to-trough decline in NAV.
  - **Calmar ratio**: annualized return / max drawdown.
  - **Beta vs SPY**: OLS regression of daily returns on SPY daily returns.
  - **Alpha vs SPY**: annualized_return − beta × SPY_annualized_return (Jensen's alpha).
- [ ] 3.2 Write `format_gips_table(performance_dict) -> str`. Returns a Markdown table suitable for inclusion in the README, investor report, and YC application. Format: columns are period (1M, 3M, 6M, YTD, 1Y, SI), rows are return/vol/Sharpe/maxDD.
- [ ] 3.3 Write `generate_performance_presentation(output_path) -> Path`. Generates a one-page PDF (via Plan 6's weasyprint setup) with the GIPS performance table, a NAV chart (matplotlib), and the required GIPS compliance disclosures (see Task 5).
- [ ] 3.4 Update `README.md` walk-forward table automatically from `format_gips_table()` on each monthly report run. Live performance numbers should always be current in the README.

---

## Task 4: Risk Disclosure Generator

**File:** `compliance/risk_disclosure.py`

### Steps
- [ ] 4.1 Write `generate_risk_disclosures(strategy_params) -> str`. Takes current strategy parameters (max leverage = 1.0, asset classes, geographic exposure, liquidity profile) and generates standard risk disclosure language. This is NOT legal advice — it is a starting template that must be reviewed by a lawyer before sharing with any investor.
- [ ] 4.2 Standard disclosures to include: (a) past performance is not indicative of future results, (b) systematic strategies may fail without warning when market dynamics shift, (c) the strategy uses leverage [or: does not use leverage], (d) liquidity risk (positions may be illiquid in a market stress event), (e) model risk (all signals are derived from historical data which may not represent future conditions), (f) concentration risk (the portfolio holds [N] positions — loss of any single position has material impact).
- [ ] 4.3 Auto-generate monthly and include in investor report (Plan 6 Task 6). Save to `docs/risk_disclosures.md`.
- [ ] 4.4 Add `_get_current_risk_parameters() -> dict` that reads live system state to populate disclosure (number of positions, asset classes, max leverage, whether event trading is enabled, etc.).

---

## Task 5: Methodology Document

**File:** `docs/methodology.md`

This is a written document, not code. It must be written once and updated after any major strategy change. It is the document you hand to a due-diligence reviewer or YC partner.

### Sections to write
- [ ] 5.1 **Executive Summary** (1 page): what the strategy does, who runs it, when it was live, what the live track record is.
- [ ] 5.2 **Data Sources** (1–2 pages): for each data source (Yahoo Finance, FRED, EDGAR, Capitol Trades, Reddit, Google Trends), describe: what data is fetched, what lag is applied, what point-in-time constraints are enforced, and how failures are handled.
- [ ] 5.3 **Alpha Sources** (2–3 pages): for each alpha sleeve (trend, mean reversion, stat-arb, ML, volatility, fundamental, earnings, analyst, LLM fundamental, options flow, insider, short interest, altdata), describe: the academic/practitioner basis, how the signal is computed, what IC validation was performed, and what weight it receives.
- [ ] 5.4 **Regime System** (1 page): HMM states, features, transition dynamics, how regime affects sleeve weights and portfolio construction, how emergency refits work.
- [ ] 5.5 **Portfolio Construction** (1 page): MVO optimizer, factor constraints, Black-Litterman blending, sector constraints, max-weight caps, SPY 200MA overlay.
- [ ] 5.6 **Risk Management** (1 page): factor risk model, VaR/CVaR, kill switch, approval gate, large-trade market impact controls.
- [ ] 5.7 **Execution** (1 page): TWAP executor, implementation shortfall measurement, capacity model.
- [ ] 5.8 **Debate Layer** (0.5 pages): purpose, agents, advisory-only constraint, gate conditions.
- [ ] 5.9 **Self-Improvement** (0.5 pages): self-improve loop, factor discovery, gate conditions, human review requirement.
- [ ] 5.10 **Performance Attribution** (0.5 pages): how returns are decomposed (factor-explained vs. idiosyncratic), how IS is measured, how IC is tracked.
- [ ] 5.11 **Compliance & Audit** (0.5 pages): audit trail structure, hash chain integrity, monthly verification, record retention.

---

## Task 6: Methodology Index (Machine-Readable)

**File:** `compliance/methodology_index.py`

### Steps
- [ ] 6.1 Write `build_methodology_index() -> dict`. Returns a machine-readable registry of all strategy components: `{"alpha_sleeves": [...], "data_sources": [...], "risk_controls": [...], "execution_methods": [...]}`. Each entry has: `{"name", "file", "description", "ic_threshold_required", "current_weight", "last_validated_date"}`.
- [ ] 6.2 This index is used by the audit trail to record the exact system state (which sleeves were active, at what weights) at every rebalance. Future audit can reconstruct exactly what strategy configuration was live on any date.
- [ ] 6.3 Export to `dashboard/methodology_index.json` daily.

---

## Task 7: Real Capital Deployment

This is not a code task — it is an operational decision.

### Steps
- [ ] 7.1 **Transfer real capital to Alpaca live account.** Even $5,000–$25,000 is enough to establish a real track record. Paper trading does not count for investors.
- [ ] 7.2 **Disable paper mode.** Change `ALPACA_PAPER = True` to `ALPACA_PAPER = False` in `.env`. Verify all order size limits are appropriate for the new NAV.
- [ ] 7.3 **Set EVENT_TRADING_ENABLED = True** only after 30 days of paper-mode event trading with positive IC (per Plan 3 acceptance criteria).
- [ ] 7.4 **Operate for 12 months minimum.** The GIPS performance presentation requires at least 12 months of live track record for a 1Y return figure. 6 months is sufficient for a first seed raise if other metrics are strong.
- [ ] 7.5 **Monthly reporting.** Generate the investor report every month. Keep all reports. Do not modify historical reports — this is your audit record.
- [ ] 7.6 **Tax accounting.** Use a separate account (Alpaca does not commingle funds). Keep all trade records. Consult a CPA for how to report trading gains from algorithmic systems.

---

## Task 8: Tests

**File:** `tests/test_compliance.py` — 12 tests

- [ ] `test_audit_trail_appends_only` — write two entries; file has exactly two lines
- [ ] `test_audit_trail_hash_chain_valid` — hash of entry N matches prev_hash of entry N+1
- [ ] `test_audit_trail_tamper_detected` — manually modify entry → integrity check fails
- [ ] `test_audit_trail_records_order_submitted` — eod_runner integration → audit entry created
- [ ] `test_gips_twr_correct` — known daily returns → correct time-weighted return
- [ ] `test_gips_sharpe_correct` — known returns and vol → correct Sharpe
- [ ] `test_gips_max_drawdown_correct` — known NAV series → correct max drawdown
- [ ] `test_gips_table_contains_required_periods` — output has 1M, 3M, 6M, YTD, 1Y, SI columns
- [ ] `test_risk_disclosure_contains_required_language` — output contains "past performance" and "model risk"
- [ ] `test_methodology_index_has_all_sleeves` — index contains all active sleeve names
- [ ] `test_methodology_index_exported_to_dashboard` — `dashboard/methodology_index.json` created
- [ ] `test_audit_integrity_script_runs_clean` — clean audit trail → script exits 0

---

## The Timeline

| Milestone | Target | What it enables |
|-----------|--------|-----------------|
| Plans 1–3 complete | Month 3–4 | Factor model, MVO optimizer, event trading live |
| Plans 4–5 complete | Month 6–7 | Alt data validated, TWAP execution live |
| Plan 6 complete | Month 8–9 | Live dashboard, monthly reports |
| Plan 7 infrastructure complete | Month 6 | Audit trail running from this point forward |
| Real capital deployed | Month 6 | Track record clock starts |
| 6-month live track record | Month 12 | Sufficient for seed raise conversations |
| 12-month live track record | Month 18 | Full YC application strength |

---

## What the YC Application Looks Like

At month 18, you have:

- **12 months of auditable live returns** with positive Sharpe, clean audit trail, monthly investor reports
- **A system that reads SEC filings in real-time, processes congressional trades, and responds to options anomalies** — all documented in the methodology
- **An AI debate layer that acts as a circuit breaker**, preventing systematic errors from executing — uniquely defensible in a way that pure quant systems are not
- **A factor risk model** that decomposes exactly what bets the system is making at all times
- **A self-improving alpha discovery system** that continuously proposes new factors (gated on human review)
- **A live dashboard** you can show a YC partner in real time

The pitch: *"We built an AI-native systematic trading system that combines academic quant factors with real-time event intelligence (SEC filings, congressional trades, options anomalies) and an LLM-powered debate layer that prevents systematic errors. Here are 12 months of live audited returns. Here is the methodology. Here is the audit trail. We want to raise [X] to expand to institutional-quality position sizes."*

That pitch is fundable.

---

## Acceptance Criteria

1. Every rebalance produces an audit trail entry chain-linked to all previous entries
2. Monthly integrity verification runs and passes for 3+ consecutive months
3. GIPS performance table is accurate and auto-updates in README monthly
4. Methodology document written, reviewed, and version-controlled
5. Risk disclosures are accurate for the current strategy configuration
6. All 12 tests passing
7. Real capital deployed (even small amount); paper trading is supplementary, not primary
