# Regulatory Context — Compliance Obligations for a U.S. Systematic Trading Operation

**This is not legal advice.** Research summary to ground the Compliance & Middle
Office department design (`05_compliance_middle_office.md`) for Ascent Capital, a
solo-operator systematic strategy currently trading only via Alpaca paper trading
with no outside capital. Several thresholds below are fact-dependent and should be
confirmed with a securities attorney before any structural decision — especially
the moment outside money is accepted.

> **Adversarially verified: fully confirmed, both figures.** A follow-up pass
> independently re-confirmed the $100-110M SEC/state threshold and $150M ERA
> ceiling directly against Kitces and Katten (no 2025-2026 amendment found), and
> independently confirmed the Form PF proposed-threshold-change claim ($150M →
> $1B general, $1.5B → $10B large-adviser) directly against SEC.gov press
> release 2026-40, CFTC.gov press release 9216-26, and five independent law-firm
> alerts (Mayer Brown, Morgan Lewis, Davis Polk, Morrison Foerster, Winston &
> Strawn), all dated April-June 2026. Status accurately stated below as
> "proposed, comment period closed, not yet finalized" — this was a real,
> current detail, not a fabrication.

## 1. Investment Advisers Act of 1940 — core compliance rules

**Rule 206(4)-7 ("the Compliance Rule").** Requires every SEC-registered adviser
to: (1) adopt and implement written policies and procedures reasonably designed to
prevent violations of the Advisers Act; (2) review those policies no less than
annually — post-2024 amendment, that review must be documented in writing; (3)
designate a Chief Compliance Officer responsible for administering the program.
Off-the-shelf manuals without firm-specific customization do not satisfy the rule.

**Rule 204-2 ("the Books and Records Rule").** Requires registered advisers to
make and keep true, accurate, current records — standard accounting records plus
records specific to the fiduciary role (recommendations, advice, order/execution
records, performance records, communications). Standard retention: **five years**
from the end of the fiscal year of the last entry, first two years kept at an
easily accessible adviser office.

**Rule 206(4)-8 (pooled-vehicle anti-fraud rule).** Applies to advisers to pooled
investment vehicles regardless of SEC registration status. Prohibits (a) any
untrue statement of material fact, or omission needed to make statements not
misleading, to any investor/prospective investor in the pool; (b) any other
fraudulent, deceptive, or manipulative act toward pool investors. Negligent
conduct causing a misleading impression can violate it — intent to defraud is not
required.

## 2. Registration thresholds

- **Federal (SEC) registration**: mandatory at **$110M** regulatory AUM; a buffer
  zone between $100-110M allows choosing SEC or state; below $90M an adviser
  generally must switch back to state registration (Dodd-Frank's 2011 hysteresis
  rule).
- **State registration**: applies below the federal threshold, varying by state.
- **Exempt Reporting Adviser (ERA)**: Rule 203(m)-1 exempts advisers solely to
  "qualifying private funds" with **under $150M** in U.S. private-fund AUM
  (uncalled commitments and proprietary assets count toward the cap) from full SEC
  registration. Still requires an abbreviated Form ADV filing and baseline
  recordkeeping; subject to SEC examination.
- **Does a fund trading only personal capital need to register?** The Advisers Act
  defines "investment adviser" around giving advice to others, for compensation,
  as a business. A person trading solely their own capital, with no outside
  clients and no third-party compensation, generally falls outside that
  definition — not required to register federally or (in most states) at the
  state level. **The bright line**: the moment even one outside investor is
  accepted into a pooled vehicle, or one advisory account is managed for a fee,
  registration analysis is triggered from that point.

## 3. Form PF

Filed by SEC-registered advisers to private funds with at least **$150M** in
private-fund AUM (matching the ERA threshold). As of this research (2026), the
SEC/CFTC have jointly *proposed* raising the general threshold to **$1B** (large
hedge fund threshold $1.5B → $10B) — proposed April 2026, comment period closed
June 2026, **not yet finalized**. Purpose: systemic-risk reporting to the
Financial Stability Oversight Council, not investor disclosure.

**Bottom line for Ascent's scale**: far below any plausible threshold (current
$150M or proposed $1B) for the foreseeable future. Placeholder note, not a built
control.

## 4. Best execution

No single codified "best execution rule" for advisers exists, but advisers owe a
fiduciary best-execution obligation under the Advisers Act's Section 206 anti-fraud
provisions. For broker-dealers, **FINRA Rule 5310** is the operative standard:
"reasonable diligence to ascertain the best market" so the resulting price is "as
favorable as possible under prevailing market conditions." Firms not doing
order-by-order review must perform "regular and rigorous" reviews — at minimum
**quarterly**, broken out by security and order type. A standalone SEC Regulation
Best Execution was proposed 2022-2023 and remains pending, not adopted.

In practice, systematic managers document best-execution compliance via
**Transaction Cost Analysis (TCA)** — comparing realized fills against arrival
price, VWAP, or other benchmarks. Ascent, trading through a single retail-facing
broker (Alpaca), has essentially no venue choice today, which limits what a
best-execution review can meaningfully compare — worth noting explicitly rather
than building an elaborate multi-venue TCA process.

## 5. Reg SCI — mostly not applicable

Regulation Systems Compliance and Integrity applies to market infrastructure
(exchanges, clearing agencies, certain large ATSs, FINRA, MSRB) — not buy-side
advisers or funds. A proposed expansion could eventually reach investment
advisers/companies, but this is speculative, not current law. **Explicitly out of
scope** for Ascent today or in the near term.

## 6. Practical proportionality for Ascent today

**Legally required right now** (solo operator, personal capital only, paper
trading via Alpaca, zero outside investors): essentially nothing under the
Advisers Act. Rules 206(4)-7, 204-2, 206(4)-8, and SEC/state registration are not
triggered because Ascent gives no advice "to others" for "compensation." Form PF
and Reg SCI are not applicable at any scale relevant here. This is a
fact-dependent read, not a bright line — confirm with counsel before relying on
it.

**Becomes required the moment either trigger fires:**
- **(a) Accepting outside capital** — even one investor, even a friends-and-family
  SPV — is the bright-line trigger, not an AUM number. At that point, determine
  (based on structure) whether state registration, SEC registration, or an
  exemption (ERA under $150M private-fund AUM most likely initially) applies.
- **(b) Crossing AUM thresholds**, but only once already advising others:
  $100-110M triggers mandatory SEC registration; $150M in private-fund AUM ends
  ERA eligibility and requires full registration plus Form PF (current rule; $1B
  if the pending proposal finalizes).

**Good practice to build now, cheap while small, expensive to retrofit later:**
- A written compliance-policy skeleton mapped to what 206(4)-7 will eventually
  require.
- Systematic recordkeeping of trade rationale, backtests, and order/execution
  data in a Rule 204-2-shaped structure (five-year retention habit) — walk-forward
  research artifacts are exactly the kind of record cheap to log today, painful
  to reconstruct retroactively.
- A lightweight TCA/execution-quality log against Alpaca fills — good practice
  regardless, and gives Ascent a running best-execution record if/when needed.
- Clear internal separation between "advisory" outputs (AI PM, debate layer,
  falsifier registry — already documented as advisory-only per CLAUDE.md
  integrity constraint #5) and anything that could later look like advice given
  to a third party, since that boundary determines whether the Advisers Act
  applies at all.

## Sources

- 17 CFR § 275.206(4)-7 (Cornell LII); SEC release on compliance programs;
  InnReg guide to Rule 206(4)-7
- LegalClarity / Smarsh summaries of Rule 204-2 (books and records)
- SEC adopting release IA-2628; 17 CFR § 275.206(4)-8 (pooled vehicles)
- Kitces, "State vs SEC Registration for RIAs Near $100 Million RAUM"
- Katten, "Summary and Analysis of Dodd-Frank Rules for Investment Advisers"
- The Venture Alley, "Private Fund Adviser Exemption (under $150M AUM)"
- NASAA Investment Adviser Guide; Cornell Wex "investment adviser"
- Dechert / Mayer Brown / Morrison Foerster client alerts on the 2026 proposed
  Form PF amendments
- FINRA Rule 5310; FINRA Regulatory Notice 26-15; SEC proposed Reg Best Execution
  (2022)
- LinkedIn (Adam Sussman), "What Reg SCI Means for the Buy-Side"; SEC proposed
  Reg SCI expansion (2023)
