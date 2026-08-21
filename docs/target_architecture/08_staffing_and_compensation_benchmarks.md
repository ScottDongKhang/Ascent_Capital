# Real-World Quant Fund Staffing & Compensation — Grounded Reference

Sourced from recruiting/compensation-guide data and industry commentary — not
primary comp-survey-firm tables (Johnson Associates, Glocap, Options Group were
identified but their current reports were not directly retrievable; treat figures
below as recruiting-market aggregates, one tier below true survey-firm data).
Explicit gaps are stated rather than filled with invented numbers.

> **Adversarially verified, one gap partially closed.** CRO compensation was
> flagged as a real gap (general-industry figures only, not hedge-fund-specific).
> A follow-up pass found one real hedge-fund-specific anchor: Infovest21's 2013
> survey of 15 hedge funds with ≥$1B AUM found CRO total compensation averaging
> **$1.7M** — real data, but 13 years stale and a small sample. This suggests the
> general-industry $190-429K figures already in this document likely
> undercount large-fund CRO total comp once bonus is included; still no current
> (2024-2026) hedge-fund-specific figure was found despite targeted search of
> Hedge Fund Law Report, Options Group, and eFinancialCareers. SEC registration
> thresholds ($100-110M state/SEC choice, $150M ERA ceiling) were independently
> re-confirmed as unchanged since Dodd-Frank 2011, with one added nuance: a
> **$90M de-registration floor** on the way back down from SEC to state
> registration (hysteresis, not a flat $100M line both directions).

## 1. Department headcount ratios, by AUM tier

Hard headcount-per-AUM data is thin and mostly proprietary.

- **Emerging managers (sub-~$500M)**: Marex/AIMA's "Standing Strong: Emerging
  Manager Survey 2024" (respondents averaged $107M AUM) tracks headcount and
  breakeven cost structure but public summaries don't break it down function-by-
  function. **[Estimate, not sourced]**: industry commentary consistently frames
  sub-$100M-$250M managers as running with one dedicated compliance/ops hybrid
  person (often outsourced) and no dedicated risk officer — risk absorbed by the
  PM/COO.
- **Multi-manager platforms ($10B+)**: individual trading pods cap around **5-7
  people** (one PM owning P&L, a small number of analysts/researchers, execution
  support) — coordination overhead is said to exceed incremental alpha beyond
  that size. Millennium reportedly runs 330+ independent pods managing ~$79B AUM
  (late 2025). Both Millennium and Citadel are described as running centralized,
  independent risk teams that sit outside and above the pods — but no source
  gives a clean risk-staff-to-investment-staff ratio.
- **No authoritative survey** (AIMA, Preqin, HFR, EY, KPMG) with a clean "X risk
  staff per Y AUM" table was found in this pass. The AIMA/MFA/KPMG "Cost of
  Compliance" series is the closest lead but headcount ratios weren't retrievable.

## 2. Compensation benchmarks (systematic/quant funds)

| Role | Reported range | Confidence |
|---|---|---|
| **Quant Researcher** | National avg base ~$190K (NY ~$198K); top payers (Five Rings, Jane Street) ~$300K flat base; entry-level NY $125-150K base + 50-100% bonus ($200-300K total); mid-career total comp trending $500K+ | Aggregated recruiting-site/H1B data, directional |
| **Portfolio Manager** | Base $200K-$500K, bonus 200-500%+ of base, plus carry (1-3% of profits) at platforms with P&L-linked comp; total $1M-$10M+; top PMs at $5B+ funds cited $5-15M/yr | Standard "PM economics" narrative for multi-manager platforms |
| **Compliance Officer** (general) | Base $120K-$200K, bonus 20-40% of base | General compliance role, not systematic-fund-specific |
| **Dedicated full-time CCO** | ~$250K+/yr fully loaded (salary+benefits+bonus) | Outsourced-CCO marketing literature — directional, used as comparison baseline |
| **Chief Risk Officer** | General-industry avg ~$429K/yr (Glassdoor); hedge-fund-adjacent search-firm figure ~$217K avg, $190-216K typical base | Inconsistent, likely missing bonus/carry; **no systematic-fund-specific total-comp figure found** |
| **Execution Trader** | No credible systematic-fund-specific figure retrievable | **Gap — not fabricated.** Directionally lower comp/status than research/PM roles (alpha concentrates in signal, not execution) |

## 3. Three-lines-of-defense staffing in practice

- **CRO adoption is far from universal.** Hedge Fund Law Report cites survey data
  that only about **50% of institutional investors' managers have a dedicated
  CRO**. Where there's no dedicated CRO, the substantive risk function is
  performed by the COO, the CCO, or the PM — or not performed at all.
- **No sourced AUM threshold** for "needs a dedicated CRO" was found — treat any
  specific cutoff as an estimate.
- **At scale, risk is centralized and structurally separate** from the investment
  team (consistent with the three-lines model's intent), but no source gives a
  clean headcount ratio.
- **Practical pattern for small/emerging managers**: a collapsed second line —
  risk and compliance combined into one person or one outsourced relationship,
  not run as separate CRO/CCO desks, until the fund is large enough (unquantified
  in sources) to afford separation.

## 4. Minimum viable compliance function for a small/emerging manager

- **Rule 206(4)-7** requires a registered adviser to designate one supervised
  person as CCO — mandatory regardless of firm size once SEC-registered, but the
  rule allows the *function* to be outsourced, not the *designation*.
- **Practitioner-documented CCO staffing options for small managers**: (a) hire a
  dedicated internal CCO, (b) fold CCO duties onto an existing person (GC, COO,
  CFO — the most common small-manager pattern), or (c) outsource to a third-party
  compliance consulting firm.
- **Registration thresholds**: under $100M AUM → generally state registration
  (some states like NY require SEC registration from $25-100M); $100-110M → may
  choose SEC or state; above $110M → SEC registration generally mandatory. The
  **private-fund-adviser exemption** (Exempt Reporting Adviser status) applies
  under $150M AUM for advisers managing only private funds — abbreviated Form ADV,
  no full registration, no Rule 206(4)-7 written-policy mandate, no CCO-designation
  requirement.
- **Practical floor for a solo/personal-capital operator** (Ascent Capital's
  current state): if trading only personal/seed capital with no outside investors,
  the manager likely isn't an "investment adviser" under the Act at all (no
  compensation for advising others), and none of this applies. The moment outside
  capital is raised, the ERA private-fund exemption (sub-$150M, private funds
  only) is the realistic minimum-viable path.

## Sources

- Marex/AIMA, "Standing Strong: Emerging Manager Survey 2024"
- AIMA/MFA/KPMG, "The Cost of Compliance" survey series
- Navnoor Bawa Research, "How Millennium, Citadel & Point72 Structure Pods"
- "Young and Calculated" (Substack), quant firm org-structure analyses
- Wall Street Careers, "Hedge Fund Job Salary Guide 2026"
- Myntbit, "Quant Hedge Fund Compensation Guide 2026"
- Hedge Fund Law Report, "What Is a Chief Risk Officer, and Should Hedge Fund
  Managers Have One?"
- Hedge Fund Law Report, CCO designation and compensation article (paywalled
  beyond abstract)
- Wealth & Finance International, "Evaluating 5 Top Outsourced CCO Services for
  Hedge Funds"
- ZipRecruiter / Glassdoor, Chief Risk Officer salary data (2026 snapshots)
- SEC, Investment Advisers Act Rule 206(4)-7; Cornell LII 17 CFR § 275.204-2
- Katten, "Guide to Exemptions from Investment Adviser Registration"
- Day Pitney, Dodd-Frank state-vs-SEC registration threshold summary
- Proskauer, "Hedge Start: When Is SEC Registration Necessary?"

**Explicit gaps**: function-by-function headcount ratios normalized to AUM from
AIMA/Preqin/HFR/EY/KPMG; named Johnson Associates/Glocap/Options Group current-year
comp tables; a systematic-fund-specific execution-trader comp range; a sourced AUM
threshold for "needs a dedicated CRO."
