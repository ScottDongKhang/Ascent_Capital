# Market Data & Alternative Data — Real-World Cost Grounding

> **Adversarially verified, with corrections.** (1) Bloomberg pricing is
> **tightened, not corrected**: a NeuGroup article quoting Bloomberg's own
> customer pricing letter ("linked to weighted global inflation," 6.5% increase
> for 2025) corroborates the $31,980/seat figure at a step up from third-party
> trackers alone. (2) S&P Capital IQ's wide range is **confirmed genuinely
> opaque, not a research gap** — Vendr's 55-purchase procurement dataset
> independently reproduces $14,800-$215,000/yr (median ~$53,000); treat that
> band, not the $327,500 outlier, as the best-attested figure. (3) The alt-data
> market-size discrepancy is **worse than originally reported**: Neudata's own
> materials cite *two different figures for the same year* — $2.8B (its "State
> of the Alternative Data Market 2026" blog) and $15.4B (a separate Neudata/PR
> Newswire release) — likely a directly-tracked-floor vs. modeled-total split
> that Neudata itself doesn't clearly label. The $2.8B figure (plus the
> Morgan-Stanley $1M-per-$1B-AUM and Neudata-survey $1.6M-average-buyer
> figures below) remains the more credible read for "what a fund actually
> spends," but cite the $15.4B figure too rather than treating $2.8B as
> Neudata's single number. (4) The AIMA "Casting the Net" 0.1%-of-AUM /
> 4.4%-of-fee-revenue figures **could not be re-confirmed** — the primary PDF
> was unreadable via automated fetch on a second attempt; secondary commentary
> only loosely corroborates "single-digit basis points of AUM," not the exact
> percentages. Treat those two figures as unresolved, not re-verified.

Every figure below is sourced; where no credible source could be found the gap is
stated explicitly rather than filled with an invented number. Most vendor pricing
is not publicly listed — figures are third-party trackers/aggregators (Vendr,
CostBench, Godel Discount) rather than vendor-published list price, and are
directionally reliable, not precise.

## 1. Core market data vendors — 2024-2026 list pricing

| Vendor | Reported annual cost per seat | Source confidence |
|---|---|---|
| **Bloomberg Terminal** | ~$31,980/yr single seat (2026); ~$28,320/yr at 2+ seats | Widely corroborated across independent pricing trackers, not a Bloomberg-published number |
| **LSEG Workspace** (formerly Refinitiv Eikon) | ~$22,000/yr, range $18,000-$30,000/yr depending on data bundle | Third-party aggregator (Vendr) |
| **FactSet** | ~$4,000-$50,000/yr depending on configuration; typical enterprise $15,000-$40,000/yr | Third-party (CostBench, Vendr); real-time exchange fees additive |
| **S&P Capital IQ Pro** | Wide disagreement: $14,800-$215,000/yr (median ~$53,000) vs. a separate small-sample estimate of ~$327,500/yr | High variance, low confidence — flagged, not resolved |

None of the big-four terminal vendors publish real list prices. Treat these as
directional ranges for budgeting, not quotes.

## 2. Alternative data market

- **Market size estimates disagree by an order of magnitude** depending on scope:
  Grand View Research ($18.8B in 2025, 37.6% CAGR toward $276.9B by 2033) vs.
  Neudata's buyer-side-only figure of **$2.8B** (2025, +27% YoY) — Neudata measures
  actual third-party dataset spend; the larger figures likely include broader
  data-technology/services categories.
- **Hedge funds are the largest buyer segment** — ~67.7% revenue share, ~78%
  alt-data penetration among hedge funds (Grand View Research).
- **Real spend figures** (Neudata 2025 survey, 60 buyers): Morgan Stanley
  benchmark of **~$1M per $1B AUM** in year-one alt-data spend; average hedge fund
  **~$1.6M/yr**; largest firms **$5M+/yr across ~43 vendors**. A separate Oppenheimer
  2025 survey reports large funds ($1B+ AUM) spending **$15-60M/yr** — likely total
  data/research budget, not alt-data-only; flagged as a discrepancy, not reconciled.
- **Named vendors** (industry-report-cited, not independently cost-verified):
  YipitData, Thinknum, Nasdaq Data Link (formerly Quandl), Advan Research,
  RavenPack, M Science, Similarweb, Eagle Alpha.
- **One concrete per-vendor data point**: YipitData enterprise contracts reportedly
  start high-six-figures annually (~$500K-$1M+), 6-month minimum terms (Ramp
  vendor-spend tracker — third-party estimate, not YipitData-published).
- **Gap, not filled**: no credible per-dataset pricing found for Advan, Thinknum,
  Similarweb, or Nasdaq Data Link at institutional/hedge-fund tiers.

## 3. Compute / infrastructure

No credible public figures exist for exact fund-level cloud spend by AUM tier —
funds treat this as confidential. The one sourced order-of-magnitude figure: a
cost-optimized stack (e.g., AWS Athena + Iceberg) is cited at **$10K-$50K/month**
for a smaller quant fund's data infrastructure (AWS industry blog — vendor-authored,
directional). No comparable figure exists for large multi-strategy platform infra
spend; not fabricated here.

## 4. Total research/data budget as % of AUM or fee revenue

- AIMA's "Casting the Net" report: hedge funds collectively spend an estimated
  ~$250M industry-wide on alt data, **~0.1% of aggregated AUM** or **~4.4% of
  management-fee revenue** (assuming 1.35-1.5% average management fee — AIMA's
  own benchmark study puts average management fee at 1.35%, only 5% of funds ≥2%).
- A separate, older AIMA/MFA figure: managers spend up to **10% of total operating
  costs** on the combined bucket of compliance + technology + back-office — a
  broader category than data/research alone.
- No Preqin/Eurekahedge survey with a systematic-fund-specific "data budget as %
  of AUM" figure was found — flagged as a gap.

## 5. Point-in-time / survivorship-bias-free data

Quant equity research specifically requires point-in-time fundamentals — not just
historical prices — because reported financials get restated after the fact, and
backtesting with today's restated numbers on a historical date introduces
look-ahead bias. Separately, using only currently-listed constituents for a
historical universe omits delisted/bankrupt/merged names (survivorship bias),
estimated in industry commentary to inflate backtested annual returns by roughly
1-2 percentage points, compounding over a multi-year backtest.

Standard commercial fix: **Compustat Point-in-Time** + the **CRSP/Compustat Merged
Database**, typically accessed via **WRDS** (Wharton Research Data Services).
Institutional WRDS access is reported at **$50K+/yr** baseline, with per-dataset
add-ons (e.g., ~$25,000/yr for 13F holdings data, per informal institutional
pricing threads — low confidence). This is the real cost delta over "just" a
terminal subscription: terminals give current/adjusted data; survivorship-bias-free
point-in-time fundamentals is a separate, additional line item.

## Sources

- Godel Discount, "Bloomberg Terminal Cost 2026" — godeldiscount.com
- CostBench Bloomberg/FactSet/S&P Capital IQ pricing calculators — costbench.com
- Vendr LSEG/Refinitiv and FactSet buyer guides — vendr.com
- WinApplications, "Refinitiv Eikon (LSEG Workspace) 2026 Enterprise Guide"
- GeminIQ, "S&P Capital IQ Pricing and Cost Explained"
- Grand View Research, "Alternative Data Market Size And Growth Report"
- Precedence Research / IMARC Group, alternative data market size reports
- Neudata, "AI Adoption Doubles as Alternative Data Budgets Surge" (2025 industry report)
- Hedgeweek / AltHub, 2025 alt-data budget survey coverage
- ResearchAndMarkets / Businesswire, "Alternative Data Market Global Outlook & Forecast 2024-2029"
- Ramp vendor-spend tracker, YipitData pricing estimate
- AWS Industries Blog, "GenAI in Factor Modeling Data Pipelines: A Hedge Fund Workflow on AWS"
- AIMA, "Casting the Net: How Hedge Funds are Using Alternative Data"
- AIMA, "Global Hedge Fund Benchmark Study" (April 2021)
- WRDS, "Linking CRSP with Compustat"
- EconJobRumors thread on WRDS institutional pricing (informal, low-confidence)

**Explicit gaps**: per-dataset pricing for Advan Research, Thinknum, Similarweb,
Nasdaq Data Link at institutional tiers; large multi-strategy platform compute
spend; a Preqin/Eurekahedge systematic-fund-specific data-budget-as-%-of-AUM
figure; commercial (non-WRDS) Compustat Point-in-Time list pricing.
