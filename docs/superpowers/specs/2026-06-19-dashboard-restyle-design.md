# Dashboard Restyle — Editorial "Quiet Professional" Direction

**Date:** 2026-06-19
**Owner:** Scott
**Target:** `scripts/generate_performance_page.py` → `docs/index.html` (GitHub Pages)
**Status:** Design approved (6 visual mockups, `mockups/01`–`06`). Ready to plan + implement.

---

## 1. Goal

Full visual restyle of the public dashboard into a single coherent **dark-editorial** identity — serif display, warm-black paper, hairline rules, one gold accent. The page should **show what the system is by displaying its actual artifacts** (numbers, the real debate, the real book, the construction pipeline) rather than describing itself with marketing copy. Attention through craft, not noise. Quiet, professional, never "AI-slop."

Two capabilities the current page lacks, added here:
1. **Reasoning on demand** — every holding and the verdict expand to show the *why*, qualitative only.
2. **"How the book is built"** — the construction pipeline (data → live order) made visible, with the proprietary model sealed.

Non-goal: changing any trading/portfolio logic. This is presentation only. The daily regeneration hook (`main()` writing `docs/index.html` + README stats) is preserved.

---

## 2. Hard constraints

- **Confidentiality (the edge).** The HTML must NEVER emit: sleeve identities-to-weights mapping, signal/indicator parameters, regime transition thresholds, tilt strengths, blend weights, or any tunable. Public guardrails that are inferable anyway (max 10%/name, 1/sector, cluster cap 20%, 3-state HMM, kill-switch 15%, >2% NAV approval) MAY be shown. When in doubt: show the *what* and the *guardrail*, seal the *how*.
- **Truthfulness.** Show real data including unflattering numbers (portfolio currently trails SPY; alpha is negative). No invented metrics. Reasoning text must derive from real sources (verdict, prethesis, alpha breakdown) or be a clearly generic honest fallback — never fabricated specifics.
- **Graceful degradation.** Every data source can be missing. A missing file → that section/element renders an honest empty state, never a crash. Mirrors existing `main()` try/except discipline.
- **Accessibility / motion.** All animation respects `prefers-reduced-motion`. No motion is required to read the page (JS-off / reduced-motion shows final state).
- **Single file output.** Self-contained `docs/index.html` (inline CSS/JS), as today. External deps limited to existing CDN (Chart.js) + Google Fonts.

---

## 3. Visual system (design tokens)

```
bg #0c0b0a   ink #f3f0e9   text #b9b3a7   muted #857e70   faint #6a6457
rule #262219 (hairline)    rule2 #1b180f (fainter)
gold #cba569  gold-dark #a8854c   up #6aa97f   down #c47b6e   seal #8a7f6d
```

- **Serif (display/headings/big numbers):** Source Serif 4.
- **Mono (labels, data, eyebrows, chips):** IBM Plex Mono.
- **Sans (body):** Inter.
- Layout: max-width 1080px, 40px gutters, hairline section rules, generous vertical rhythm.

---

## 4. Page structure (top → bottom)

| # | Section | Source | New? |
|---|---------|--------|------|
| 1 | Masthead | static + `generated_at` | restyle |
| 2 | Lede (NAV + figures) | `compute_stats` | restyle |
| 3 | Equity curve | `build_chart_data` (Chart.js, re-themed) | restyle |
| 4 | Secondary stats strip | `compute_stats` | restyle |
| 5 | **How the book is built** (funnel + 7-stage pipeline) | regime_signal, earned_authority, latest verdict, universe count | **NEW** |
| 6 | The latest verdict (debate) | `load_verdicts` | restyle + enrich |
| 7 | The AI desk (earned authority, allocation, counterfactual, scorecard) | existing AI PM loaders | restyle |
| 8 | The book (holdings) | `fetch_current_positions` + new reasoning/sparkline | restyle + enrich |
| 9 | Event timeline | `load_verdicts` | restyle, optional fold |
| 10 | Footer | static | restyle |

Mockup `05-rich.html` = sections 1–4 + 6 + 8. Mockup `06-construction.html` = section 5. The AI desk (7) keeps its existing real sub-widgets (authority ladder, allocation donut, four-track counterfactual, override scorecard) re-skinned to the editorial tokens.

---

## 5. Section detail + data wiring

### 5.2 Lede & 5.4 stats — fix the live SPY/alpha `nan` bug
Current `index.html` renders `alpha-ctr` and `spy-ctr` as `nan%`. Root cause lives in `fetch_spy` / `compute_stats` (SPY series has a trailing `None`/NaN bar; arithmetic propagates NaN). **In scope:** make `compute_stats` compute `spy_return`/`alpha` from the last *non-NaN* SPY value aligned to the same window as portfolio (forward-fill or drop trailing NaN before the ratio). The restyle puts these numbers in the hero, so they must be correct. Count-up animation on NAV + return.

### 5.3 Equity curve
Keep Chart.js (retains hover, verdict annotations, regime bands — all real and valuable). Re-theme: gold portfolio line, dashed muted SPY, hairline grid, mono ticks, no glow/gradient-fill beyond a very faint gold area. Draw-on-load via Chart.js animation (respect reduced-motion). Honest caption retained (trailing SPY, Sharpe-unreliable, WF OOS 0.52).

### 5.5 How the book is built (NEW)
- **Funnel strip:** `Universe → Scored → Candidates → Constructed → Held`. Counts: `Universe` from universe config size (rounded to ~nearest 50 — rounding is mild edge-protection); `Held` = `len(positions)`; `Scored`/`Candidates` are rounded representative constants (documented as approximate in code comment). Bars fill on scroll-in.
- **7 stages**, each `{index, name, visible|sealed, one-line what-it-does, tail (throughput or sealed chip), expandable detail with public chips + a `▤ sealed` note}`:
  1. Universe & Data — *visible*
  2. Signals — *sealed* (sleeve identities/weights/params)
  3. Conviction Ranking — *sealed* (blend weights)
  4. Construction — *mixed*: public caps as chips, **tilt strengths sealed**
  5. Regime Overlay — *mixed*: current state shown (from `regime_signal.json`), **thresholds sealed**
  6. Adversarial Gate — *visible*, shows latest verdict recommendation+confidence, links to §6
  7. AI PM & Execution — *visible*, shows authority level/budget from `earned_authority.json`, caps, kill switch
- Live values injected: current regime, latest verdict rec+conf, authority level/title/budget, held count. Everything else is static copy. Closing pull-quote.
- Implement as a data-driven list (Python list of stage dicts → HTML) so copy lives in one place and is easy to audit for confidentiality.

### 5.6 The latest verdict
From newest `verdict_*.json` (`load_verdicts()[−1]`). Render: verdict badge (proceed/reduce/halt → green/gold/red), **conviction meter** (fills to `verdict.confidence`), four agent voices (bull/bear/devils_advocate/regime_specialist) — first paragraph shown, Round-2 rebuttal behind one shared "Read the round-two rebuttals" expander. **Judge synthesis** block (gold left-rule) from verdict rationale/summary (fallback: compose from recommendation + confidence). **Key risks** numbered list from `verdict.key_risks`. Truncate agent text to ~1–2 sentences for the closed state (full text already lives in the debate JSON; we excerpt).

### 5.8 The book
- All positions from `fetch_current_positions()`, sorted by weight desc.
- **Allocation ribbon:** one segment per holding, `flex: weight`; segment highlights when its row opens.
- **Per-row sparkline:** 30-session close series per symbol from `prices_live.parquet` (`df[df.symbol==s].sort_values('date').close.tail(30)`), rendered as inline SVG path (green if last≥first else red). Build once: group the parquet, map symbol→path. Missing symbol → omit sparkline (render blank cell), no crash.
- **Expand → reasoning**, two columns + redaction strip:
  - *Why it's here* — `_position_reasoning(sym)`: prefer prethesis `conviction_reasons`/`high_conviction_names` match; else a generic honest line ("Held on the composite cross-sectional ranking").
  - *What the committee said* — scan latest verdict `key_risks` + agent arguments for the symbol; if mentioned, surface as a flag (red); else "No adversarial flag this cycle."
  - *Redaction strip* — `{label} ▦▦▦▦ ▤ Sealed`, label chosen deterministically per symbol from a fixed rotation (Sleeve attribution / Entry signal / Risk weighting / Cluster cap / Regime threshold / Correlation guard / Exposure rule). Purely decorative-honest: signals a sealed mechanism exists; encodes nothing.
- Footnote: "Signal weights, sleeve attribution, and regime thresholds are sealed by design."

### 5.7 The AI desk
Re-skin existing helpers (`_earned_authority_html`, allocation donut, `_counterfactual_chart_html`, `_override_scorecard_html`, `_thesis_html`, `_promotion_gates_html`) to editorial tokens. No data changes. Authority ladder → editorial "rung" styling from mockup 03.

---

## 6. Interactions (all reduced-motion-safe)

| Interaction | Mechanism |
|---|---|
| Expand holding row / stage / debate | CSS `grid-template-rows: 0fr→1fr` transition (smooth auto-height), caret rotate |
| Count-up (NAV, return) | rAF easing, final value if reduced-motion |
| Equity line draw | Chart.js animation (once) |
| Weight bars / conviction meter / funnel bars fill | width transition triggered on IntersectionObserver enter |
| Section fade-up on scroll | `.rev` + IntersectionObserver |

No looping/bouncing/parallax. One-shot, purposeful.

---

## 7. Implementation approach

- Rewrite `build_html()` and its section helpers; keep the `main()` orchestration, all `load_*`/`fetch_*` functions, README updater, and the CLI/daily-hook contract intact.
- Add: `_sparkline_paths(symbols) -> dict[str,str]` (reads prices_live once), `_position_reasoning(sym, verdict, prethesis) -> (why, committee_html, flagged)`, `_construction_section_html(regime, verdict, authority, n_held, universe_n)`, `_redaction_label(sym)`.
- Fix `compute_stats` SPY/alpha NaN handling.
- All new copy strings centralized (stage list, redaction labels) for one-pass confidentiality audit.
- Keep Chart.js; restyle via its options. Sparklines = hand-built inline SVG (no per-row chart instances — performance).

## 8. Testing / verification

- Regenerate locally, open in headless Chrome (existing pattern) — assert: zero JS console errors, all sections present, expanders toggle, reduced-motion renders final state.
- **Confidentiality grep gate:** after generation, assert the HTML does NOT contain any sleeve-weight/parameter tokens (e.g. `DEFAULT_ALPHA_WEIGHTS` values, known param names). Add a small check in the test that the rendered page contains none of a denylist of sensitive strings.
- Verify SPY/alpha now numeric (not `nan%`).
- Existing generator tests still pass; add tests for the three new pure helpers (`_position_reasoning` fallback, `_sparkline_paths` missing-symbol, `_construction_section_html` with missing data).
- Mobile + desktop visual check (headless screenshots).

## 9. Out of scope
Trading logic, new metrics/data collection, light-mode toggle, the `methodology.md`/`risk_disclosures.md` sub-pages (restyle later if desired).
