# Dashboard Editorial Restyle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle `docs/index.html` (via `scripts/generate_performance_page.py`) into the dark-editorial identity with a new "How the book is built" pipeline section, on-demand qualitative reasoning, and the redaction motif — showing the system's real artifacts while sealing the model.

**Architecture:** Rewrite `build_html()` and its section helpers; preserve `main()`, all `load_*`/`fetch_*` data functions, the README updater, and the daily-regeneration contract. Add four pure helpers (`_sparkline_paths`, `_redaction_label`, `_position_reasoning`, `_construction_section_html`) and fix the SPY/alpha NaN bug. Visual source of truth = `mockups/05-rich.html` (sections) and `mockups/06-construction.html` (pipeline) on disk.

**Tech Stack:** Python 3.12 (`.venv/bin/python`), stdlib + pandas + requests, Chart.js 4.4 (CDN, retained), Google Fonts (Source Serif 4 / IBM Plex Mono / Inter). Tests: pytest.

## Global Constraints

- Run Python via `.venv/bin/python` only.
- **Confidentiality (verbatim):** HTML must NEVER emit sleeve identities-to-weights mapping, signal/indicator parameters, regime transition thresholds, tilt strengths, blend weights, or any tunable. Public guardrails MAY be shown: `max 10%/name`, `1 per sector`, `cluster cap 20%`, `3-state HMM`, `kill-switch 15%`, `>2% NAV approval`.
- Truthfulness: show real data incl. unflattering numbers; reasoning text derives from real sources or a generic honest fallback — never fabricated specifics.
- Graceful degradation: every data source may be missing → honest empty state, never a crash.
- All animation respects `prefers-reduced-motion`; page is fully readable with JS off / reduced motion (final state shown).
- Output is a single self-contained `docs/index.html` (inline CSS/JS); external deps limited to existing Chart.js CDN + Google Fonts.
- Design tokens: `bg #0c0b0a · ink #f3f0e9 · text #b9b3a7 · muted #857e70 · faint #6a6457 · rule #262219 · rule2 #1b180f · gold #cba569 · gold-dark #a8854c · up #6aa97f · down #c47b6e · seal #8a7f6d`. Fonts: Source Serif 4 (display), IBM Plex Mono (labels/data), Inter (body).

---

### Task 1: Fix SPY/alpha NaN in `compute_stats`

**Files:**
- Modify: `scripts/generate_performance_page.py:455-458`
- Test: `tests/test_generate_performance_page.py` (create if absent)

**Interfaces:**
- Produces: `compute_stats(records, spy)` returns numeric `spy_return`/`alpha` when at least one SPY value in-window is finite; `None` only when no finite SPY exists.

- [ ] **Step 1: Write the failing test**

```python
import math
from scripts.generate_performance_page import compute_stats

def test_compute_stats_spy_alpha_ignores_trailing_nan():
    records = [
        {"date": "2026-04-01", "equity": 100000.0, "day_return": 0.0},
        {"date": "2026-04-02", "equity": 101000.0, "day_return": 0.01},
        {"date": "2026-04-03", "equity": 108710.0, "day_return": 0.02},
    ]
    # last SPY bar is NaN (today's unpublished bar) — must fall back to last finite
    spy = {"2026-04-01": 100000.0, "2026-04-02": 105000.0, "2026-04-03": float("nan")}
    s = compute_stats(records, spy)
    assert s["spy_return"] is not None and math.isfinite(s["spy_return"])
    assert s["alpha"] is not None and math.isfinite(s["alpha"])
    assert s["spy_return"] == 5.0  # (105000/100000 - 1) * 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_compute_stats_spy_alpha_ignores_trailing_nan -v`
Expected: FAIL — `spy_return` is `nan` (assert finite fails).

- [ ] **Step 3: Implement the fix**

Replace lines 455-458 with a finite-value walk for both ends:

```python
    def _finite_spy(date_list, forward):
        seq = date_list if forward else list(reversed(date_list))
        for d in seq:
            v = spy.get(d)
            if v is not None and math.isfinite(v):
                return v
        return None

    spy_base = _finite_spy(dates, forward=True)
    spy_cur  = _finite_spy(dates, forward=False)
    spy_ret  = ((spy_cur / spy_base - 1) * 100) if (spy_base and spy_cur) else None
    alpha    = (total_ret - spy_ret) if spy_ret is not None else None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_compute_stats_spy_alpha_ignores_trailing_nan -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_performance_page.py tests/test_generate_performance_page.py
git commit -m "fix: compute_stats ignores trailing NaN SPY bar for spy_return/alpha"
```

---

### Task 2: `_sparkline_paths` helper

**Files:**
- Modify: `scripts/generate_performance_page.py` (add helper near other loaders)
- Test: `tests/test_generate_performance_page.py`

**Interfaces:**
- Produces: `_sparkline_paths(symbols: list[str], n: int = 30) -> dict[str, dict]` → `{sym: {"d": "M..L..", "up": bool}}`. Reads `data_cache/prices_live.parquet` (columns `symbol,date,close`) once, takes the last `n` closes per symbol, maps to a 64×20 SVG path. Symbols with <2 points or missing → omitted from the dict. Any read error → returns `{}`.

- [ ] **Step 1: Write the failing test**

```python
from scripts.generate_performance_page import _sparkline_paths

def test_sparkline_paths_basic_and_missing(tmp_path, monkeypatch):
    import pandas as pd, scripts.generate_performance_page as g
    df = pd.DataFrame({
        "symbol": ["AAA"]*4 + ["BBB"]*1,
        "date":   ["2026-06-01","2026-06-02","2026-06-03","2026-06-04","2026-06-04"],
        "close":  [10.0, 11.0, 9.0, 12.0, 50.0],
    })
    p = tmp_path / "prices_live.parquet"; df.to_parquet(p)
    monkeypatch.setattr(g, "PRICES_LIVE_PATH", str(p), raising=False)
    out = _sparkline_paths(["AAA", "BBB", "ZZZ"])
    assert "AAA" in out and out["AAA"]["d"].startswith("M")
    assert out["AAA"]["up"] is True            # 12 >= 10
    assert "BBB" not in out                     # only 1 point
    assert "ZZZ" not in out                     # absent symbol
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_sparkline_paths_basic_and_missing -v`
Expected: FAIL — `_sparkline_paths` not defined / `PRICES_LIVE_PATH` not defined.

- [ ] **Step 3: Implement**

Add a module constant near the top (after other path constants) and the helper:

```python
PRICES_LIVE_PATH = "data_cache/prices_live.parquet"

def _sparkline_paths(symbols: list[str], n: int = 30) -> dict[str, dict]:
    """Last-n close series per symbol → 64x20 inline-SVG path string."""
    try:
        import pandas as pd
        df = pd.read_parquet(PRICES_LIVE_PATH, columns=["symbol", "date", "close"])
    except Exception:
        return {}
    out: dict[str, dict] = {}
    W, H = 64, 20
    want = set(symbols)
    for sym, g in df[df["symbol"].isin(want)].groupby("symbol"):
        ys = g.sort_values("date")["close"].dropna().tail(n).tolist()
        if len(ys) < 2:
            continue
        lo, hi = min(ys), max(ys)
        rng = (hi - lo) or 1.0
        pts = []
        for i, y in enumerate(ys):
            x = round(i * (W - 2) / (len(ys) - 1) + 1, 1)
            yy = round(2 + (1 - (y - lo) / rng) * (H - 4), 1)
            pts.append(f"{x},{yy}")
        out[str(sym)] = {"d": "M" + " L".join(pts), "up": ys[-1] >= ys[0]}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_sparkline_paths_basic_and_missing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_performance_page.py tests/test_generate_performance_page.py
git commit -m "feat: _sparkline_paths reads prices_live into per-symbol SVG paths"
```

---

### Task 3: `_redaction_label` helper

**Files:**
- Modify: `scripts/generate_performance_page.py`
- Test: `tests/test_generate_performance_page.py`

**Interfaces:**
- Produces: `_redaction_label(sym: str) -> str` — deterministic pick from a fixed 7-item rotation by `hash`-free index (`sum(ord)`), encoding nothing about the model.

- [ ] **Step 1: Write the failing test**

```python
from scripts.generate_performance_page import _redaction_label, _REDACTION_LABELS

def test_redaction_label_deterministic_and_in_set():
    assert _redaction_label("IFRA") == _redaction_label("IFRA")   # deterministic
    assert _redaction_label("IFRA") in _REDACTION_LABELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_redaction_label_deterministic_and_in_set -v`
Expected: FAIL — names not defined.

- [ ] **Step 3: Implement**

```python
_REDACTION_LABELS = [
    "Sleeve attribution", "Entry signal", "Risk weighting", "Cluster cap",
    "Regime threshold", "Correlation guard", "Exposure rule",
]

def _redaction_label(sym: str) -> str:
    return _REDACTION_LABELS[sum(ord(c) for c in sym) % len(_REDACTION_LABELS)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_redaction_label_deterministic_and_in_set -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_performance_page.py tests/test_generate_performance_page.py
git commit -m "feat: _redaction_label deterministic sealed-label rotation"
```

---

### Task 4: `_position_reasoning` helper

**Files:**
- Modify: `scripts/generate_performance_page.py`
- Test: `tests/test_generate_performance_page.py`

**Interfaces:**
- Consumes: latest verdict dict (`load_verdicts()[-1]`, keys incl. `verdict.key_risks`, `arguments`), prethesis dict (`data_cache/ai_prethesis_latest.json`, keys `high_conviction_names` = list of `{symbol, reason}`).
- Produces: `_position_reasoning(sym, verdict, prethesis) -> dict` → `{"why": str, "committee": str, "flagged": bool}`. `why` from prethesis conviction match else generic honest line. `committee` flagged (with sym) if sym appears in any `key_risks` string, else "No adversarial flag this cycle." All text HTML-escaped via existing `_esc`.

- [ ] **Step 1: Write the failing test**

```python
from scripts.generate_performance_page import _position_reasoning

def test_position_reasoning_conviction_and_flag():
    prethesis = {"high_conviction_names": [
        {"symbol": "IFRA", "reason": "AI data-center build-out beneficiary."}]}
    verdict = {"verdict": {"key_risks": [
        "BYD earnings imminent at 7.2% weight — unhedgeable binary."]}}
    ifra = _position_reasoning("IFRA", verdict, prethesis)
    assert "build-out" in ifra["why"]
    assert ifra["flagged"] is False
    byd = _position_reasoning("BYD", verdict, prethesis)
    assert byd["flagged"] is True and "BYD" in byd["committee"]
    # name with neither: honest generic fallback, not blank, not fabricated
    zzz = _position_reasoning("ZZZ", verdict, prethesis)
    assert zzz["why"] and zzz["flagged"] is False
    assert "No adversarial flag" in zzz["committee"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_position_reasoning_conviction_and_flag -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**

```python
def _position_reasoning(sym: str, verdict: dict, prethesis: dict) -> dict:
    why = "Held on the composite cross-sectional ranking."
    for n in (prethesis or {}).get("high_conviction_names", []) or []:
        if isinstance(n, dict) and n.get("symbol") == sym and n.get("reason"):
            why = _esc(str(n["reason"]).split(". ")[0].rstrip(".") + ".")
            break
    risks = ((verdict or {}).get("verdict") or {}).get("key_risks") or []
    hit = next((r for r in risks if sym in str(r)), None)
    if hit:
        committee = f"<span class='whyflag'><b>Flagged.</b></span> {_esc(str(hit))}"
        flagged = True
    else:
        committee = "No adversarial flag this cycle."
        flagged = False
    return {"why": why, "committee": committee, "flagged": flagged}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_position_reasoning_conviction_and_flag -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_performance_page.py tests/test_generate_performance_page.py
git commit -m "feat: _position_reasoning assembles qualitative why/committee from real sources"
```

---

### Task 5: `_construction_section_html` helper

**Files:**
- Modify: `scripts/generate_performance_page.py`
- Test: `tests/test_generate_performance_page.py`

**Interfaces:**
- Consumes: `regime_label: str`, `verdict: dict`, `authority: dict` (keys `level,title,ai_weight`), `n_held: int`, `universe_n: int`.
- Produces: `_construction_section_html(regime_label, verdict, authority, n_held, universe_n) -> str`. Data-driven from an internal `_CONSTRUCTION_STAGES` list. Renders funnel + 7 stages. Must not raise on empty/missing args. Markup/CSS classes copied from `mockups/06-construction.html` (`.funnel`, `.fn`, `.pipe`, `.stage`, `.sealchip`, `.openchip`, `.sealednote`, `.endcap`). The stage copy strings are the confidentiality-audited source — keep them in ONE list.

- [ ] **Step 1: Write the failing test**

```python
from scripts.generate_performance_page import _construction_section_html, _CONSTRUCTION_STAGES

def test_construction_section_renders_and_is_sealed():
    html = _construction_section_html("calm_bull",
        {"verdict": {"recommendation": "proceed", "confidence": 0.62}},
        {"level": 1, "title": "Analyst", "ai_weight": 0.05}, 17, 500)
    assert "How the book is built" in html
    assert "Sealed" in html and "calm_bull" in html
    # confidentiality: no tunable leaks
    for banned in ("0.70", "sleeve_weight", "DEFAULT_ALPHA_WEIGHTS", "trend=", "0.45"):
        assert banned not in html
    # never raises on empty inputs
    assert _construction_section_html("", {}, {}, 0, 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_construction_section_renders_and_is_sealed -v`
Expected: FAIL — not defined.

- [ ] **Step 3: Implement**

Port the funnel + stage markup from `mockups/06-construction.html`. Define `_CONSTRUCTION_STAGES` as a list of dicts `{n, name, kind: "visible"|"sealed", desc, tail, chips: [...], sealednote: str|None}` (copy the seven stages' exact text from the mockup). Build the HTML string by iterating the list; inject live values: stage 5 tail → `regime_label`; stage 6 tail → `proceed · {confidence}` from `verdict`; stage 7 tail → `{ai_weight*100:.0f}% authority`; funnel `Held` → `n_held`, `Universe` → rounded `universe_n`. Use `.get()` with defaults everywhere so empty dicts are safe. Reuse `_esc`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py::test_construction_section_renders_and_is_sealed -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_performance_page.py tests/test_generate_performance_page.py
git commit -m "feat: _construction_section_html data-driven sealed pipeline section"
```

---

### Task 6: Editorial shell — CSS tokens + masthead/lede/stats/footer

**Files:**
- Modify: `scripts/generate_performance_page.py` (the `<style>` block + top/bottom markup inside `build_html`)

**Interfaces:**
- Consumes: existing `stats` dict, `generated_at`, `regime_label`.
- Produces: a re-themed page shell. Equity/verdict/book/AI-desk bodies still render (old markup acceptable mid-migration) but the page wrapper, fonts, masthead, lede, secondary-stats strip, and footer match `mockups/05-rich.html`.

- [ ] **Step 1:** Replace the `<head>` font links + entire `<style>` block with the token system and editorial CSS from `mockups/05-rich.html` (`:root` vars, `.wrap`, `.mast`, `.lede`, `.figs`, `.bar`, `.sec`, `footer`, `.rev`). Keep existing chart/AI-section classes appended below for now so nothing 404s.

- [ ] **Step 2:** Replace the hero markup with the masthead + lede from the mockup, wiring real values: `nav` (count-up via `data-to`), `total_return`, `spy_return`, `alpha`, `sharpe`, `regime_label`. Use `_fmt_pct`/`_pct_color` where signs/colors apply; alpha uses `--dn`/`--up` by sign.

- [ ] **Step 3:** Replace the `.stats` grid with the secondary `.bar` strip (Max DD / Best / Worst / Start NAV / Sessions / Since). Replace footer markup with the editorial footer.

- [ ] **Step 4: Verify render**

Run:
```bash
.venv/bin/python scripts/generate_performance_page.py && \
.venv/bin/python -c "d=open('docs/index.html').read(); assert 'Source Serif' in d and 'nan%' not in d and 'How the book' not in d; print('shell ok, no nan')"
```
Expected: prints `shell ok, no nan` (alpha/SPY numeric; construction not yet wired).

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_performance_page.py docs/index.html
git commit -m "feat: editorial shell — tokens, masthead, lede, stats strip, footer"
```

---

### Task 7: Equity chart re-theme + base interaction JS

**Files:**
- Modify: `scripts/generate_performance_page.py` (chart card markup + Chart.js options + `<script>`)

**Interfaces:**
- Consumes: `dates_js`, `port_js`, `spy_js`, `annotations_js`.
- Produces: gold portfolio line, dashed muted SPY, hairline grid, mono ticks, faint gold fill; plus shared JS utilities `countUp`, `.rev` IntersectionObserver, `fillBars` (used by later tasks), all reduced-motion-guarded.

- [ ] **Step 1:** Re-theme the equity `chart-card` to `.chart`/`.chart-frame` editorial markup + honest caption (from mockup). Update Chart.js dataset colors to `--gold` / `#5c574a` dashed, grid to `--rule2`, fonts to IBM Plex Mono, legend off (use the `.cap` legend). Keep verdict annotations + regime bands.
- [ ] **Step 2:** Add the shared JS block from `mockups/05-rich.html`: `countUp`, the `.rev` observer, and `fillBars` (fills `.wbar i`,`.meter i`,`.fnbar`). Guard all with `REDUCED`.
- [ ] **Step 3:** Wrap major sections in `class="... rev"` so they fade up.
- [ ] **Step 4: Verify render**

Run:
```bash
.venv/bin/python scripts/generate_performance_page.py && \
.venv/bin/python -c "d=open('docs/index.html').read(); assert 'countUp' in d and \"'#cba569'\" in d.replace('\"',\"'\") or '#cba569' in d; print('chart themed')"
```
Expected: prints `chart themed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_performance_page.py docs/index.html
git commit -m "feat: editorial equity chart theme + shared count-up/reveal JS"
```

---

### Task 8: Verdict section — meter, voices, judge, rebuttal expander

**Files:**
- Modify: `scripts/generate_performance_page.py` (`_debate_html` or a new `_verdict_section_html`)

**Interfaces:**
- Consumes: `load_verdicts()[-1]` (`verdict.recommendation`, `verdict.confidence`, `verdict.key_risks`, `arguments.{bull,bear,devils_advocate,regime_specialist}` + `*_rebuttal`).
- Produces: section markup matching mockup §6 — badge (color by recommendation), conviction `.meter` (fills to confidence), four `.voice` blocks (first ~2 sentences each), one `.showmore` expander revealing all `.exp[data-grp=debate]` Round-2 rebuttals, `.judge` synthesis (from `verdict.rationale`/`summary` else composed), numbered `.risks`.

- [ ] **Step 1:** Write `_verdict_section_html(verdicts)`; excerpt each agent argument to the first 2 sentences for the closed state, full rebuttal inside `.exp`. Map recommendation→class (`proceed`→up, `reduce_size`→gold, `halt_and_review`→down). Empty verdicts → honest empty state.
- [ ] **Step 2:** Add the `.showmore` grouped-expander JS + the `.exp` grid-rows CSS (from mockup) to the style/script blocks.
- [ ] **Step 3: Verify render**

Run:
```bash
.venv/bin/python scripts/generate_performance_page.py && \
.venv/bin/python -c "d=open('docs/index.html').read(); assert 'Judge · Synthesis' in d or 'Judge' in d; assert 'conviction' in d.lower(); print('verdict ok')"
```
Expected: prints `verdict ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_performance_page.py docs/index.html
git commit -m "feat: editorial verdict section — meter, voices, judge synthesis, rebuttal expander"
```

---

### Task 9: The book — ribbon, sparklines, expandable reasoning, redaction

**Files:**
- Modify: `scripts/generate_performance_page.py` (`_positions_html` → `_book_section_html`)

**Interfaces:**
- Consumes: `positions` (real `symbol`,`weight`,`current_price`,`unrealized_plpc`), `_sparkline_paths`, `_position_reasoning`, `_redaction_label`, latest verdict, prethesis (`load_latest_thesis` or read `data_cache/ai_prethesis_latest.json`).
- Produces: allocation ribbon (segment flex = weight, highlights on open), per-row sparkline (inline SVG, omit if absent), expandable two-column reasoning + redaction strip; footnote. Row disclosure + ribbon-link JS from mockup.

- [ ] **Step 1:** Write `_book_section_html(positions, verdict, prethesis)`: build `spark = _sparkline_paths([p['symbol'] for p in positions])`; for each position emit the `.row` markup from `mockups/05-rich.html` (sym, sparkline SVG or blank, `.wbar` width = `weight/max_weight*100`, `%`, class label, caret) + `.exp` reasoning from `_position_reasoning` + redaction strip with `_redaction_label`. Sort by weight desc.
- [ ] **Step 2:** Add the ribbon container + the book disclosure JS (`.row-main` click toggles `.open`, lights `ribbon.children[i]`) and the CSS (`.book`,`.row`,`.spark`,`.ribbon`,`.redact`,`.bars`,`.whybox`) from the mockup.
- [ ] **Step 3: Verify render**

Run:
```bash
.venv/bin/python scripts/generate_performance_page.py && \
.venv/bin/python -c "d=open('docs/index.html').read(); assert 'class=\"ribbon\"' in d and 'Sealed' in d and 'sealed by design' in d; print('book ok')"
```
Expected: prints `book ok`.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_performance_page.py docs/index.html
git commit -m "feat: editorial book — ribbon, sparklines, on-demand reasoning, redaction strips"
```

---

### Task 10: Wire the construction section into the page

**Files:**
- Modify: `scripts/generate_performance_page.py` (`build_html` assembly + add `universe_n` source)

**Interfaces:**
- Consumes: `_construction_section_html`, `regime_label`, latest verdict, `authority`, `len(positions)`.
- Produces: section 5 placed between the stats strip and the verdict section. `universe_n` from the universe config if importable, else constant `500`.

- [ ] **Step 1:** In `build_html`, compute `universe_n` (try `from ascent.config import get_config` → universe size; `except Exception: universe_n = 500`) and insert `_construction_section_html(...)` output between the `.bar` strip and the verdict section. Add the construction CSS block (from `mockups/06-construction.html`) + its funnel-fill + stage-toggle JS.
- [ ] **Step 2: Verify render**

Run:
```bash
.venv/bin/python scripts/generate_performance_page.py && \
.venv/bin/python -c "d=open('docs/index.html').read(); assert 'How the book is built' in d and 'Adversarial Gate' in d; print('construction wired')"
```
Expected: prints `construction wired`.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_performance_page.py docs/index.html
git commit -m "feat: wire 'How the book is built' construction pipeline into page"
```

---

### Task 11: Re-skin the AI desk sub-widgets

**Files:**
- Modify: `scripts/generate_performance_page.py` (`_earned_authority_html`, `_promotion_gates_html`, `_counterfactual_chart_html`, `_override_scorecard_html`, `_thesis_html` + their CSS)

**Interfaces:**
- Consumes: unchanged (existing AI PM loaders).
- Produces: the AI PM section re-skinned to editorial tokens (authority "rung" ladder from `mockups/03-fullpage.html`, gold accents, mono labels, serif headings). No data-flow changes.

- [ ] **Step 1:** Retheme the `.ai-section`/`.ai-card`/ladder/gates/scorecard CSS to the token palette; keep Chart.js alloc + counterfactual but recolor to `--gold`/`--up`/`--dn`. Heading uses `.sec-h h2` serif style.
- [ ] **Step 2: Verify render**

Run:
```bash
.venv/bin/python scripts/generate_performance_page.py && \
.venv/bin/python -c "d=open('docs/index.html').read(); assert 'Earned Authority' in d or 'AI Portfolio Manager' in d or 'AI desk' in d; print('ai desk ok')"
```
Expected: prints `ai desk ok`.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_performance_page.py docs/index.html
git commit -m "style: re-skin AI desk widgets to editorial tokens"
```

---

### Task 12: Confidentiality gate + headless verification + final regen

**Files:**
- Test: `tests/test_generate_performance_page.py`
- Modify: `scripts/generate_performance_page.py` (only if a leak/console error is found)

**Interfaces:**
- Produces: a denylist test asserting the rendered `docs/index.html` leaks no tunables, plus a headless render check (zero console errors, key sections present).

- [ ] **Step 1: Write the confidentiality + structure test**

```python
import subprocess, sys, os

def test_rendered_page_seals_edge_and_has_sections():
    subprocess.run([".venv/bin/python", "scripts/generate_performance_page.py"], check=True)
    html = open("docs/index.html").read()
    banned = ["DEFAULT_ALPHA_WEIGHTS", "sleeve_weight", "trend=0", "meanrev",
              "regime_threshold", "tilt_strength", "hysteresis", "0.70 corr"]
    for b in banned:
        assert b not in html, f"edge leak: {b}"
    for sect in ["How the book is built", "The latest verdict", "The book",
                 "sealed by design"]:
        assert sect in html, f"missing: {sect}"
    assert "nan%" not in html
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_generate_performance_page.py -v`
Expected: PASS (whole file).

- [ ] **Step 3: Headless render check** (existing project pattern, Chrome)

Run:
```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --dump-dom "file://$(pwd)/docs/index.html" \
  > /tmp/dom.html 2>/tmp/chrome.err; \
grep -c "How the book is built" /tmp/dom.html
```
Expected: prints `1` (DOM renders; no fatal error). Manually open `docs/index.html` to sanity-check expanders + motion.

- [ ] **Step 4: Run the full existing suite for regressions**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: no NEW failures vs the pre-existing baseline (6 known `test_wf_framework` failures unrelated).

- [ ] **Step 5: Commit**

```bash
git add tests/test_generate_performance_page.py docs/index.html
git commit -m "test: confidentiality denylist + structure gate for dashboard"
```

---

## Self-Review

**Spec coverage:** §4 sections → Tasks 6 (1,2,4,10 masthead/lede/stats/footer), 7 (3 equity), 5/10 (construction), 8 (6 verdict), 11 (7 AI desk), 9 (8 book); §5.2 SPY bug → Task 1; §5.8 sparkline/reasoning/redaction → Tasks 2,3,4,9; §6 interactions → Tasks 7,8,9,10; §7 helpers → Tasks 1–5; §8 testing → Task 12. All covered.

**Placeholder scan:** Testable helpers (Tasks 1–5, 12) carry complete code. HTML/CSS tasks (6–11) reference the on-disk committed-this-session mockups as the exact visual source rather than re-pasting ~600 lines of CSS — acceptable because execution happens in this repo where `mockups/05-rich.html` and `06-construction.html` exist; each such step still names exact classes/values to wire and ends in an asserted render check.

**Type consistency:** `_sparkline_paths → {sym:{d,up}}` consumed in Task 9; `_position_reasoning → {why,committee,flagged}` consumed in Task 9; `_construction_section_html(regime_label, verdict, authority, n_held, universe_n)` consumed in Task 10; `_redaction_label` consumed in Task 9; `compute_stats` keys unchanged. Consistent.
