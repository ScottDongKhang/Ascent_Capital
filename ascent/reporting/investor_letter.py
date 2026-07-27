"""
ascent/reporting/investor_letter.py

Monthly investor letter generator for Ascent Capital.
Triggers on the first trading day of each new month after run_all_agents completes.
Saves to outputs/investor_reports/YYYY-MM-letter.md

Voice/format: the Ascent Capital template — first-person singular, concentrated momentum,
no technology discussion, Goldman-partner + serious-builder sentence test throughout.
"""
from __future__ import annotations

import json
import math
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Apr 1 was live-trading inception. Apr 1–5 weights are not in the run log
# (different initial portfolio). ITD computation starts from first recorded weights.
INCEPTION_DATE      = date(2026, 4, 1)   # displayed in letter header
FIRST_WEIGHTS_DATE  = date(2026, 4, 6)   # first date weights exist in run log
LETTER_DIR     = Path("outputs/investor_reports")
RUNS_LOG       = Path("logs/multi_agent_run.jsonl")
ATTR_LOG       = Path("logs/attribution_log.jsonl")
PRICES_CACHE   = Path("data_cache/prices_live.parquet")
DEBATE_DIR     = Path("outputs/debate_log")
REGIME_PATH    = Path("dashboard/regime_signal.json")
KILL_SW_PATH   = Path("logs/kill_switch_state.json")

# Walk-forward OOS Sharpe — update when new OOS record is computed
WF_SHARPE = 0.52
WF_PERIOD  = "Jan 2020–Apr 2026"


# ── Detection ────────────────────────────────────────────────────────────────

def is_first_trading_day_of_month(today: date) -> bool:
    """True if today is the first Mon–Fri of a new calendar month."""
    if today.weekday() >= 5:
        return False
    prev = today - timedelta(days=1)
    while prev.weekday() >= 5:
        prev -= timedelta(days=1)
    return prev.month != today.month


def prior_month(today: date) -> tuple[int, int]:
    """Return (year, month) of the month before today."""
    if today.month == 1:
        return today.year - 1, 12
    return today.year, today.month - 1


# ── Data helpers ─────────────────────────────────────────────────────────────

def _load_prices() -> dict:
    """Load prices_live.parquet → {symbol: {date: close}}."""
    try:
        import pyarrow.parquet as pq
        import pandas as pd
        df = pq.read_table(str(PRICES_CACHE)).to_pandas()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        pivot = df.pivot(index="date", columns="symbol", values="close")
        returns = pivot.pct_change()
        return returns.to_dict("index")  # {date: {symbol: return}}
    except Exception as e:
        log.warning(f"[Letter] Could not load prices: {e}")
        return {}


def _load_weights() -> list[dict]:
    """
    Load weight entries from multi_agent_run.jsonl.
    Deduplicated by date — keeps the LAST entry per date (most recent run).
    Returned sorted ascending by date.
    """
    by_date: dict[str, dict] = {}
    try:
        with open(RUNS_LOG) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "weights" in r and r["weights"] and "date" in r:
                        by_date[r["date"]] = r  # overwrite = keep last
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return sorted(by_date.values(), key=lambda r: r["date"])


def _active_weights(weight_entries: list[dict], target_date: date) -> Optional[dict]:
    """
    Return the most recent weights on or before target_date.
    Returns None if no weights exist on or before that date — those days
    are skipped in return computation rather than estimated with a proxy.
    (Apr 1–5, 2026 had a completely different initial portfolio whose exact
    weights are not in the run log; using Apr 6 weights as proxy would be wrong.)
    """
    best = None
    best_date = None
    for r in weight_entries:
        rd = r.get("date", "")
        if rd <= str(target_date):
            if best_date is None or rd > best_date:
                best = r["weights"]
                best_date = rd
    return best


def _trading_days_in_month(year: int, month: int, daily_returns: dict) -> list[date]:
    """All trading days (Mon–Fri) in the given month that have price data."""
    days = []
    d = date(year, month, 1)
    while d.month == month:
        if d.weekday() < 5 and d in daily_returns:
            days.append(d)
        d += timedelta(days=1)
    return days


def _compute_period_returns(
    days: list[date],
    weight_entries: list[dict],
    daily_returns: dict,
) -> list[dict]:
    """Compute {date, port, spy} for each trading day."""
    results = []
    for d in days:
        row = daily_returns.get(d, {})
        if not row:
            continue
        w = _active_weights(weight_entries, d)
        if not w:
            continue
        port = sum(w.get(s, 0) * row.get(s, 0)
                   for s in w if row.get(s) is not None and not math.isnan(row.get(s, float("nan"))))
        spy = row.get("SPY", 0) or 0
        logged = str(d) in {r.get("date", "") for r in weight_entries}
        results.append({"date": d, "port": port, "spy": spy, "logged": logged})
    return results


def _cumulative(results: list[dict]) -> tuple[float, float]:
    """Return (cum_port, cum_spy) as multipliers."""
    cp = cs = 1.0
    for r in results:
        cp *= (1 + r["port"])
        cs *= (1 + r["spy"])
    return cp, cs


def _max_drawdown(results: list[dict], key: str) -> float:
    """Peak-to-trough max drawdown for 'port' or 'spy'."""
    peak = 1.0
    nav = 1.0
    mdd = 0.0
    for r in results:
        nav *= (1 + r[key])
        if nav > peak:
            peak = nav
        dd = (nav - peak) / peak
        if dd < mdd:
            mdd = dd
    return mdd


def _annualized_vol(results: list[dict], key: str) -> float:
    """Annualized daily volatility."""
    rets = [r[key] for r in results]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def _beta_estimate(results: list[dict]) -> float:
    """OLS beta of fund vs SPY from daily returns."""
    if len(results) < 5:
        return 1.0
    px = [r["port"] for r in results]
    sx = [r["spy"]  for r in results]
    n  = len(px)
    mx = sum(px) / n
    my = sum(sx) / n
    cov = sum((px[i] - mx) * (sx[i] - my) for i in range(n))
    var = sum((sx[i] - my) ** 2 for i in range(n))
    return cov / var if var > 0 else 1.0


# ── Attribution ──────────────────────────────────────────────────────────────

def _compute_attribution(
    days: list[date],
    weight_entries: list[dict],
    daily_returns: dict,
) -> dict[str, float]:
    """Sum portfolio contribution (weight × return) per symbol over the period."""
    contrib: dict[str, float] = {}
    for d in days:
        row = daily_returns.get(d, {})
        if not row:
            continue
        w = _active_weights(weight_entries, d)
        if not w:
            continue
        for sym, wt in w.items():
            ret = row.get(sym)
            if ret is None or math.isnan(ret):
                continue
            contrib[sym] = contrib.get(sym, 0.0) + wt * ret
    return contrib


def _top_n(contrib: dict[str, float], n: int, positive: bool) -> list[tuple[str, float]]:
    sorted_c = sorted(contrib.items(), key=lambda x: x[1], reverse=positive)
    return sorted_c[:n]


# ── Events ───────────────────────────────────────────────────────────────────

def _get_rebalance_events(year: int, month: int) -> list[dict]:
    """Read debate verdicts for the given month."""
    events = []
    prefix = f"verdict_{year}-{month:02d}"
    try:
        for fpath in sorted(DEBATE_DIR.glob(f"{prefix}*.json")):
            try:
                d = json.loads(fpath.read_text())
                v = d.get("verdict", {})
                if isinstance(v, dict):
                    events.append({
                        "date": fpath.stem.replace("verdict_", ""),
                        "recommendation": v.get("recommendation", "unknown"),
                        "confidence": v.get("confidence"),
                        "changes": v.get("position_changes", []),
                        "key_risks": (v.get("key_risks") or [])[:2],
                    })
            except Exception:
                pass
    except Exception:
        pass
    return events


def _get_regime() -> str:
    try:
        d = json.loads(REGIME_PATH.read_text())
        return d.get("regime", "unknown")
    except Exception:
        return "unknown"


def _get_kill_switch_state() -> dict:
    try:
        return json.loads(KILL_SW_PATH.read_text())
    except Exception:
        return {}


def _get_next_rebalance(after: date) -> Optional[str]:
    """Return the next rebalance date after `after` from rebalance_calendar.csv."""
    cal_path = Path("rebalance_calendar.csv")
    try:
        with open(cal_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    d = date.fromisoformat(line.split(",")[0].strip())
                    if d > after:
                        return str(d)
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return None


def _get_skill_score_progress() -> dict:
    """Return days elapsed per agent from skill_scores_log.jsonl."""
    log_path = Path("logs/skill_scores_log.jsonl")
    try:
        with open(log_path) as f:
            lines = [l for l in f if l.strip()]
        if not lines:
            return {}
        last = json.loads(lines[-1])
        return {
            agent: data.get("n_days", 0)
            for agent, data in last.items()
            if isinstance(data, dict)
        }
    except Exception:
        return {}


# ── Letter generation via LLM ────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are writing a monthly investor letter for Ascent Capital.

Ascent Capital is an AI-native fund — not a quant fund that explains its models, not a \
fundamental fund that added a screener. The infrastructure is the baseline assumption, \
invisible like electricity. What is visible is the thinking: theses formed before positions \
are entered, causal mechanisms that must be stated and tracked, adversarial review before \
every execution, and a structured memory of what worked and what did not. The letter reflects \
that character without ever naming the tools. A reader should sense a different kind of \
investment process — one where judgment is accountable, not just expressed.

PROTECT THE EDGE: Do not name, describe, or imply the specific investment strategy or \
methodology anywhere in the letter. Do not use terms like "momentum", "cross-sectional", \
"factor", "signal", "systematic", "quant", or any language that reveals how positions are \
selected. Describe positions by their thesis and outcome only. A reader should know what \
happened and why it mattered — not how the portfolio is constructed.

IDENTITY: First-person singular. One person's thinking, with the weight of a structured \
process behind it. Precise, direct, occasionally sharp. Never corporate. Never generic quant.

SENTENCE TEST: Would a Goldman partner and a serious builder both respect this sentence for \
different reasons? If not, rewrite it.

WHAT "AI-NATIVE" SOUNDS LIKE IN THIS LETTER — no technology words required:
  — Positions are entered with stated theses, not just signals. When a thesis proved out, say \
    so precisely. When it did not, name what broke.
  — Every rebalance involves an adversarial review. The letter can reference what the review \
    challenged, what it changed, and what it left alone — without explaining how the review works.
  — Causal mechanisms are tracked. A position is not just "up" or "down" — it either confirmed \
    or failed the thesis it was entered on.
  — Losses are logged as structured learning, not narrative. If a position broke its mechanism, \
    that is a data point that changes how the next similar position will be sized.
  — The fund knows what it does not know. Uncertainty is stated precisely, not hedged vaguely.

NEVER WRITE: "the model said", "the algorithm selected", "the system flagged", "AI determined", \
"machine learning identified". These are tool descriptions. Write the judgment, not the tool.

BANNED WORDS (hard delete): pleased, excited, challenging, headwinds, tailwinds, environment, \
opportunity set, constructive, cautious optimism, that said, it is worth noting, however, \
as mentioned, going forward, remained resilient, navigated, demonstrated

LAYOUT: Use --- between sections. Section headers in plain caps. No bold in headers. \
No bullet points in the letter body. Tables only for numbers.

OUTPUT EXACTLY THIS STRUCTURE — no deviation:

ASCENT CAPITAL
{month_name} {year} Investor Letter
Benchmark: S&P 500 | Paper Trading | Simulated vs Live Prices

---

I. THE NUMBER

[Performance table: Fund (gross) | S&P 500 | Net Alpha — monthly column + ITD column]
[Risk block: Max Drawdown Fund vs Benchmark | Annualized Vol Fund vs Benchmark | \
Sharpe live (if calculable) and walk-forward OOS | Beta to SPY estimated | Calmar if meaningful]
[Exactly two sentences. First: honest risk-adjusted interpretation. Second: does not soften it. \
No third sentence.]

---

II. WHAT DROVE RETURNS

[Four positions: top 2 contributors then top 2 detractors. Each gets one paragraph structured \
as flowing prose that moves through this logic — never labeled, never broken into sub-points:
  What happened + the number →
  Whether the stated thesis proved out, or whether this was pure factor / signal with no thesis →
  What the adversarial review said about it, if anything →
  One honest uncertainty or open question that remains
After each paragraph, one italicized data line:
  Weight at entry → exit | Contribution | Thesis status: confirmed / falsified / no thesis stated
If a position made money but had no stated thesis, credit the factor explicitly — do not imply \
process credit where none exists. If a position lost money and broke its mechanism, say what \
the mechanism was and what broke it. If there is no lesson, say there is no lesson.]

---

III. ONE THING WE LEARNED

[One idea. Not a list. 350–450 words. One thing examined with full analytical attention — \
preferably something about how the investment process itself behaved, not just how the market \
moved. What did the thesis-tracking reveal? What did the adversarial review get right or wrong? \
What is the fund now doing differently because of what happened this month? \
A serious allocator should finish this section knowing something precise about how this fund \
thinks differently from a traditional momentum manager.
End with exactly this sentence form:
"This would be wrong if: [specific, observable, 30-day falsification condition]."]

---

IV. RISK STATE

[Status readout. No narrative. Exactly four labeled items:
Regime: [classification + one-line implication]
Event Risk: [any binary events, position weights, timing, upcoming rebalance]
Exposure: [gross / net equity]
Kill Switch: [current distance to −8% soft / −15% hard]
One additional monitoring note if warranted. Total section: maximum 150 words.]

---

V. WHAT WOULD PROVE US WRONG

The strategy fails if the behavioral patterns that drive returns are arbitraged away. The edge \
depends on persistent mispricings that arise from how investors process information over time. \
If those patterns erode — through the accumulation of capital chasing similar theses, or through \
structural changes in how markets react to new data — the return premium shrinks. Crowding is \
monitored continuously. There is no definitive answer to how much capital a behavioral edge \
can absorb before it degrades.

The strategy fails if regime classification lags genuine transitions. The defensive infrastructure \
— gross exposure reductions in stressed and crisis regimes, the SPY 200-day overlay, defensive \
sleeve reallocation — is only valuable if the regime model identifies transitions before the damage \
is done. If it systematically lags genuine regime shifts by more than a few sessions, the \
protection is theoretical. The COVID crash was a regime transition of extraordinary speed. The \
walk-forward data suggests the model survived it. The live record has not been tested against \
anything similar.

The strategy fails if process is confused for edge. The risk I take most seriously is believing \
that because the analytical process is rigorous, the outcomes will be good. Process quality is \
necessary but not sufficient. I can make excellent decisions and lose money. I can make poor \
decisions and make money. The live record is too short to distinguish between them. \
Any allocator drawing strong conclusions from this period — in either direction — is drawing \
conclusions the data cannot support.

The strategy fails if concentration becomes a liability. Fifteen positions at full conviction is \
a risk-amplifying choice, not a risk-reducing one. In a market where specific names face \
simultaneous company-level stress — regulatory action, sector rotation, earnings disappointment \
— the portfolio has limited capacity to absorb the shocks. This trade-off is accepted because \
the information ratio of a concentrated portfolio is believed to exceed that of a diluted one \
at the available signal quality. That belief could be wrong.

---

VI. WHAT'S AHEAD

[Write 2–4 short paragraphs. No bullet points. Cover in natural prose:
  — The next scheduled rebalance: date and what makes it notable (use the FORWARD LOOKING data provided)
  — Any operational milestones approaching: skill score unlock timeline, system changes going live
  — One or two market or position-level things being watched that could change the thesis
  — Nothing speculative. Nothing promised. Only what is concrete and observable.
Voice standard applies: no banned words, no softening, no hype. \
State what is coming and why it matters. If nothing notable is coming, say so briefly.]

---

THANK YOU

[Two to three sentences. First-person. Genuine — not performative. \
Acknowledge that the reader is following a paper-trading fund: that is early-stage trust, \
not track-record trust, and it means something. \
State one concrete commitment — to transparency, to honesty about mistakes, to writing \
the same way whether the month was good or bad. \
Do not use: grateful, honored, privileged, pleased, excited. \
Write like someone who means it, not someone who is required to say it.]

---

Ascent Capital is currently in paper trading mode. All performance figures are simulated against \
live market prices and do not represent audited live returns. Past simulated performance does not \
guarantee future results. This letter does not constitute an offer or solicitation. \
For accredited investors only.

Ascent Capital Management | {month_name} {year}
"""


def _build_user_prompt(
    year: int,
    month: int,
    monthly_results: list[dict],
    itd_results: list[dict],
    attribution: dict[str, float],
    events: list[dict],
    regime: str,
    kill_sw: dict,
    current_weights: dict,
    next_rebalance: Optional[str] = None,
    skill_progress: Optional[dict] = None,
) -> str:
    import calendar

    month_name = calendar.month_name[month]

    # ── Performance ──────────────────────────────────────────────────
    if monthly_results:
        cm, cs = _cumulative(monthly_results)
        m_port  = (cm - 1) * 100
        m_spy   = (cs - 1) * 100
        m_alpha = m_port - m_spy
        m_mdd_f = _max_drawdown(monthly_results, "port") * 100
        m_mdd_s = _max_drawdown(monthly_results, "spy")  * 100
        m_vol_f = _annualized_vol(monthly_results, "port") * 100
        m_vol_s = _annualized_vol(monthly_results, "spy")  * 100
    else:
        m_port = m_spy = m_alpha = m_mdd_f = m_mdd_s = m_vol_f = m_vol_s = 0.0

    if itd_results:
        ip, is_ = _cumulative(itd_results)
        itd_port  = (ip  - 1) * 100
        itd_spy   = (is_ - 1) * 100
        itd_alpha = itd_port - itd_spy
        itd_mdd_f = _max_drawdown(itd_results, "port") * 100
        itd_mdd_s = _max_drawdown(itd_results, "spy")  * 100
        itd_vol_f = _annualized_vol(itd_results, "port") * 100
        itd_vol_s = _annualized_vol(itd_results, "spy")  * 100
        beta      = _beta_estimate(itd_results)
        n_days    = len(itd_results)
    else:
        itd_port = itd_spy = itd_alpha = 0.0
        itd_mdd_f = itd_mdd_s = itd_vol_f = itd_vol_s = 0.0
        beta = 1.0
        n_days = 0

    # ── Attribution top 4 ────────────────────────────────────────────
    top2_pos = _top_n(attribution, 2, positive=True)
    top2_neg = _top_n(attribution, 2, positive=False)

    def fmt_attr(items):
        lines = []
        for sym, val in items:
            wt = current_weights.get(sym, "unknown")
            wt_str = f"{wt*100:.1f}%" if isinstance(wt, float) else "exited"
            lines.append(f"  {sym}: contribution {val*100:+.2f}% | current weight {wt_str}")
        return "\n".join(lines) or "  (none)"

    # ── Events ───────────────────────────────────────────────────────
    rebal_lines = []
    for ev in events:
        changes = ev.get("changes", [])
        change_str = "; ".join(
            f"{c['symbol']} {c['current_weight']*100:.1f}%→{c['new_weight']*100:.1f}%"
            for c in changes
        ) or "no position changes"
        rebal_lines.append(
            f"  {ev['date']}: verdict={ev['recommendation']} "
            f"(confidence {ev.get('confidence','?')}) | {change_str}"
        )
        if ev.get("key_risks"):
            for r in ev["key_risks"]:
                rebal_lines.append(f"    risk flagged: {r[:120]}")
    rebal_str = "\n".join(rebal_lines) or "  (no rebalances this month)"

    # ── Kill switch ───────────────────────────────────────────────────
    ks_drawdown = kill_sw.get("current_drawdown_pct", 0.0)
    dist_soft   = -8.0 - ks_drawdown   if ks_drawdown < 0 else 8.0
    dist_hard   = -15.0 - ks_drawdown  if ks_drawdown < 0 else 15.0

    # ── Current top holdings ──────────────────────────────────────────
    sorted_wts = sorted(current_weights.items(), key=lambda x: x[1], reverse=True)[:10]
    holdings_str = "  " + ", ".join(f"{s} {w*100:.1f}%" for s, w in sorted_wts)

    # ── Forward-looking context ───────────────────────────────────────
    next_reb_str = f"Next scheduled rebalance: {next_rebalance}" if next_rebalance else "Next rebalance: unknown"
    skill_str = ""
    if skill_progress:
        needs_63 = {a: max(0, 63 - n) for a, n in skill_progress.items()}
        soonest = min(needs_63.values()) if needs_63 else 63
        skill_str = (
            f"Skill score warmup: {skill_progress} days elapsed per agent. "
            f"Earliest unlock in ~{soonest} more trading days."
        )

    return f"""PERFORMANCE:
  Month ({month_name} {year}):
    Fund (gross): {m_port:+.2f}%
    S&P 500:      {m_spy:+.2f}%
    Net alpha:    {m_alpha:+.2f}%
  Inception-to-Date (Apr 6 – month end, first recorded portfolio):
    Fund (gross): {itd_port:+.2f}%
    S&P 500:      {itd_spy:+.2f}%
    Net alpha:    {itd_alpha:+.2f}%

RISK METRICS:
  Max drawdown — month:  Fund {m_mdd_f:.2f}%  /  SPY {m_mdd_s:.2f}%
  Max drawdown — ITD:    Fund {itd_mdd_f:.2f}%  /  SPY {itd_mdd_s:.2f}%
  Annualized vol — month: Fund {m_vol_f:.1f}%  /  SPY {m_vol_s:.1f}%
  Annualized vol — ITD:   Fund {itd_vol_f:.1f}%  /  SPY {itd_vol_s:.1f}%
  Beta to SPY (ITD, OLS): {beta:.2f}
  Walk-forward OOS Sharpe ({WF_PERIOD}): {WF_SHARPE}
  Live Sharpe: not calculable at {n_days} sessions
  Calmar: not meaningful at this sample length

ATTRIBUTION — {month_name} {year} (weight × realized return, beginning-of-period weights, not fill-confirmed):
  Top contributors:
{fmt_attr(top2_pos)}
  Top detractors:
{fmt_attr(top2_neg)}

EVENTS:
  Rebalances and verdicts this month:
{rebal_str}
  Regime at month-end: {regime}
  Kill switch state: current drawdown ~{ks_drawdown:.1f}% | distance to soft warning (~{dist_soft:.1f}pp) | distance to hard stop (~{dist_hard:.1f}pp)

CURRENT PORTFOLIO (top 10 by weight):
{holdings_str}

RETURN DISTRIBUTION (for Section III):
  Positive alpha days: {len([r for r in monthly_results if r['alpha'] > 0]) if monthly_results else 0} of {len(monthly_results)} sessions
  Largest positive alpha day: {max((r['port']-r['spy'] for r in monthly_results), default=0)*100:+.2f}%
  Largest negative alpha day: {min((r['port']-r['spy'] for r in monthly_results), default=0)*100:+.2f}%
  Gap days (pipeline did not run, weights carried forward): {len([r for r in monthly_results if not r.get('logged', True)]) if monthly_results else 'unknown'}

INVESTMENT PROCESS CONTEXT (use this to write AI-native sections — do not describe the technology, \
describe the judgment and what it produced):
  — Before each rebalance, a thesis is formed independently before quant data arrives. \
    Positions entered with a stated causal mechanism must have that mechanism confirmed or \
    falsified each cycle. If the mechanism breaks, the position is reviewed for exit regardless \
    of the momentum signal.
  — Before each execution, an adversarial review challenges the portfolio. The review makes one \
    specific, falsifiable call per rebalance — not general guidance. That call is tracked. \
    Use the EVENTS data above to describe what the review flagged and what it changed.
  — Losses are logged as structured outcomes: was the thesis stated, did the mechanism hold, \
    was the exit triggered by the mechanism breaking or by signal decay. Use this framing for \
    the detractor paragraphs — not just "position lost money."
  — The fund tracks its own prediction accuracy. When a thesis is entered, a falsification \
    condition is stated. Section III should reflect this character — analytical accountability, \
    not just observation.
  Attribution figures reconstructed from model weights × prices. Not fill-confirmed.

FORWARD LOOKING (for Section VI):
  {next_reb_str}
  {skill_str if skill_str else "Skill score warmup data unavailable."}

Write the investor letter now. Follow the template exactly. Use the data above. \
Do not invent numbers not in the data. Do not use banned words. \
Write as an AI-native fund manager who does not need to say "AI" to sound like one — \
the accountability, the thesis-tracking, the adversarial discipline, and the structured \
learning are what make it different. Make that difference visible in every paragraph."""


def _write_letter_via_llm(user_prompt: str, year: int, month: int) -> str:
    """Call Sonnet to generate the letter."""
    import calendar
    month_name = calendar.month_name[month]

    try:
        from ascent.llm.client import get_client, SONNET_MODEL, extract_text
        client = get_client()
        system = _SYSTEM_PROMPT.format(month_name=month_name, year=year)
        resp = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return extract_text(resp)
    except Exception as e:
        log.error(f"[Letter] LLM call failed: {e}")
        return f"[Letter generation failed: {e}]\n\n{user_prompt}"


# ── Main entry point ─────────────────────────────────────────────────────────

def generate_monthly_letter(today: date | None = None) -> Optional[str]:
    """
    Generate and save the monthly investor letter if today is the first trading day
    of a new month. Returns the saved file path, or None if not triggered.
    """
    if today is None:
        today = date.today()

    if not is_first_trading_day_of_month(today):
        return None

    year, month = prior_month(today)
    log.info(f"[Letter] Generating investor letter for {year}-{month:02d}")
    print(f"[Letter] Generating monthly investor letter for {year}-{month:02d}...")

    # ── Load prices ──────────────────────────────────────────────────
    daily_returns = _load_prices()
    if not daily_returns:
        print("[Letter] No price data — skipping letter generation")
        return None

    weight_entries = _load_weights()

    # ── Monthly days ─────────────────────────────────────────────────
    monthly_days = _trading_days_in_month(year, month, daily_returns)
    monthly_results = _compute_period_returns(monthly_days, weight_entries, daily_returns)
    for r in monthly_results:
        r["alpha"] = r["port"] - r["spy"]

    # ── ITD days — start from FIRST_WEIGHTS_DATE, not INCEPTION_DATE ────────
    # Apr 1–5 had a different initial portfolio whose weights aren't in the log.
    itd_days: list[date] = []
    d = FIRST_WEIGHTS_DATE
    end = (date(year, month, 1) - timedelta(days=1))  # last day of prior month
    while d <= end:
        if d.weekday() < 5 and d in daily_returns:
            itd_days.append(d)
        d += timedelta(days=1)
    itd_days += monthly_days
    itd_results = _compute_period_returns(list(dict.fromkeys(itd_days)), weight_entries, daily_returns)

    # ── Attribution ──────────────────────────────────────────────────
    attribution = _compute_attribution(monthly_days, weight_entries, daily_returns)

    # ── Current weights (most recent run) ────────────────────────────
    current_weights: dict = {}
    if weight_entries:
        current_weights = weight_entries[-1].get("weights", {})

    # ── Events and regime ────────────────────────────────────────────
    events          = _get_rebalance_events(year, month)
    regime          = _get_regime()
    kill_sw         = _get_kill_switch_state()
    next_rebalance  = _get_next_rebalance(date(year, month, 1) + timedelta(days=31))
    skill_progress  = _get_skill_score_progress()

    # ── Build prompt and generate ────────────────────────────────────
    user_prompt = _build_user_prompt(
        year, month,
        monthly_results, itd_results,
        attribution, events, regime, kill_sw,
        current_weights,
        next_rebalance=next_rebalance,
        skill_progress=skill_progress,
    )

    letter = _write_letter_via_llm(user_prompt, year, month)

    # ── Save ─────────────────────────────────────────────────────────
    LETTER_DIR.mkdir(parents=True, exist_ok=True)
    fname = LETTER_DIR / f"{year}-{month:02d}-investor-letter.md"
    fname.write_text(letter)
    print(f"[Letter] Saved → {fname}")
    return str(fname)
