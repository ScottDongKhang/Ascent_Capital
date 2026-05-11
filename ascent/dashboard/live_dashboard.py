"""
ascent/dashboard/live_dashboard.py
Operator dashboard — internal real-time system monitoring.

Launch: streamlit run ascent/dashboard/live_dashboard.py --server.port 8502

Four tabs:
  Portfolio — live NAV, positions, sector allocation
  Risk      — factor exposures, drawdown, VaR, kill switch
  Alpha     — trailing IC by sleeve, self-improve status
  Events    — live event agent feed
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    import streamlit as st
    import pandas as pd
    import numpy as np
except ImportError:
    raise SystemExit("Run: pip install streamlit pandas numpy")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ascent Capital — Operator Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .metric-card {
        background: #1a1a2e; border-radius: 8px; padding: 16px; margin: 4px;
        border: 1px solid #2d2d44;
    }
    .stMetric label { color: #a0a0b0 !important; font-size: 12px !important; }
    [data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: str, n: int = 100) -> list:
    p = Path(path)
    if not p.exists():
        return []
    lines = []
    with open(p) as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except Exception:
                pass
    return lines[-n:]


def _load_json(path: str, default=None):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _get_latest_pnl() -> dict:
    rows = _load_jsonl("logs/us_equities_pnl.jsonl", n=1)
    return rows[-1] if rows else {}


def _nav_series_df() -> pd.DataFrame:
    try:
        from ascent.data.store.timescale import get_nav_series
        df = get_nav_series(lookback_days=90)
        if not df.empty:
            return df
    except Exception:
        pass
    rows = _load_jsonl("logs/us_equities_pnl.jsonl", n=90)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "date": r.get("date"),
        "nav":  r.get("nav"),
        "daily_return": r.get("daily_return"),
        "spy_return":   r.get("spy_return"),
    } for r in rows if r.get("nav")])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def _factor_exposures() -> dict:
    return _load_json("dashboard/factor_exposures.json", {})


def _holdings() -> dict:
    rows = _load_jsonl("logs/holdings_log.jsonl", n=1)
    return rows[-1] if rows else {}


def _skill_scores() -> dict:
    return _load_json("dashboard/agent_skill_scores.json", {})


def _merged_weights() -> dict:
    return _load_json("execution/merged_weights.json", {})


def _skill_log() -> list:
    return _load_jsonl("logs/skill_scores_log.jsonl", n=200)


def _event_trades() -> list:
    return _load_jsonl("logs/event_trades.jsonl", n=20)


# ── Main dashboard ────────────────────────────────────────────────────────────

st.title("Ascent Capital — Operator Dashboard")

last_pnl = _get_latest_pnl()
nav      = last_pnl.get("nav", 0)
dr       = last_pnl.get("daily_return", 0)
spy_r    = last_pnl.get("spy_return", 0)
alpha    = last_pnl.get("alpha_vs_spy", 0)
run_date = last_pnl.get("date", "—")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("NAV", f"${nav:,.0f}" if nav else "—")
with col2:
    st.metric("Daily Return", f"{dr:+.2%}" if dr else "—", delta_color="normal")
with col3:
    st.metric("SPY Return", f"{spy_r:+.2%}" if spy_r else "—")
with col4:
    st.metric("Alpha vs SPY", f"{alpha:+.2%}" if alpha else "—", delta_color="normal")
with col5:
    st.metric("As of", run_date)

st.divider()

tab_port, tab_risk, tab_alpha, tab_events = st.tabs(["Portfolio", "Risk", "Alpha", "Events"])


# ── Portfolio tab ─────────────────────────────────────────────────────────────

with tab_port:
    st.subheader("NAV History")
    nav_df = _nav_series_df()
    if not nav_df.empty and "nav" in nav_df.columns:
        st.line_chart(nav_df[["nav"]])

    st.subheader("Current Weights")
    weights = _merged_weights().get("weights", {})
    if weights:
        wdf = pd.DataFrame(
            [{"Symbol": k, "Weight": f"{v:.1%}"} for k, v in sorted(weights.items(), key=lambda x: -x[1])]
        )
        st.dataframe(wdf, use_container_width=True, hide_index=True)
    else:
        st.info("No merged weights available yet.")

    holdings = _holdings()
    if holdings:
        st.caption(f"Holdings as of: {holdings.get('date', '—')}")


# ── Risk tab ─────────────────────────────────────────────────────────────────

with tab_risk:
    st.subheader("Factor Exposures")
    fe = _factor_exposures()
    if fe:
        fe_df = pd.DataFrame(
            [{"Factor": k, "Exposure (z)": round(v, 3)} for k, v in fe.items() if isinstance(v, (int, float))]
        )
        if not fe_df.empty:
            import plotly.express as px  # type: ignore
            fig = px.bar(fe_df, x="Factor", y="Exposure (z)", color="Exposure (z)",
                         color_continuous_scale="RdBu_r", range_color=[-1.5, 1.5])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Factor exposures will appear after the next daily run.")

    st.subheader("Kill Switch State")
    ks = _load_json("logs/kill_switch_state.json", {})
    if ks:
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Drawdown", f"{ks.get('current_drawdown', 0):.2%}")
        with col_b:
            status = ks.get("state", "unknown")
            color  = "red" if "halt" in status else "green"
            st.markdown(f"State: **:{color}[{status}]**")
    else:
        st.info("Kill switch state not yet written.")

    st.subheader("Alerts (last 10)")
    alerts = _load_jsonl("logs/alerts.jsonl", n=10)
    if alerts:
        for a in reversed(alerts):
            sev = a.get("severity", "INFO")
            icon = "🔴" if sev == "CRITICAL" else "🟡"
            st.write(f"{icon} `{a.get('timestamp','')}` — {a.get('message','')}")
    else:
        st.success("No active alerts.")


# ── Alpha tab ────────────────────────────────────────────────────────────────

with tab_alpha:
    st.subheader("Agent Skill Scores (Trailing 63-day Sharpe)")
    skill = _skill_scores()
    if skill:
        sdf = pd.DataFrame(
            [{"Agent": k, "Sharpe": round(v, 3)} for k, v in skill.items() if isinstance(v, (int, float))]
        )
        st.dataframe(sdf, hide_index=True, use_container_width=True)
    else:
        st.info("Skill scores appear after 63 trading days of live data.")

    st.subheader("Self-Improve Status")
    si_log = _load_jsonl("logs/self_improve_log.jsonl", n=1)
    if si_log:
        last = si_log[-1]
        st.json(last)
    else:
        st.info("Self-improve loop has not run yet (SELF_MODIFY_ENABLED=False until OOS gate met).")

    st.subheader("Slippage IC Feedback")
    sic = _load_jsonl("logs/slippage_ic_log.jsonl", n=5)
    if sic:
        st.dataframe(pd.DataFrame(sic), hide_index=True, use_container_width=True)
    else:
        st.info(f"Slippage IC feedback requires MIN_FILLS=50 (currently warming up).")


# ── Events tab ───────────────────────────────────────────────────────────────

with tab_events:
    st.subheader("Event Agent Feed (last 20)")
    events = _event_trades()
    if events:
        edf = pd.DataFrame(events)
        cols_to_show = [c for c in ["timestamp", "symbol", "event_type", "side", "shares", "status"] if c in edf.columns]
        st.dataframe(edf[cols_to_show] if cols_to_show else edf, hide_index=True, use_container_width=True)
    else:
        st.info("No event trades yet. EVENT_TRADING_ENABLED=False until paper validation (~July 2026).")

    st.subheader("Intraday Adjustments")
    intra = _load_jsonl("logs/intraday_adjustments.jsonl", n=10)
    if intra:
        st.dataframe(pd.DataFrame(intra), hide_index=True, use_container_width=True)
    else:
        st.info("No intraday adjustments triggered.")

# Auto-refresh every 30 seconds
st.caption("Dashboard auto-refreshes every 30s during market hours.")
st.markdown(
    "<script>setTimeout(() => window.location.reload(), 30000)</script>",
    unsafe_allow_html=True,
)
