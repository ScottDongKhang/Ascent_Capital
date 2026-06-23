"""
EDGE TEST #4 (Option A) — micro/small-cap momentum IC, SURVIVORSHIP-BIASED screen.

Extends test #3 to genuinely smaller names than Ascent's 936-name large/mid-cap
universe. Same momentum-IC method as #3, so it is directly comparable.

!! SURVIVORSHIP BIAS !!  The tickers are CURRENTLY listed (yfinance). Micro-caps
that delisted / went to zero are missing, and those are disproportionately the
losers. So this OVERSTATES momentum predictiveness. Read it as an UPPER BOUND:
  - IC ~ 0  -> reasonably solid "no wedge" (bias direction is against this result)
  - IC big  -> INCONCLUSIVE (could just be the bias). Not citable either way.

Candidates are filtered to small/micro by FETCHED market cap (< $2B small,
< $500M micro); anything that fails to fetch or lacks full 2021-2026 history is
dropped automatically.
"""
import sys, json, time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import yfinance as yf

REPO = "/Users/scott/Downloads/ascent capital v2 up to phase 5.1"

# Convenience sample of currently-listed US small/micro-cap tickers across sectors.
# Not a clean universe — survivorship-biased by construction. Invalid/too-large
# names are dropped after fetch.
CANDIDATES = [
    "GEVO","FUBO","CLSK","RIOT","BTBT","SOUN","BBAI","RGTI","QUBT","IONQ",
    "DNA","RKLB","ASTS","ACHR","JOBY","EVGO","CHPT","BLNK","FCEL","PLUG",
    "AMPX","QS","MVIS","OUST","LIDR","INVZ","GOEV","NKLA","RIDE","WKHS",
    "CENN","SOLO","HYLN","XL","FFIE","LCID","RIVN","PSNY","FSR","NIU",
    "BARK","WRBY","CURV","REAL","RENT","TDUP","PRPL","CSPR","LOVE","SFIX",
    "CVNA","VRM","SFT","GRPN","EVER","CARS","YELP","ANGI","TRUE","CARG",
    "PARR","REI","SBOW","VTLE","CRGY","NOG","TALO","GPOR","CIVI","BTU",
    "AMRC","RUN","NOVA","SPWR","MAXN","ARRY","SHLS","FSLR","ENPH","SEDG",
    "BIGC","YEXT","DOMO","SUMO","PD","FROG","AI","UPST","LPRO","OPFI",
    "RDFN","COMP","OPEN","EXPI","DOUG","MAPS","GRWG","SMG","HYFM","IIPR",
    "ATRO","KTOS","AVAV","PL","SPCE","VLD","BKSY","SIDU","RDW","MNTS",
]

SMALL_CAP_MAX = 2e9
MICRO_CAP_MAX = 5e8


def fetch_universe():
    print(f"fetching {len(CANDIDATES)} candidates...", flush=True)
    raw = yf.download(CANDIDATES, start="2020-06-01", end="2026-01-31",
                      auto_adjust=True, progress=False, threads=True)
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    # keep names with near-full history over 2021-2026
    close = close.loc["2020-06-01":"2026-01-31"]
    full = close.columns[close.notna().mean() > 0.95].tolist()
    close = close[full]
    print(f"  {len(full)} have >95% history coverage", flush=True)

    # classify by fetched market cap (best-effort; pool falls back to all survivors)
    caps = {}
    for t in full:
        try:
            fi = yf.Ticker(t).fast_info
            mc = None
            try:
                mc = fi["market_cap"]
            except Exception:
                mc = getattr(fi, "market_cap", None)
            if mc:
                caps[t] = float(mc)
        except Exception:
            pass
        time.sleep(0.05)
    small = [t for t in full if caps.get(t, 1e12) < SMALL_CAP_MAX]
    micro = [t for t in full if caps.get(t, 1e12) < MICRO_CAP_MAX]
    print(f"  market-cap classified: {len(small)} small (<2B), {len(micro)} micro (<500M); "
          f"caps fetched for {len(caps)}/{len(full)}", flush=True)
    return close, small, micro, caps


def momentum_ic(close, names):
    """Same 12-1 momentum IC method as test #3, on the given name pool."""
    px = close[names].sort_index()
    idx = px.index
    points = list(range(273, len(idx) - 22, 21))
    ics = []
    for p in points:
        sig = px.iloc[p - 21] / px.iloc[p - 273] - 1.0
        fwd = px.iloc[p + 21] / px.iloc[p] - 1.0
        panel = pd.DataFrame({"sig": sig, "fwd": fwd}).dropna()
        if len(panel) < 20 or panel["sig"].nunique() < 5:
            continue
        ic, _ = spearmanr(panel["sig"], panel["fwd"])
        if ic == ic:
            ics.append(ic)
    if len(ics) < 5:
        return None
    arr = np.array(ics)
    return {"mean_ic": float(arr.mean()), "std": float(arr.std(ddof=1)),
            "n_months": len(arr), "t_vs_0": float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr))))}


def main():
    close, small, micro, caps = fetch_universe()
    full_pool = list(close.columns)
    out = {"_warning": "SURVIVORSHIP-BIASED (currently-listed only) — upper bound, not citable",
           "n_pool": len(full_pool), "n_small": len(small), "n_micro": len(micro)}
    buckets = [("POOL_all_survivors", full_pool)]
    if len(small) >= 20:
        buckets.append(("SMALL_<2B", small))
    if len(micro) >= 20:
        buckets.append(("MICRO_<500M", micro))
    for label, names in buckets:
        if len(names) < 20:
            out[label] = {"skipped": f"only {len(names)} names, need >=20 for cross-sectional IC"}
            print(f"[{label}] skipped — only {len(names)} names", flush=True)
            continue
        r = momentum_ic(close, names)
        out[label] = r
        print(f"[{label}] {r}", flush=True)

    # compare to large-cap reference from test #3 (high_liq tercile)
    out["_reference_largecap_ic_test3"] = {"mean_ic": 0.0095, "t_vs_0": 0.32}
    sm = out.get("SMALL_<2B") or out.get("POOL_all_survivors", {})
    if isinstance(sm, dict) and "mean_ic" in sm:
        diff = sm["mean_ic"] - 0.0095
        out["_verdict"] = {
            "small_minus_largecap_ic": round(diff, 4),
            "reading": ("INCONCLUSIVE wedge signal (but survivorship bias inflates it — needs clean data)"
                        if diff > 0.02 and sm["t_vs_0"] > 1.5 else
                        "NO WEDGE even with survivorship bias helping — momentum no better in small/micro names"),
        }
        print("VERDICT:", json.dumps(out["_verdict"], indent=2), flush=True)

    open(f"{REPO}/outputs/wf_results/edge_test4_microcap_ic.json", "w").write(json.dumps(out, indent=2))
    print("wrote outputs/wf_results/edge_test4_microcap_ic.json", flush=True)


if __name__ == "__main__":
    main()
