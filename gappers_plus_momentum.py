# streamlit run gappers_plus_momentum.py
import os
import time
import math
import pytz
import numpy as np
import pandas as pd
import datetime as dt
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# Config & utilities
# ─────────────────────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
BASE = "https://api.polygon.io"

API_KEY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY", "")
if not API_KEY:
    st.error("Missing POLYGON_API_KEY. Add it as an env var or Streamlit secret.")
    st.stop()

def poly_get(path, params=None, retries=3, backoff=0.9):
    params = dict(params or {})
    params["apiKey"] = API_KEY
    url = f"{BASE}{path}"
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(backoff * (i + 1))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise last_err if last_err else RuntimeError("Polygon request failed")

def today_et():
    return dt.datetime.now(ET).date()

def prev_business_day(d: dt.date) -> dt.date:
    wd = d.weekday()
    if wd == 0:  # Mon → Fri
        return d - dt.timedelta(days=3)
    if wd == 6:  # Sun → Fri
        return d - dt.timedelta(days=2)
    if wd == 5:  # Sat → Fri
        return d - dt.timedelta(days=1)
    return d - dt.timedelta(days=1)

def session_window(session: str, on_date: dt.date):
    """
    premarket  : 04:00–09:30 ET
    regular    : 09:30–16:00 ET
    afterhours : 16:00–20:00 ET
    """
    if session == "premarket":
        start = ET.localize(dt.datetime(on_date.year, on_date.month, on_date.day, 4, 0))
        end   = ET.localize(dt.datetime(on_date.year, on_date.month, on_date.day, 9, 30))
    elif session == "regular":
        start = ET.localize(dt.datetime(on_date.year, on_date.month, on_date.day, 9, 30))
        end   = ET.localize(dt.datetime(on_date.year, on_date.month, on_date.day, 16, 0))
    else:
        start = ET.localize(dt.datetime(on_date.year, on_date.month, on_date.day, 16, 0))
        end   = ET.localize(dt.datetime(on_date.year, on_date.month, on_date.day, 20, 0))
    return start, end

def detect_session_now():
    now = dt.datetime.now(ET)
    d = now.date()
    pm_s, pm_e = session_window("premarket", d)
    rg_s, rg_e = session_window("regular", d)
    ah_s, ah_e = session_window("afterhours", d)
    if pm_s <= now < pm_e: return "premarket"
    if rg_s <= now < rg_e: return "regular"
    if ah_s <= now < ah_e: return "afterhours"
    return "afterhours"  # outside hours → use last available window (after-hours)

def finviz_link(sym: str) -> str:
    return f"https://finviz.com/quote.ashx?t={sym.upper()}"

# ─────────────────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────────────────
def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ─────────────────────────────────────────────────────────────────────────────
# Data fetch (cache minute bars for the day)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=30)
def get_1m_day(symbol: str, day: dt.date) -> pd.DataFrame:
    dstr = day.strftime("%Y-%m-%d")
    j = poly_get(f"/v2/aggs/ticker/{symbol}/range/1/minute/{dstr}/{dstr}",
                 params={"adjusted":"true","sort":"asc","limit":50000})
    res = j.get("results", [])
    if not res:
        return pd.DataFrame()
    df = pd.DataFrame(res)
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["ts","open","high","low","close","volume"]]

def filter_window(df: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    if df.empty: return df
    if end > dt.datetime.now(ET):
        end = dt.datetime.now(ET)
    return df[(df["ts"] >= start) & (df["ts"] <= end)].copy()

# Resample 1m → 5m/15m (right-closed, aligns to clock)
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if df.empty: return df
    df = df.set_index("ts")
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    out = df.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open","close"])
    out = out.reset_index()
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Snapshot gainers for candidate universe
# ─────────────────────────────────────────────────────────────────────────────
def get_gainers_candidates(limit=60, include_otc=False) -> list[str]:
    syms = set()
    try:
        j = poly_get("/v2/snapshot/locale/us/markets/stocks/gainers", params={"limit": limit})
        for t in j.get("tickers", []):
            sym = t.get("ticker")
            if sym: syms.add(sym)
    except Exception:
        pass
    if include_otc:
        try:
            j = poly_get("/v2/snapshot/locale/us/markets/otc/gainers", params={"limit": max(1,limit//2)})
            for t in j.get("tickers", []):
                sym = t.get("ticker")
                if sym: syms.add(sym)
        except Exception:
            pass
    return sorted(syms)

# ─────────────────────────────────────────────────────────────────────────────
# References for “Gap vs Reference” mode
# ─────────────────────────────────────────────────────────────────────────────
def get_rth_close(symbol: str, day: dt.date) -> float | None:
    dstr = day.strftime("%Y-%m-%d")
    try:
        j = poly_get(f"/v1/open-close/{symbol}/{dstr}", params={"adjusted":"true"})
        if "close" in j and j["close"] not in (None, 0):
            return float(j["close"])
    except Exception:
        return None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Analyses
# ─────────────────────────────────────────────────────────────────────────────
def analyze_gap(symbol: str, sess: str, on_date: dt.date, ref_mode: str):
    """
    Gap vs reference:
      ref_mode ∈ {'prev_close','window_start','today_open'}
    """
    d1m = get_1m_day(symbol, on_date)
    if d1m.empty: return None

    w_start, w_end = session_window(sess, on_date)
    dwin = filter_window(d1m, w_start, w_end)
    if dwin.empty: return None

    # reference price
    ref = None
    if ref_mode == "window_start":
        ref = float(dwin.iloc[0]["open"])
    elif ref_mode == "today_open":
        # first 09:30 bar
        rs, re = session_window("regular", on_date)
        rwin = filter_window(d1m, rs, re)
        if not rwin.empty: ref = float(rwin.iloc[0]["open"])

    if ref is None:
        if sess == "premarket":
            base_day = prev_business_day(on_date)
        elif sess == "afterhours":
            base_day = on_date
        else:
            base_day = prev_business_day(on_date)
        ref = get_rth_close(symbol, base_day)

    if not ref or ref <= 0: return None

    last = float(dwin.iloc[-1]["close"])
    chg = last - ref
    chg_pct = 100.0 * chg / ref
    w_vol = int(dwin["volume"].sum())
    w_open = float(dwin.iloc[0]["open"])
    w_high = float(dwin["high"].max())
    w_low  = float(dwin["low"].min())

    # RSI on window closes
    rsi = float(rsi_wilder(dwin["close"], 14).iloc[-1])

    return {
        "Symbol": symbol,
        "Last": last,
        "%Change": chg_pct,
        "$Change": chg,
        "Open": w_open, "High": w_high, "Low": w_low, "Close": last,
        "Window Vol": w_vol,
        "RSI14": rsi,
        "Finviz": finviz_link(symbol),
    }

def analyze_momentum_vs_sma(symbol: str, on_date: dt.date, sma_len=20):
    """
    Compare last price to SMA(N) of PRIOR bars for:
      - 1m series
      - 5m resample
      - 15m resample
    Works any time (if after 20:00 ET, uses latest available intraday bar).
    """
    d1m = get_1m_day(symbol, on_date)
    if d1m.empty or len(d1m) < sma_len + 1:
        return None

    # Use bars from 04:00 to min(now, 20:00)
    start = ET.localize(dt.datetime(on_date.year,on_date.month,on_date.day,4,0))
    end   = ET.localize(dt.datetime(on_date.year,on_date.month,on_date.day,20,0))
    if dt.datetime.now(ET) < end:
        end = dt.datetime.now(ET)
    d1m = filter_window(d1m, start, end)

    if d1m.empty or len(d1m) < sma_len + 1:
        return None

    last_px = float(d1m.iloc[-1]["close"])
    last_vol = int(d1m.iloc[-1]["volume"])

    # 1m SMA edge
    s1 = d1m["close"].astype(float)
    sma1 = s1.shift(1).rolling(sma_len, min_periods=sma_len).mean()  # prior N
    edge1 = None if math.isnan(sma1.iloc[-1]) or sma1.iloc[-1]==0 else 100.0*(last_px/sma1.iloc[-1]-1)

    # 5m/15m resamples
    d5m  = resample_ohlcv(d1m, "5T")
    d15m = resample_ohlcv(d1m, "15T")

    edge5 = edge15 = None
    if len(d5m) >= sma_len + 1:
        s5 = d5m["close"].astype(float)
        sma5 = s5.shift(1).rolling(sma_len, min_periods=sma_len).mean()
        edge5 = None if math.isnan(sma5.iloc[-1]) or sma5.iloc[-1]==0 else 100.0*(float(s5.iloc[-1])/sma5.iloc[-1]-1)

    if len(d15m) >= sma_len + 1:
        s15 = d15m["close"].astype(float)
        sma15 = s15.shift(1).rolling(sma_len, min_periods=sma_len).mean()
        edge15 = None if math.isnan(sma15.iloc[-1]) or sma15.iloc[-1]==0 else 100.0*(float(s15.iloc[-1])/sma15.iloc[-1]-1)

    # 1m RVOL: last 1m volume / SMA(20) of prior 1m vols
    v_sma20 = d1m["volume"].astype(float).shift(1).rolling(20, min_periods=20).mean().iloc[-1]
    rvol1m = None if (v_sma20 is None or math.isnan(v_sma20) or v_sma20==0) else (last_vol / v_sma20)

    # RSI on 1m
    rsi = float(rsi_wilder(d1m["close"], 14).iloc[-1])

    return {
        "Symbol": symbol,
        "Last": last_px,
        "Edge 1m vs SMA": edge1,
        "Edge 5m vs SMA": edge5,
        "Edge 15m vs SMA": edge15,
        "RVOL(1m)": rvol1m,
        "RSI14": rsi,
        "1m Vol": last_vol,
        "Finviz": finviz_link(symbol),
    }

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gap & Momentum Screener (Polygon)", page_icon="⚡", layout="wide")
st.title("⚡ Gap & Momentum Screener (Polygon)")

with st.sidebar:
    st.subheader("Universe")
    use_snapshot = st.checkbox("Use Snapshot Gainers", value=True)
    gainers_limit = st.slider("Gainers limit", 20, 200, 80, step=20)
    include_otc = st.checkbox("Include OTC gainers", value=False)
    watchlist = st.text_area("Add symbols (comma-separated)", "AAPL, NVDA, TSLA, AMD, MSFT, META, AMZN, SMCI, AVGO, GOOGL", height=80)

    st.subheader("Filters (common)")
    min_price = st.number_input("Min Last Price ($)", value=5.0, step=0.5, min_value=0.0)
    min_window_vol = st.number_input("Min Window Volume (Gap mode)", value=2_000_000, step=100_000, min_value=0)
    rsi_enable = st.checkbox("Require RSI band", value=True)
    rsi_min, rsi_max = st.slider("RSI(14) range", 0, 100, (60, 80))

    st.subheader("Auto-refresh")
    auto = st.checkbox("Auto-refresh every ~20s", value=False)

tabs = st.tabs(["Gap vs Reference", "Momentum vs SMA (1/5/15)"])

# Build candidate list
candidates = []
if use_snapshot:
    candidates.extend(get_gainers_candidates(limit=gainers_limit, include_otc=include_otc))
if watchlist.strip():
    more = [s.strip().upper() for s in watchlist.split(",") if s.strip()]
    candidates.extend(more)
candidates = sorted(set(candidates))
if not candidates:
    st.warning("No candidates. Enable snapshot gainers or add watchlist symbols.")
    st.stop()

if auto:
    # Light auto-refresh by re-running script (Streamlit Cloud safe rate if universe modest)
    st.experimental_rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Gap vs Reference
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    left, mid, right = st.columns([1,1,1])
    with left:
        sess_choice = st.selectbox("Session", ["Auto (detect)","Pre-Market","Regular","After-Hours"], index=0)
        sess = detect_session_now() if sess_choice=="Auto (detect)" else \
               {"Pre-Market":"premarket","Regular":"regular","After-Hours":"afterhours"}[sess_choice]
    with mid:
        ref_mode_label = st.selectbox("Reference Price", ["Prev RTH Close","Window Start","Today 09:30 Open"], index=0)
        ref_mode = {"Prev RTH Close":"prev_close","Window Start":"window_start","Today 09:30 Open":"today_open"}[ref_mode_label]
    with right:
        topn_gap = st.slider("Show Top N (by %Change)", 10, 200, 60)

    d = today_et()
    ws, we = session_window(sess, d)
    st.caption(f"**Now (ET):** {dt.datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} • Session Window: {ws.strftime('%H:%M')}–{we.strftime('%H:%M')}")

    rows = []
    skipped = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(analyze_gap, s, sess, d, ref_mode): s for s in candidates}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r: rows.append(r)
            except Exception:
                skipped += 1

    if not rows:
        st.info("No rows matched (quiet window or strict filters). Try another session or relax filters.")
    else:
        df = pd.DataFrame(rows)
        # apply filters
        df = df[df["Last"] >= float(min_price)]
        df = df[df["Window Vol"] >= int(min_window_vol)]
        if rsi_enable:
            df = df[df["RSI14"].between(rsi_min, rsi_max, inclusive="both")]
        if df.empty:
            st.info("Rows found, but none passed your filters.")
        else:
            df = df.sort_values("%Change", ascending=False).head(int(topn_gap)).reset_index(drop=True)
            df.insert(0, "Rank", df.index + 1)
            # formatting
            df["%Change"] = df["%Change"].map(lambda x: f"{x:,.2f}%")
            for k in ["Last","$Change","Open","High","Low","Close"]:
                df[k] = df[k].map(lambda x: f"{x:,.2f}")
            df["Window Vol"] = df["Window Vol"].map(lambda x: f"{int(x):,}")
            # link
            df["Finviz"] = df["Finviz"].map(lambda u: f'<a href="{u}" target="_blank">Open</a>')

            st.subheader("Gap vs Reference — Results")
            st.write(
                df[["Rank","Symbol","Last","%Change","$Change","Open","High","Low","Close","Window Vol","RSI14","Finviz"]]
                  .to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
    if skipped:
        st.caption(f"Skipped {skipped} symbols due to missing/invalid data.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Momentum vs SMA (1/5/15)
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        sma_len = st.number_input("SMA length (periods)", value=20, min_value=5, max_value=200, step=1)
    with c2:
        min_edge = st.number_input("Min edge vs SMA (%) — any of 1/5/15m", value=0.5, step=0.1)
    with c3:
        min_rvol = st.number_input("Min RVOL(1m) vs SMA20", value=1.50, step=0.1)

    topn_mom = st.slider("Show Top N (by max edge)", 10, 200, 60)

    rows2 = []
    skipped2 = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(analyze_momentum_vs_sma, s, today_et(), sma_len): s for s in candidates}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r: rows2.append(r)
            except Exception:
                skipped2 += 1

    if not rows2:
        st.info("No data for momentum/SMA (maybe too few bars yet).")
    else:
        df2 = pd.DataFrame(rows2)
        # filters: price, rvol, rsi band optional, edge threshold
        df2 = df2[df2["Last"] >= float(min_price)]
        df2 = df2[(df2["RVOL(1m)"].fillna(0) >= float(min_rvol))]
        if rsi_enable:
            df2 = df2[df2["RSI14"].between(rsi_min, rsi_max, inclusive="both")]

        # compute max edge across 1/5/15
        df2["Max Edge %"] = df2[["Edge 1m vs SMA","Edge 5m vs SMA","Edge 15m vs SMA"]].max(axis=1, skipna=True)
        df2 = df2[df2["Max Edge %"].fillna(-1e9) >= float(min_edge)]

        if df2.empty:
            st.info("Rows found, but none passed edge/RVOL/RSI filters.")
        else:
            df2 = df2.sort_values(["Max Edge %","RVOL(1m)"], ascending=[False, False]).head(int(topn_mom)).reset_index(drop=True)
            df2.insert(0, "Rank", df2.index + 1)

            # format
            pct_cols = ["Edge 1m vs SMA","Edge 5m vs SMA","Edge 15m vs SMA","Max Edge %"]
            for c in pct_cols:
                df2[c] = df2[c].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}%")
            df2["Last"]     = df2["Last"].map(lambda x: f"{x:,.2f}")
            df2["RVOL(1m)"] = df2["RVOL(1m)"].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")
            df2["RSI14"]    = df2["RSI14"].map(lambda x: f"{x:,.1f}" if not pd.isna(x) else "")

            df2["Finviz"] = df2["Finviz"].map(lambda u: f'<a href="{u}" target="_blank">Open</a>')

            st.subheader("Momentum vs SMA — Results")
            st.write(
                df2[["Rank","Symbol","Last","Edge 1m vs SMA","Edge 5m vs SMA","Edge 15m vs SMA","Max Edge %","RVOL(1m)","RSI14","Finviz"]]
                   .to_html(escape=False, index=False),
                unsafe_allow_html=True
            )

    if skipped2:
        st.caption(f"Skipped {skipped2} symbols due to missing/invalid data.")
