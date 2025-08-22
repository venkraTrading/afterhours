# movers_unified.py
# Auto-session movers (Polygon) with adaptive filters + progressive fallback for pre/after-hours.

import os
import math
from datetime import datetime, date, time as dtime, timezone, timedelta
from zoneinfo import ZoneInfo  # proper DST handling

import requests
import numpy as np
import pandas as pd
import streamlit as st

API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    st.error("Missing POLYGON_API_KEY environment variable.")
    st.stop()

# Proper Eastern Time with DST
ET = ZoneInfo("America/New_York")

# Session windows (ET)
SESSIONS = {
    "Pre-Market":  (dtime(4, 0),  dtime(9, 30)),
    "Regular":     (dtime(9, 30), dtime(16, 0)),
    "After-Hours": (dtime(16, 0), dtime(20, 0)),
}

def now_et():
    return datetime.now(tz=ET)

def pick_session_for_now(dt_et: datetime):
    t = dt_et.timetz()
    for label, (s, e) in SESSIONS.items():
        if s <= t.replace(tzinfo=None) <= e:
            return label
    # If not inside any window, pick the nearest upcoming one today; else default to After-Hours
    if t < SESSIONS["Pre-Market"][0]:
        return "Pre-Market"
    elif t < SESSIONS["Regular"][0]:
        return "Regular"
    elif t < SESSIONS["After-Hours"][0]:
        return "After-Hours"
    else:
        # past 20:00 — default to After-Hours window for today
        return "After-Hours"

@st.cache_data(ttl=120, show_spinner=False)
def polygon_get(url, params=None):
    params = params or {}
    params["apiKey"] = API_KEY
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def to_et(ts_ns_or_ms):
    if ts_ns_or_ms is None:
        return None
    try:
        t = int(ts_ns_or_ms)
    except Exception:
        return None
    if t > 10**13:
        dt_utc = datetime.fromtimestamp(t / 1e9, tz=timezone.utc)
    else:
        dt_utc = datetime.fromtimestamp(t / 1e3, tz=timezone.utc)
    return dt_utc.astimezone(ET)

def event_in_window(rec, start_et, end_et):
    lt = rec.get("lastTrade") or {}
    lq = rec.get("lastQuote") or {}

    lt_et = to_et(lt.get("t"))
    lq_et = to_et(lq.get("t"))

    if lt_et and start_et <= lt_et <= end_et:
        return True, lt_et, "trade"
    if lq_et and start_et <= lq_et <= end_et:
        return True, lq_et, "quote"
    return False, (lt_et or lq_et), None

@st.cache_data(ttl=180, show_spinner=False)
def fetch_snapshots(include_otc: bool):
    base = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    results, params, next_url = [], {"limit": 25000}, base
    if include_otc:
        params["include_otc"] = "true"
    while next_url:
        data = polygon_get(next_url, params=params)
        results.extend(data.get("tickers") or [])
        next_url, params = data.get("next_url"), {}
        if len(results) > 500_000:
            break
    return results

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sic(ticker: str):
    url = "https://api.polygon.io/v3/reference/tickers"
    data = polygon_get(url, {"ticker": ticker, "limit": 1})
    res = (data.get("results") or [])
    if not res:
        return None
    r = res[0]
    return r.get("sic_description") or r.get("description") or r.get("name")

def finviz_link(sym): 
    return f"[{sym}](https://finviz.com/quote.ashx?t={sym})"

def build_df(raw, start_et, end_et, min_price, min_vol):
    rows = []
    for rec in raw:
        sym = rec.get("ticker")
        if not sym:
            continue
        prev_close = (rec.get("prevDay") or {}).get("c")
        if prev_close is None or prev_close < min_price:
            continue
        day_vol = (rec.get("day") or {}).get("v")
        if day_vol is None or day_vol < min_vol:
            continue

        last_trade = rec.get("lastTrade") or {}
        last_quote = rec.get("lastQuote") or {}
        price = last_trade.get("p")
        if price is None:
            bp, ap = last_quote.get("bp"), last_quote.get("ap")
            price = ((bp + ap) / 2.0) if (bp is not None and ap is not None) else (ap if ap is not None else bp)

        in_sess, et_time, src = event_in_window(rec, start_et, end_et)
        if not in_sess:
            continue

        chg = np.nan
        chg_pct = np.nan
        if price is not None and prev_close not in (None, 0):
            chg = float(price) - float(prev_close)
            chg_pct = (chg / float(prev_close)) * 100.0

        rows.append({
            "Symbol": sym,
            "Price": price,
            "Prev Close": prev_close,
            "CHG": chg,
            "CHG %": chg_pct,
            "Volume": day_vol,
            "Last (ET)": et_time.strftime("%H:%M:%S") if et_time else "",
            "Src": src or "",
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        want = list(df["Symbol"].unique())[:400]
        meta = {s: fetch_sic(s) for s in want}
        df["Sector / SIC"] = df["Symbol"].map(meta).fillna("—")
    return df

def fmt_table(df: pd.DataFrame):
    out = df.copy()
    out.insert(0, " ", [finviz_link(s) for s in out["Symbol"]])
    out["Price"] = out["Price"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")
    out["Prev Close"] = out["Prev Close"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")
    out["CHG"] = out["CHG"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    out["CHG %"] = out["CHG %"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    out["Volume"] = out["Volume"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
    return out

# ───────────────── UI ─────────────────
st.set_page_config(page_title="Unified Movers (Auto Session)", page_icon="⚡", layout="wide")
st.title("⚡ Unified Movers — Auto Session (Polygon)")

with st.sidebar:
    st.subheader("Options")
    include_otc = st.checkbox("Include OTC", value=True)  # enable by default for pre/after breadth
    top_n = st.slider("Show Top N by |% Change|", 10, 200, 60, 5)
    if st.button("Refresh"):
        fetch_snapshots.clear()
        fetch_sic.clear()

now = now_et()
session_label = pick_session_for_now(now)
start_t, end_t = SESSIONS[session_label]
scan_date = now.date()  # always today

start_et = datetime.combine(scan_date, start_t, tzinfo=ET)
end_et   = datetime.combine(scan_date, end_t,   tzinfo=ET)

# Adaptive defaults
if session_label == "Regular":
    min_price_default = 5.0
    min_vol_default   = 2_000_000
else:
    min_price_default = 1.0
    min_vol_default   = 100_000

st.markdown(
    f"**Now (ET):** {now.strftime('%Y-%m-%d %H:%M:%S')}  •  "
    f"**Session:** {session_label}  •  "
    f"**Window:** {start_et.strftime('%H:%M')}–{end_et.strftime('%H:%M')} ET"
)

# Fetch & build with progressive fallback
with st.spinner("Fetching snapshots…"):
    raw = fetch_snapshots(include_otc)

def build_with(min_price, min_vol):
    df = build_df(raw, start_et, end_et, min_price, min_vol)
    df = df.sort_values("CHG %", ascending=False, na_position="last")
    leaders = df[df["CHG %"] > 0].head(int(top_n))
    laggards = df[df["CHG %"] < 0].tail(int(top_n))
    return df, leaders, laggards

# Try default → relax volume if empty
df_all, leaders, laggards = build_with(min_price_default, min_vol_default)
if df_all.empty:
    df_all, leaders, laggards = build_with(min_price_default, max(50_000, min_vol_default // 2))
if df_all.empty:
    df_all, leaders, laggards = build_with(min_price_default, 0)

if df_all.empty:
    st.warning("Still no rows. It might be a weekend/holiday or too early (<04:00 ET). "
               "Try enabling OTC (sidebar) or check back a bit later.")
    st.stop()

# Display
c1, c2 = st.columns(2)
with c1:
    st.subheader("Leaders")
    st.dataframe(
        fmt_table(leaders)[[" ", "Symbol", "Sector / SIC", "Price", "Prev Close", "CHG", "CHG %", "Volume", "Last (ET)", "Src"]],
        use_container_width=True, height=600, hide_index=True,
    )
with c2:
    st.subheader("Laggards")
    st.dataframe(
        fmt_table(laggards)[[" ", "Symbol", "Sector / SIC", "Price", "Prev Close", "CHG", "CHG %", "Volume", "Last (ET)", "Src"]],
        use_container_width=True, height=600, hide_index=True,
    )

st.caption("If the table was empty at first, the app relaxed volume filters automatically for pre/after-hours.")
