# movers_unified.py
# Streamlit app: Pre/Regular/After-hours movers (Polygon)
#
# Features:
# - Session selector (Pre-Market / Regular / After-Hours)
# - Leaders & Laggards tables by % change vs prior close
# - Finviz link for every symbol
# - Optional sector/SIC via Polygon reference endpoint (best-effort)
# - Defaults: Min price = $5, Min volume = 2,000,000
# - OTC toggle
# - Robust session gating: prefer last trade in window; fall back to last quote

import os
import math
import time
from datetime import datetime, date, time as dtime, timezone, timedelta

import requests
import numpy as np
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    st.error("Missing POLYGON_API_KEY environment variable.")
    st.stop()

ET = timezone(timedelta(hours=-5))  # Eastern without DST knowledge; for cloud simplicity
# If you want more accurate ET with DST, install "pytz" and use: ET = pytz.timezone("US/Eastern")

SESSION_DEFS = {
    "Pre-Market (04:00–09:30 ET)": (dtime(4, 0), dtime(9, 30)),
    "Regular (09:30–16:00 ET)":   (dtime(9, 30), dtime(16, 0)),
    "After-Hours (16:00–20:00 ET)": (dtime(16, 0), dtime(20, 0)),
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=120, show_spinner=False)
def polygon_get(url, params=None):
    params = params or {}
    params["apiKey"] = API_KEY
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def to_et(ts_ns_or_ms):
    """Convert Polygon ns/ms epoch to ET datetime (or None)."""
    if ts_ns_or_ms is None:
        return None
    try:
        t = int(ts_ns_or_ms)
    except Exception:
        return None
    # Heuristic: ns vs ms
    if t > 10**13:
        dt_utc = datetime.fromtimestamp(t / 1e9, tz=timezone.utc)
    else:
        dt_utc = datetime.fromtimestamp(t / 1e3, tz=timezone.utc)
    return dt_utc.astimezone(ET)

def event_in_session(rec, start_et, end_et):
    """
    Return (in_session: bool, et_time: datetime|None, src: 'trade'|'quote'|None).
    Prefer lastTrade if inside the window; else try lastQuote.
    """
    lt = rec.get("lastTrade") or {}
    lq = rec.get("lastQuote") or {}

    lt_et = to_et(lt.get("t"))
    lq_et = to_et(lq.get("t"))

    if lt_et and start_et <= lt_et <= end_et:
        return True, lt_et, "trade"
    if lq_et and start_et <= lq_et <= end_et:
        return True, lq_et, "quote"
    # return the freshest time we have for display even if out of session
    return False, (lt_et or lq_et), None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_snapshots(include_otc: bool):
    """
    Pull all US stock snapshots (paginate). Returns list of snapshot dicts.
    This is a heavy call; we cache a few minutes.
    """
    base = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    results = []
    params = {"limit": 25000}
    if include_otc:
        params["include_otc"] = "true"
    next_url = base

    while next_url:
        data = polygon_get(next_url, params=params)
        results.extend(data.get("tickers", []) or [])
        next_url = data.get("next_url")  # already includes apiKey? No—Polygon omits it.
        params = {}  # for next_url we must pass no params except apiKey (added in polygon_get)

        # Safety guard if API ever loops
        if len(results) > 500_000:
            break

    return results

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_sic_description(ticker: str):
    """
    Best-effort sector/industry label using Polygon reference endpoint (SIC description).
    Cached per symbol to keep calls light.
    """
    url = "https://api.polygon.io/v3/reference/tickers"
    data = polygon_get(url, {"ticker": ticker, "limit": 1})
    results = data.get("results") or []
    if not results:
        return None
    # Prefer SIC description or industry-like field if present
    res = results[0]
    # Polygon commonly returns 'sic_description'. Fallbacks just in case.
    return res.get("sic_description") or res.get("description") or res.get("name")

def build_dataframe(raw, start_et, end_et, require_trade_in_session, min_ref_close, min_day_volume):
    rows = []
    for rec in raw:
        sym = rec.get("ticker")
        if not sym:
            continue

        prev = rec.get("prevDay") or {}
        prev_close = prev.get("c")
        if prev_close is None:
            continue
        if prev_close < min_ref_close:
            continue

        day = rec.get("day") or {}
        day_vol = day.get("v")  # today’s running volume (RTH mainly; Polygon accumulates over day)
        if day_vol is None or day_vol < min_day_volume:
            continue

        # Price: prefer last trade price; else quote mid; else ask/bid if one is missing
        last_trade = rec.get("lastTrade") or {}
        last_quote = rec.get("lastQuote") or {}
        price = last_trade.get("p")
        if price is None:
            bp, ap = last_quote.get("bp"), last_quote.get("ap")
            if bp is not None and ap is not None:
                price = (bp + ap) / 2.0
            else:
                price = ap if ap is not None else bp

        # Session inclusion
        in_sess, et_time, src = event_in_session(rec, start_et, end_et)
        if require_trade_in_session:
            lt_et = to_et(last_trade.get("t"))
            if not (lt_et and start_et <= lt_et <= end_et):
                continue
        else:
            if not in_sess:
                continue

        # Computes change vs prior close
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
        # Optional sector/SIC; do in a batched, cached way to limit requests
        # Only enrich top-N * 2 to save time; user can request more if needed
        want = list(df["Symbol"].unique())[:400]
        meta = {sym: fetch_sic_description(sym) for sym in want}
        df["Sector / SIC"] = df["Symbol"].map(meta).fillna("—")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Unified Movers (Polygon)", page_icon="📈", layout="wide")
st.title("📈 Unified Movers — Pre / Regular / After-Hours (Polygon)")

with st.sidebar:
    st.subheader("Scan Settings")

    session_label = st.selectbox(
        "Session (ET)",
        list(SESSION_DEFS.keys()),
        index=2  # default After-Hours
    )
    d = st.date_input("Date (ET)", value=date.today())

    # Filters
    st.subheader("Filters")
    include_otc = st.checkbox("Include OTC", value=False)
    require_trade = st.checkbox("Require a trade in the session", value=False)

    min_price = st.number_input("Min price (prior close) $", value=5.0, step=0.5, min_value=0.0)
    min_vol = st.number_input("Min volume (today) shares", value=2_000_000, step=100_000, min_value=0)

    top_n = st.slider("Show Top N by |% Change|", min_value=10, max_value=200, value=60, step=5)

    # Manual refresh (avoid experimental autorefresh)
    do_refresh = st.button("🔄 Refresh now")

# Compute ET window
start_t, end_t = SESSION_DEFS[session_label]
start_et = datetime.combine(d, start_t, ET)
end_et = datetime.combine(d, end_t, ET)

subtitle = f"Results — {start_et.strftime('%Y-%m-%d %H:%M')}–{end_et.strftime('%H:%M')} ET"
st.markdown(f"**{subtitle}**")

# Fetch data
with st.spinner("Fetching Polygon snapshots…"):
    if do_refresh:
        # bust caches
        fetch_snapshots.clear()
        fetch_sic_description.clear()

    raw = fetch_snapshots(include_otc)
    df_all = build_dataframe(
        raw=raw,
        start_et=start_et,
        end_et=end_et,
        require_trade_in_session=require_trade,
        min_ref_close=float(min_price),
        min_day_volume=int(min_vol),
    )

# Guard for empty
if df_all.empty:
    st.warning("No rows matched your filters/time window. Try another session, enable OTC, or relax constraints.")
    st.stop()

# Rank, Leaders, Laggards
df_all = df_all.sort_values("CHG %", ascending=False, na_position="last")
leaders = df_all[df_all["CHG %"] > 0].head(int(top_n)).copy()
laggards = df_all[df_all["CHG %"] < 0].tail(int(top_n)).copy()

def finviz_link(sym):  # helper to make markdown link
    return f"[{sym}](https://finviz.com/quote.ashx?t={sym})"

def fmt_table(df: pd.DataFrame):
    if df.empty:
        return df
    out = df.copy()
    out.insert(0, " ", [finviz_link(s) for s in out["Symbol"]])
    out["Price"] = out["Price"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")
    out["Prev Close"] = out["Prev Close"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "—")
    out["CHG"] = out["CHG"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    out["CHG %"] = out["CHG %"].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    out["Volume"] = out["Volume"].map(lambda v: f"{int(v):,}" if pd.notna(v) else "—")
    return out

c1, c2 = st.columns(2)

with c1:
    st.subheader("Leaders")
    st.dataframe(
        fmt_table(leaders)[[" ", "Symbol", "Sector / SIC", "Price", "Prev Close", "CHG", "CHG %", "Volume", "Last (ET)", "Src"]],
        use_container_width=True,
        height=600,
        hide_index=True,
    )

with c2:
    st.subheader("Laggards")
    st.dataframe(
        fmt_table(laggards)[[" ", "Symbol", "Sector / SIC", "Price", "Prev Close", "CHG", "CHG %", "Volume", "Last (ET)", "Src"]],
        use_container_width=True,
        height=600,
        hide_index=True,
    )

st.caption("Tip: Click the ticker icon to open Finviz in a new tab. "
           "‘Src’ shows whether the match came from a trade or a quote within the selected window.")
