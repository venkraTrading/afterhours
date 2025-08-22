#!/usr/bin/env python3
# afterhours_movers_live.py
import os
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, timezone
import pytz
import streamlit as st

# ─────────────────────────── Setup ───────────────────────────
st.set_page_config(page_title="After/Pre-Market Movers (Live)", page_icon="📈", layout="wide")

API_KEY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY")
if not API_KEY:
    st.error("Missing POLYGON_API_KEY. Add it under **Settings → Secrets** or export it in the environment.")
    st.stop()

ET = pytz.timezone("America/New_York")
SNAPSHOT_API = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"

# ───────────────────────── Controls ──────────────────────────
st.title("📈 Live Movers — Pre-Market / Regular / After-Hours")

with st.sidebar:
    st.subheader("Session & Options")
    date_et = st.date_input("Trading date (ET)", value=datetime.now(ET).date())
    session = st.radio(
        "Session",
        ["Pre-Market (04:00–09:30)", "Regular (09:30–16:00)", "After-Hours (16:00–20:00)"],
        index=2,
    )
    top_n = st.slider("Top N (leaders / laggards)", 10, 100, 50, step=10)
    include_otc = st.toggle("Include OTC", value=False, help="Snapshots can include OTC; usually keep disabled.")
    min_prev_close = st.number_input("Min previous close ($)", value=1.0, step=0.5, help="Filter out pennies if you wish.")
    only_has_trade = st.toggle("Require a trade in the chosen session", value=True)

# ───────────────────── Helpers & Time Window ─────────────────
def et_dt(y, m, d, hh, mm, ss=0):
    return ET.localize(datetime(y, m, d, hh, mm, ss))

def session_window(et_date, which: str):
    """Return (start_et, end_et) datetimes for session on a given ET date."""
    y, m, d = et_date.year, et_date.month, et_date.day
    if which.startswith("Pre-Market"):
        return et_dt(y, m, d, 4, 0), et_dt(y, m, d, 9, 30)
    if which.startswith("Regular"):
        return et_dt(y, m, d, 9, 30), et_dt(y, m, d, 16, 0)
    # After-Hours
    return et_dt(y, m, d, 16, 0), et_dt(y, m, d, 20, 0)

start_et, end_et = session_window(date_et, session)

def to_et(ts_ns_or_ms):
    """
    Polygon snapshot lastTrade.t is in UNIX ns (nanoseconds). Some environments may
    return ms. We'll detect by magnitude.
    """
    if ts_ns_or_ms is None:
        return None
    try:
        t = int(ts_ns_or_ms)
        # Heuristic: if timestamp is > 10^13, treat as ns; if ~10^12-10^13, also ns; else ms.
        if t > 10**13:
            dt_utc = datetime.fromtimestamp(t / 1e9, tz=timezone.utc)
        else:
            # treat as ms
            dt_utc = datetime.fromtimestamp(t / 1e3, tz=timezone.utc)
        return dt_utc.astimezone(ET)
    except Exception:
        return None

def within_session(et_dt_obj, start_et, end_et):
    if et_dt_obj is None:
        return False
    return start_et <= et_dt_obj <= end_et

# ───────────────────────── Fetching ─────────────────────────
@st.cache_data(ttl=30)  # small TTL to avoid hammering; update every reload/refresh button
def fetch_snapshots(include_otc: bool):
    """Paginate through all ticker snapshots."""
    params = {"limit": 1000, "apiKey": API_KEY}
    if include_otc:
        params["include_otc"] = "true"

    out = []
    next_url = SNAPSHOT_API
    pages = 0
    while next_url and pages < 50:  # hard safety
        pages += 1
        r = requests.get(next_url, params=params if pages == 1 else None, timeout=30)
        r.raise_for_status()
        data = r.json()
        tickers = data.get("tickers", []) or []
        out.extend(tickers)
        next_url = data.get("next_url")
        if next_url and "apiKey=" not in next_url:
            sep = "&" if "?" in next_url else "?"
            next_url = f"{next_url}{sep}apiKey={API_KEY}"
    return out

with st.spinner("Fetching snapshots…"):
    try:
        raw = fetch_snapshots(include_otc)
    except requests.HTTPError as e:
        st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
        st.stop()
    except Exception as e:
        st.error(f"Fetch error: {e}")
        st.stop()

# ───────────────────── Build DataFrame safely ───────────────
rows = []
for rec in raw:
    sym = rec.get("ticker")
    if not sym:
        continue

    last_trade = (rec.get("lastTrade") or {})
    last_px = last_trade.get("p")
    last_ts = last_trade.get("t")
    last_et = to_et(last_ts)

    prev_day = rec.get("prevDay") or {}
    prev_close = prev_day.get("c")  # previous RTH close

    # (Optional) day volume (RTH). Not after-hours volume, but still useful for context.
    day = rec.get("day") or {}
    day_vol = day.get("v")

    # Skip if previous close missing or tiny
    if prev_close is None or (min_prev_close and (prev_close < float(min_prev_close))):
        continue

    # If requiring an actual trade in the session, enforce time window
    if only_has_trade and not within_session(last_et, start_et, end_et):
        continue

    # Compute change & pct
    price = last_px if last_px is not None else np.nan
    chg = np.nan
    chg_pct = np.nan
    if price == price and prev_close:  # price not NaN and prev_close not 0/None
        chg = price - prev_close
        if prev_close != 0:
            chg_pct = (chg / prev_close) * 100.0

    rows.append({
        "Symbol": sym,
        "Price": price,
        "Ref Close": prev_close,
        "CHG": chg,
        "CHG %": chg_pct,
        "Volume": day_vol,
        "Last Trade (ET)": last_et.strftime("%H:%M:%S") if last_et else "",
    })

df_all = pd.DataFrame(rows)

# ─────────── Defensive guards (fix KeyError: 'CHG %') ───────
if df_all is None or len(df_all) == 0:
    st.info("No rows matched your filters/time window. Try a different session, enable OTC, or relax the constraints.")
    st.stop()

# Normalize column names and types
df_all.columns = [str(c).strip() for c in df_all.columns]

for col in ["Price", "Ref Close", "CHG", "CHG %", "Volume"]:
    if col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

# Recompute CHG % if missing / NaN
def compute_chg_pct(df):
    if {"Price", "Ref Close"}.issubset(df.columns):
        prev = df["Ref Close"]
        cur  = df["Price"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return (cur - prev) / prev * 100.0
    return pd.Series(np.nan, index=df.index)

if "CHG %" not in df_all.columns or df_all["CHG %"].isna().all():
    df_all["CHG %"] = compute_chg_pct(df_all)

if "CHG %" not in df_all.columns or df_all["CHG %"].dropna().empty:
    st.info("Couldn’t compute % change (missing price fields).")
    st.dataframe(df_all)
    st.stop()

# ───────────────────── Leaders / Laggards ───────────────────
leaders  = df_all[df_all["CHG %"] > 0].sort_values("CHG %", ascending=False).head(int(top_n))
laggards = df_all[df_all["CHG %"] < 0].sort_values("CHG %", ascending=True ).head(int(top_n))

# Display
left, right = st.columns(2)
display_cols = [c for c in ["Symbol", "Price", "Ref Close", "CHG", "CHG %", "Volume", "Last Trade (ET)"] if c in df_all.columns]

with left:
    st.subheader("🏆 Leaders")
    st.dataframe(
        leaders[display_cols].style.format({
            "Price": "{:.2f}",
            "Ref Close": "{:.2f}",
            "CHG": "{:+.2f}",
            "CHG %": "{:+.2f}",
            "Volume": lambda v: f"{int(v):,}" if pd.notna(v) else ""
        }).background_gradient(subset=["CHG %"], cmap="Greens"),
        use_container_width=True,
        hide_index=True
    )

with right:
    st.subheader("⚠️ Laggards")
    st.dataframe(
        laggards[display_cols].style.format({
            "Price": "{:.2f}",
            "Ref Close": "{:.2f}",
            "CHG": "{:+.2f}",
            "CHG %": "{:+.2f}",
            "Volume": lambda v: f"{int(v):,}" if pd.notna(v) else ""
        }).background_gradient(subset=["CHG %"], cmap="Reds"),
        use_container_width=True,
        hide_index=True
    )

# Footer / hint
st.caption(
    "Note: Snapshot’s `day.v` is regular-hours volume, not extended-hours volume. "
    "We filter movers by the **time of the last trade** within the selected session."
)
