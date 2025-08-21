#!/usr/bin/env python3
import os, time
from datetime import datetime, time as dtime, timezone
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st
from pytz import timezone as pytz_tz

# ── Setup ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Extended Hours Movers — Sector + Finviz", page_icon="📈", layout="wide")

API_KEY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY")
if not API_KEY:
    st.error("POLYGON_API_KEY not found. In Streamlit Cloud: Settings → Secrets.")
    st.stop()

NY = pytz_tz("America/New_York")
TITLE = "📈 Extended Hours Movers — Sector + Finviz"
st.title(TITLE)

# ── Session toggle & controls ────────────────────────────────────────────────
mode = st.radio(
    "Session",
    ["Pre-Market (04:00–09:30 ET)", "After-Hours (≥ 16:00 ET)"],
    horizontal=True,
    index=1,
)

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    min_price = st.number_input("Min last price ($)", 0.0, 1e5, 2.0, 0.5)
with c2:
    min_vol = st.number_input("Min RTH day volume (shares)", 0.0, 1e10, 200_000.0, 10_000.0,
                              help="Uses Polygon snapshot day volume as a simple liquidity filter.")
with c3:
    top_n = st.number_input("Rows per table", 5, 100, 10, 1)
with c4:
    refresh_s = st.number_input("Auto refresh (sec)", 2, 60, 10, 1)

if mode.startswith("Pre-Market"):
    st.caption("Reference = **yesterday’s regular close** (`prevDay.c`). Window: **04:00–09:30 ET**.")
else:
    st.caption("Reference = **today’s regular close** (`day.c`). Window: **≥ 16:00 ET**.")

# ── Time/window helpers ──────────────────────────────────────────────────────
def in_window(ts_ns: int, session_label: str) -> bool:
    ts_sec = ts_ns / 1_000_000_000
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    dt_et  = dt_utc.astimezone(NY)

    if session_label.startswith("Pre-Market"):
        start = datetime.combine(dt_et.date(), dtime(4, 0, 0), tzinfo=NY)
        end   = datetime.combine(dt_et.date(), dtime(9, 30, 0), tzinfo=NY)
        return start <= dt_et < end
    else:
        start = datetime.combine(dt_et.date(), dtime(16, 0, 0), tzinfo=NY)
        return dt_et >= start

def ref_close_from_snapshot(s: Dict[str, Any], session_label: str) -> float | None:
    if session_label.startswith("Pre-Market"):
        prev = s.get("prevDay") or {}
        return prev.get("c")
    else:
        day = s.get("day") or {}
        return day.get("c")

# ── Polygon fetchers ─────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_snapshots(limit_total: int = 6000) -> List[Dict[str, Any]]:
    base = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"
    params = {"limit": 1000, "apiKey": API_KEY}
    out: List[Dict[str, Any]] = []
    url = base
    while url and len(out) < limit_total:
        r = requests.get(url, params=params if url == base else None, timeout=30)
        r.raise_for_status()
        j = r.json()
        out.extend(j.get("tickers", []))
        url = j.get("next_url")
        if url and "apiKey=" not in url:
            url += f"&apiKey={API_KEY}"
    return out[:limit_total]

@st.cache_data(ttl=3600)
def fetch_sector(symbol: str) -> str:
    """
    Lightweight sector/industry lookup via Polygon v3 reference.
    Falls back to SIC description if explicit sector/industry is missing.
    """
    try:
        r = requests.get(
            f"https://api.polygon.io/v3/reference/tickers/{symbol}",
            params={"apiKey": API_KEY},
            timeout=15,
        )
        if r.status_code == 200:
            j = r.json() or {}
            res = j.get("results") or {}
            sector = (res.get("industry")
                      or res.get("sector")
                      or res.get("sic_description")
                      or "")
            return str(sector).strip()
    except Exception:
        pass
    return ""

def add_sector_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    syms = df["SYMBOL"].unique().tolist()
    # Batch fetch with simple loop; cache makes repeated deploys cheap.
    sector_map = {sym: fetch_sector(sym) for sym in syms}
    df["SECTOR"] = df["SYMBOL"].map(sector_map)
    return df

# ── Build movers table from snapshots ────────────────────────────────────────
def build_rows(snaps: List[Dict[str, Any]], session_label: str) -> pd.DataFrame:
    rows = []
    for s in snaps:
        sym  = s.get("ticker")
        day  = s.get("day") or {}
        lt   = s.get("lastTrade") or {}

        price = lt.get("p")
        ts_ns = lt.get("t")
        ref_c = ref_close_from_snapshot(s, session_label)
        vol_day = day.get("v") or 0

        if price is None or ts_ns is None or ref_c is None:
            continue
        if not in_window(ts_ns, session_label):
            continue
        if price < float(min_price) or vol_day < float(min_vol):
            continue

        chg   = float(price) - float(ref_c)
        chg_p = (chg / float(ref_c)) * 100.0 if ref_c else None

        rows.append({
            "SYMBOL": sym,
            "PRICE": float(price),
            "VOLUME": float(vol_day),
            "CHG": float(chg),
            "CHG %": float(chg_p) if chg_p is not None else None,
            "Last Trade (ET)": datetime.fromtimestamp(ts_ns/1_000_000_000, tz=timezone.utc)
                                  .astimezone(NY).strftime("%H:%M:%S"),
            "FINVIZ": f"https://finviz.com/quote.ashx?t={sym}",
        })
    return pd.DataFrame(rows)

def style_table(df: pd.DataFrame, positive: bool):
    if df.empty:
        return df.style
    fmt = {"PRICE": "{:.2f}", "VOLUME": "{:,.0f}", "CHG": "{:.2f}", "CHG %": "{:.2f}"}
    color = "#2e7d32" if positive else "#c62828"
    if positive:
        return df.style.format(fmt).bar(subset=["CHG %"], color=color, vmin=0)
    else:
        return df.style.format(fmt).bar(subset=["CHG %"], color=color, vmax=0)

# ── Fetch / build / render ───────────────────────────────────────────────────
with st.spinner("Fetching live snapshots…"):
    try:
        snaps = fetch_snapshots()
    except requests.HTTPError as e:
        st.error(f"HTTP {e.response.status_code}: {e.response.text[:250]}")
        st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

df_all = build_rows(snaps, mode)

# Split
leaders  = df_all[df_all["CHG %"] > 0].sort_values("CHG %", ascending=False).head(int(top_n))
laggards = df_all[df_all["CHG %"] < 0].sort_values("CHG %", ascending=True).head(int(top_n))

# Add Sector/Industry only for rows you will show (keeps API calls small)
leaders  = add_sector_column(leaders)
laggards = add_sector_column(laggards)

# Optional: reorder columns nicely
def order_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["SYMBOL", "SECTOR", "PRICE", "CHG %", "CHG", "VOLUME", "Last Trade (ET)", "FINVIZ"]
    present = [c for c in cols if c in df.columns]
    return df[present]

leaders  = order_cols(leaders)
laggards = order_cols(laggards)

# Render
left, right = st.columns(2)
with left:
    st.subheader("LEADERS")
    if leaders.empty:
        st.info("No leaders meeting filters.")
    else:
        st.dataframe(
            style_table(leaders, positive=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "FINVIZ": st.column_config.LinkColumn("Finviz", display_text="Open"),
            },
        )
with right:
    st.subheader("LAGGARDS")
    if laggards.empty:
        st.info("No laggards meeting filters.")
    else:
        st.dataframe(
            style_table(laggards, positive=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "FINVIZ": st.column_config.LinkColumn("Finviz", display_text="Open"),
            },
        )

# Footer / auto-refresh
now_et = datetime.now(tz=NY)
if mode.startswith("Pre-Market"):
    st.caption("Window: 04:00–09:30 ET • Updated: " + now_et.strftime("%m/%d/%Y %H:%M:%S ET"))
else:
    st.caption("Window: ≥ 16:00 ET • Updated: " + now_et.strftime("%m/%d/%Y %H:%M:%S ET"))

time.sleep(int(refresh_s))
st.rerun()
