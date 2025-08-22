import os
import time
from datetime import datetime, date, time as dtime, timezone
from typing import List, Dict, Any

import pandas as pd
import requests
import streamlit as st
from pytz import timezone as pytz_tz

# Read the key from env OR Streamlit Secrets (works on Streamlit Cloud)
API_KEY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY")

st.set_page_config(page_title="After-Hours Movers — Live", page_icon="⚡", layout="wide")

if not API_KEY:
    st.error(
        "POLYGON_API_KEY not found.\n\n"
        "On Streamlit Cloud: Manage app → Settings → Secrets → add:\n"
        "POLYGON_API_KEY = \"YOUR_KEY_HERE\""
    )
    st.stop()

NY = pytz_tz("America/New_York")
st.title("⚡ After-Hours Movers (Live)")

# ── Controls ─────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    min_price = st.number_input("Min last price ($)", 0.0, 1e5, 2.0, 0.5)
with c2:
    min_vol = st.number_input("Min day volume (shares)", 0.0, 1e10, 200_000.0, 10_000.0)
with c3:
    top_n = st.number_input("Rows per table", 5, 100, 10, 1)
with c4:
    refresh_s = st.number_input("Auto refresh (sec)", 2, 60, 10, 1)

st.caption("Reference price = today’s 16:00 ET regular-session **close**. "
           "Only trades timestamped **after 16:00:00 ET** are included.")

# ── Helpers ──────────────────────────────────────────
def is_afterhours(ts_ns: int) -> bool:
    ts_sec = ts_ns / 1_000_000_000
    dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    dt_et  = dt_utc.astimezone(NY)
    et_close = datetime.combine(dt_et.date(), dtime(16,0,0), tzinfo=NY)
    return dt_et >= et_close

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

def build_table(snaps: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in snaps:
        sym  = s.get("ticker")
        day  = (s.get("day") or {})
        lt   = (s.get("lastTrade") or {})

        price = lt.get("p")
        ts_ns = lt.get("t")
        close = day.get("c")
        vol_day = day.get("v") or 0

        if price is None or ts_ns is None or close is None:
            continue
        if not is_afterhours(ts_ns):
            continue
        if price < float(min_price) or vol_day < float(min_vol):
            continue

        chg   = price - close
        chg_p = (chg / close) * 100.0 if close else None
        rows.append({
            "SYMBOL": sym,
            "PRICE": float(price),
            "VOLUME": float(vol_day),
            "CHG": float(chg),
            "CHG %": float(chg_p) if chg_p is not None else None,
            "Last Trade (ET)": datetime.fromtimestamp(ts_ns/1_000_000_000, tz=timezone.utc)
                                  .astimezone(NY).strftime("%H:%M:%S"),
        })
    return pd.DataFrame(rows)

def pretty_table(df: pd.DataFrame, positive: bool = True) -> "Styler":
    if df.empty:
        return df.style
    fmt = {
        "PRICE": "{:.2f}",
        "VOLUME": "{:,.0f}",
        "CHG": "{:.2f}",
        "CHG %": "{:.2f}",
    }
    color = "#2e7d32" if positive else "#c62828"
    sty = (df.style
             .format(fmt)
             .bar(subset=["CHG %"], color=color, vmin=0 if positive else None, vmax=None))
    return sty

# ── Run ──────────────────────────────────────────────
with st.spinner("Fetching live snapshots…"):
    snaps = fetch_snapshots()

df_all = build_table(snaps)

leaders  = df_all[df_all["CHG %"] > 0].sort_values("CHG %", ascending=False).head(int(top_n))
laggards = df_all[df_all["CHG %"] < 0].sort_values("CHG %", ascending=True).head(int(top_n))

left, right = st.columns(2)
with left:
    st.subheader("LEADERS")
    if leaders.empty:
        st.info("No leaders meeting filters.")
    else:
        st.dataframe(pretty_table(leaders, positive=True), use_container_width=True)
with right:
    st.subheader("LAGGARDS")
    if laggards.empty:
        st.info("No laggards meeting filters.")
    else:
        st.dataframe(pretty_table(laggards, positive=False), use_container_width=True)

st.caption("Table Updated: " + datetime.now(tz=NY).strftime("%m/%d/%Y %H:%M:%S ET"))

# Auto-refresh
time.sleep(int(refresh_s))
st.rerun()
