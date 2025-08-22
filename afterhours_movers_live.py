# unified_movers_app.py
# Streamlit app: Unified Movers — Auto Session (Polygon)
#
# Features:
# - Auto-detect session (Pre-Market / Regular / After-Hours), allow override
# - Pulls Polygon snapshot gainers/losers (fast) then enriches top N with minute-window stats
# - Computes session % change and volume using only the active window
# - Min price $5+ and Min window volume 2,000,000 by default
# - Finviz links, OTC toggle, strong caching to respect rate limits

import os
import time
import math
import zoneinfo
from datetime import datetime, date, timedelta

import requests
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Config & helpers
# -----------------------------
ET = zoneinfo.ZoneInfo("America/New_York")

def get_api_key() -> str:
    # Order: st.secrets then env
    key = st.secrets.get("POLYGON_API_KEY", None) if hasattr(st, "secrets") else None
    return key or os.getenv("POLYGON_API_KEY", "")

API_KEY = get_api_key()

def poly_get(url: str, params: dict | None = None):
    """GET helper with basic error handling."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        if r.status_code == 429:
            # Rate limited: back off briefly
            time.sleep(1.2)
            r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Polygon error on {url.split('?')[0]}: {e}")
        return None

def now_et():
    return datetime.now(tz=ET)

def infer_session(ts: datetime):
    # Return ("pre", "regular", or "after"), and window times in ET.
    # Pre-market: 04:00–09:30, Regular: 09:30–16:00, After-hours: 16:00–20:00
    d = ts.date()
    pre_start  = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET)
    rth_start  = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
    rth_end    = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)
    aft_end    = datetime(d.year, d.month, d.day, 20, 0, tzinfo=ET)

    if ts < pre_start:
        # before 04:00 — use last after-hours session of prior day
        y = d - timedelta(days=1)
        return ("after",
                datetime(y.year, y.month, y.day, 16, 0, tzinfo=ET),
                datetime(y.year, y.month, y.day, 20, 0, tzinfo=ET))

    if pre_start <= ts < rth_start:
        return ("pre", pre_start, rth_start)
    if rth_start <= ts < rth_end:
        return ("regular", rth_start, rth_end)
    if rth_end <= ts < aft_end:
        return ("after", rth_end, aft_end)
    # after 20:00 — keep after-hours window
    return ("after", rth_end, aft_end)

def dt_to_datestr(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")

def et_to_unix_ms(d: datetime) -> int:
    return int(d.timestamp() * 1000)

# -----------------------------
# Caching: snapshots & minute bars
# -----------------------------
@st.cache_data(ttl=30, show_spinner=False)
def get_snapshot_movers(direction: str = "gainers", include_otc: bool = False):
    """
    Polygon snapshot top gainers/losers.
    Docs: /v2/snapshot/locale/us/markets/stocks/gainers  (or losers)
    """
    base = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks"
    url = f"{base}/{direction}"
    params = {}
    if include_otc:
        params["include_otc"] = "true"

    data = poly_get(url, params)
    if not data or "tickers" not in data:
        return pd.DataFrame(columns=[
            "symbol","last","day_open","day_close","day_change","day_change_perc","prev_close",
            "todays_change","todays_change_perc","updated_at"
        ])

    rows = []
    for t in data.get("tickers", []):
        # Defensive parsing
        symbol = t.get("ticker")
        last = (t.get("lastTrade") or {}).get("p")
        updated = (t.get("lastTrade") or {}).get("t")
        day = t.get("day", {}) or {}
        prev = t.get("prevDay", {}) or {}
        rows.append({
            "symbol": symbol,
            "last": last,
            "day_open": day.get("o"),
            "day_close": day.get("c"),
            "day_change": day.get("c", np.nan) - day.get("o", np.nan) if all(x in day for x in ["o","c"]) else np.nan,
            "day_change_perc": ( (day.get("c", np.nan)-day.get("o", np.nan)) / day.get("o", np.nan) * 100.0 )
                if all(k in day for k in ["o","c"]) and day.get("o") else np.nan,
            "prev_close": prev.get("c"),
            "todays_change": t.get("todaysChange"),
            "todays_change_perc": t.get("todaysChangePerc"),
            "updated_at": updated
        })
    df = pd.DataFrame(rows)
    return df

@st.cache_data(ttl=60, show_spinner=False)
def get_minute_aggs(symbol: str, start_et: datetime, end_et: datetime, adjusted: bool = True):
    """
    Fetch 1-minute aggs for the date of start_et, filter to [start_et, end_et].
    Using /v2/aggs/ticker/{}/range/1/minute/{date}/{date}
    """
    d = start_et.date()
    date_str = d.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{date_str}/{date_str}"
    params = {
        "adjusted": "true" if adjusted else "false",
        "sort": "asc",
        "limit": 50000,
    }
    data = poly_get(url, params=params)
    if not data or data.get("resultsCount", 0) == 0:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])

    df = pd.DataFrame(data["results"])
    # Polygon uses Unix ms in UTC; convert to ET
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
    mask = (df["t"] >= start_et) & (df["t"] <= end_et)
    df = df.loc[mask, ["t","o","h","l","c","v"]].reset_index(drop=True)
    return df

# -----------------------------
# Build session metrics (enrichment)
# -----------------------------
def enrich_with_session(df_symbols: pd.DataFrame,
                        session_label: str,
                        start_et: datetime,
                        end_et: datetime,
                        min_price: float,
                        min_window_vol: float,
                        top_n: int):
    """
    For a (small) set of symbols from snapshots, fetch minute bars in the session window,
    compute session metrics, filter, and rank by % change.
    """
    rows = []
    # Keep count of API calls low – only enrich up to top_n * 2 from snapshots first
    symbols = df_symbols["symbol"].dropna().astype(str).tolist()
    symbols = symbols[: max(top_n * 2, 20)]

    for sym in symbols:
        m = get_minute_aggs(sym, start_et, end_et)
        if m.empty:
            continue

        open_px  = m.iloc[0]["o"]
        close_px = m.iloc[-1]["c"]
        high_px  = m["h"].max()
        low_px   = m["l"].min()
        wvol     = float(m["v"].sum())

        if not (isinstance(open_px, (int,float)) and isinstance(close_px, (int,float))):
            continue
        if open_px is None or open_px == 0 or np.isnan(open_px):
            continue

        change = close_px - open_px
        pct    = (change / open_px) * 100.0

        rows.append({
            "Symbol": sym,
            "Open": open_px,
            "High": high_px,
            "Low": low_px,
            "Close": close_px,
            "CHG %": pct,
            "Window Vol": wvol,
            "Finviz": f"https://finviz.com/quote.ashx?t={sym}",
        })

    if not rows:
        return pd.DataFrame(columns=["Symbol","Open","High","Low","Close","CHG %","Window Vol","Finviz"])

    out = pd.DataFrame(rows)

    # Basic filters
    out = out[(out["Close"] >= float(min_price)) & (out["Window Vol"] >= float(min_window_vol))]

    # Rank by % change (abs if you want biggest movers regardless of sign)
    out = out.sort_values("CHG %", ascending=False, na_position="last").head(int(top_n)).reset_index(drop=True)

    # Human-friendly formatting (keep numeric dtypes for sorting, show nice in UI)
    return out

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Unified Movers — Auto Session (Polygon)", layout="wide", page_icon="⚡")
st.title("⚡ Unified Movers — Auto Session (Polygon)")

if not API_KEY:
    st.error("No Polygon API key found. Set `POLYGON_API_KEY` in Secrets or environment.")
    st.stop()

now = now_et()
auto_session, auto_start, auto_end = infer_session(now)

# Sidebar
st.sidebar.header("Options")

include_otc = st.sidebar.checkbox("Include OTC", value=False)
direction = st.sidebar.selectbox("List", ["Gainers", "Losers"], index=0)
session_pick = st.sidebar.selectbox("Session", ["Auto (current)", "Pre-Market", "Regular", "After-Hours"], index=0)

# Defaults requested
min_price_default = 5.00
min_vol_default = 2_000_000

top_n = st.sidebar.slider("Show Top N by |% Change|", min_value=10, max_value=200, value=60, step=5)

# Resolve session
session_label = auto_session
start_et = auto_start
end_et = auto_end
if session_pick != "Auto (current)":
    d = now.date()
    if session_pick == "Pre-Market":
        start_et = datetime(d.year, d.month, d.day, 4, 0, tzinfo=ET)
        end_et   = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
        session_label = "pre"
    elif session_pick == "Regular":
        start_et = datetime(d.year, d.month, d.day, 9, 30, tzinfo=ET)
        end_et   = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)
        session_label = "regular"
    elif session_pick == "After-Hours":
        start_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)
        end_et   = datetime(d.year, d.month, d.day, 20, 0, tzinfo=ET)
        session_label = "after"

colA, colB, colC = st.columns([1.2, 1.2, 2.4])
with colA:
    st.metric("Now (ET)", now.strftime("%Y-%m-%d %H:%M:%S"))
with colB:
    pretty = {"pre": "Pre-Market", "regular": "Regular", "after": "After-Hours"}.get(session_label, session_label)
    st.metric("Session", pretty)
with colC:
    st.metric("Window", f"{start_et.strftime('%H:%M')}–{end_et.strftime('%H:%M')} ET")

# -----------------------------
# Fetch snapshots
# -----------------------------
with st.spinner(f"Fetching Polygon snapshot {direction.lower()}…"):
    snap = get_snapshot_movers(
        direction="gainers" if direction.lower() == "gainers" else "losers",
        include_otc=include_otc
    )

if snap.empty:
    st.info("No snapshot tickers returned. Market may be closed or you are being rate limited. Try again shortly.")
    st.stop()

# Show snapshot summary (lightweight)
with st.expander("Snapshot (raw from Polygon) — quick view", expanded=False):
    slim = snap.loc[:, ["symbol","last","todays_change_perc","day_open","day_close","prev_close"]].copy()
    slim.rename(columns={
        "symbol":"Symbol","last":"Last","todays_change_perc":"Today %","day_open":"Day Open","day_close":"Day Close","prev_close":"Prev Close"
    }, inplace=True)
    st.dataframe(slim, use_container_width=True)

# -----------------------------
# Enrich the top snapshot set with minute window stats
# -----------------------------
min_price = min_price_default
min_window_vol = min_vol_default

with st.spinner("Building movers (minute window)…"):
    df_enriched = enrich_with_session(
        df_symbols=snap[["symbol"]].dropna(),
        session_label=session_label,
        start_et=start_et,
        end_et=end_et,
        min_price=min_price,
        min_window_vol=min_window_vol,
        top_n=top_n
    )

if df_enriched.empty:
    st.warning("No rows matched filters **(min price $5+, session volume ≥ 2,000,000)** within this time window. "
               "Try enabling OTC, increasing Top N, or switching session.")
    st.stop()

# Display
def fmt_int(x):
    try:
        return f"{int(x):,}"
    except:
        try:
            return f"{float(x):,.0f}"
        except:
            return x

view = df_enriched.copy()
view["Window Vol"] = view["Window Vol"].apply(fmt_int)
view["Open"]  = view["Open"].map(lambda v: f"{v:,.2f}")
view["High"]  = view["High"].map(lambda v: f"{v:,.2f}")
view["Low"]   = view["Low"].map(lambda v: f"{v:,.2f}")
view["Close"] = view["Close"].map(lambda v: f"{v:,.2f}")
view["CHG %"] = view["CHG %"].map(lambda v: f"{v:,.2f}")

st.dataframe(
    view,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Symbol": st.column_config.TextColumn("Symbol"),
        "Open": st.column_config.TextColumn("Open"),
        "High": st.column_config.TextColumn("High"),
        "Low": st.column_config.TextColumn("Low"),
        "Close": st.column_config.TextColumn("Close"),
        "CHG %": st.column_config.TextColumn("CHG %"),
        "Window Vol": st.column_config.TextColumn("Window Vol"),
        "Finviz": st.column_config.LinkColumn("Finviz", display_text="Open"),
    },
)

st.caption("Tip: Results are cached for 30–60s to protect your Polygon limits. Multiple refreshes within that window reuse data.")
