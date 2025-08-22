# afterhours_movers_live.py
# Unified Movers (Polygon) — auto session with robust no-data handling

import os
import math
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import pytz
import streamlit as st

# -------------- Config --------------

API_KEY = os.getenv("POLYGON_API_KEY", "").strip()
assert API_KEY, "POLYGON_API_KEY missing in environment"

ET = pytz.timezone("America/New_York")

MIN_PRICE_DEFAULT = 5.0        # $5+
MIN_WINDOW_VOL_DEFAULT = 2_000_000  # ≥ 2M shares in the window
TOPN_DEFAULT = 60
UNIVERSE_CAP = 1200            # safety cap on symbols to iterate

# -------------- Helpers --------------

def et_now():
    """Return timezone-aware 'now' in US/Eastern."""
    return datetime.now(ET)

def session_window(now_et: datetime):
    """
    Detect session & return (label, win_start_et, win_end_et).
    Pre-Market: 04:00–09:30
    Regular   : 09:30–16:00
    After-Hours: 16:00–20:00
    """
    d = now_et.date()
    dt = lambda h, m=0: ET.localize(datetime(d.year, d.month, d.day, h, m))

    pre_start, pre_end = dt(4, 0), dt(9, 30)
    rth_start, rth_end = dt(9, 30), dt(16, 0)
    ah_start, ah_end = dt(16, 0), dt(20, 0)

    if pre_start <= now_et <= pre_end:
        return "Pre-Market", pre_start, pre_end
    elif rth_start <= now_et <= rth_end:
        return "Regular", rth_start, rth_end
    elif ah_start <= now_et <= ah_end:
        return "After-Hours", ah_start, ah_end
    else:
        # If we're outside any session, default to the nearest completed one (AH)
        # so the app still shows something
        return "After-Hours", ah_start, ah_end

def et_to_unix_ms(dt_et: datetime) -> int:
    """Convert timezone-aware ET datetime to unix ms."""
    return int(dt_et.timestamp() * 1000)

def prev_trading_date():
    """
    Best-effort previous trading date (YYYY-MM-DD).
    We ask Polygon marketstatus; if it fails, fallback to yesterday.
    """
    try:
        ms = requests.get(
            "https://api.polygon.io/v1/marketstatus/now",
            params={"apiKey": API_KEY},
            timeout=8,
        )
        ms.raise_for_status()
        # This endpoint doesn’t directly give prev date; we infer from ET "serverTime".
        # We’ll just use yesterday in ET (good enough for grouped/RTH based universe).
    except Exception:
        pass

    # Yesterday in ET (not UTC)
    y_et = et_now() - timedelta(days=1)
    return y_et.strftime("%Y-%m-%d")

def poly_get(url, **params):
    params = dict(params or {})
    params["apiKey"] = API_KEY
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def fetch_grouped(date_str, market="stocks"):
    """Grouped bars for RTH — used to build a universe of symbols."""
    try:
        data = poly_get(
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/{market}/{date_str}",
            adjusted="true",
        )
        if data.get("results"):
            return data["results"]
    except Exception:
        return []
    return []

def fetch_prev_close(symbol):
    """Yesterday's close (adjusted)."""
    try:
        data = poly_get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev",
            adjusted="true",
        )
        if data.get("results"):
            return data["results"][0].get("c")
    except Exception:
        return None
    return None

def fetch_minute_window(symbol, start_et, end_et):
    """
    Get minute bars within a window and return (last_price, sum_volume).
    If no bars, returns (None, 0).
    """
    try:
        data = poly_get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/"
            f"{start_et.strftime('%Y-%m-%d')}/{end_et.strftime('%Y-%m-%d')}",
            adjusted="true",
            sort="asc",
            limit=50000,
        )
        results = data.get("results", [])
        if not results:
            return None, 0

        start_ms = et_to_unix_ms(start_et)
        end_ms = et_to_unix_ms(end_et)

        # keep bars that fall within [start_ms, end_ms)
        bars = [b for b in results if start_ms <= b.get("t", 0) < end_ms]
        if not bars:
            return None, 0

        last_price = bars[-1].get("c")
        tot_vol = sum(b.get("v", 0) for b in bars)
        return last_price, tot_vol
    except Exception:
        return None, 0

def finviz_link(symbol: str) -> str:
    return f"https://finviz.com/quote.ashx?t={symbol}"

# Expected dataframe columns — used to guarantee schema even when empty
EXPECTED_COLS = [
    "Symbol",
    "Price",
    "Prev Close",
    "CHG",
    "CHG %",
    "Volume",
    "Last (ET)",
    "Src",
    "Finviz",
]

def empty_df():
    """Return an empty dataframe with expected schema."""
    df = pd.DataFrame({c: [] for c in EXPECTED_COLS})
    return df[EXPECTED_COLS]

# -------------- Builder --------------

def build_movers(min_price=MIN_PRICE_DEFAULT,
                 min_window_vol=MIN_WINDOW_VOL_DEFAULT,
                 include_otc=False):
    """
    Build full df, and leaders/laggards tables. Always returns dataframes
    with the EXPECTED_COLS schema (may be empty).
    """
    now = et_now()
    sess_label, win_start, win_end = session_window(now)

    # Universe from yesterday RTH grouped bars
    y = prev_trading_date()
    grouped = fetch_grouped(y, market="stocks")
    if include_otc:
        grouped += fetch_grouped(y, market="otc")

    # Light filter/trim universe
    # (use grouped close as a proxy to filter out very low price names quickly)
    universe = []
    for g in grouped:
        sym = g.get("T")
        close = g.get("c")
        if not sym or close is None:
            continue
        if close >= min_price:
            universe.append(sym)
        if len(universe) >= UNIVERSE_CAP:
            break

    rows = []
    for i, sym in enumerate(universe):
        # polite pacing to avoid spiky rate usage (and to be cloud-friendly)
        if i and i % 200 == 0:
            time.sleep(0.6)

        pclose = fetch_prev_close(sym)
        if not pclose or pclose <= 0:
            continue

        last, wvol = fetch_minute_window(sym, win_start, win_end)
        if last is None:
            continue
        if wvol < min_window_vol:
            continue

        chg = last - pclose
        chg_pct = (chg / pclose) * 100.0

        rows.append({
            "Symbol": sym,
            "Price": round(last, 2) if isinstance(last, (float, int)) else last,
            "Prev Close": round(pclose, 2),
            "CHG": round(chg, 2),
            "CHG %": round(chg_pct, 2),
            "Volume": int(wvol),
            "Last (ET)": win_end.strftime("%H:%M"),
            "Src": sess_label,
            "Finviz": f"[Open]({finviz_link(sym)})",
        })

    if not rows:
        return empty_df(), empty_df(), empty_df(), (sess_label, win_start, win_end)

    df = pd.DataFrame(rows)
    # Guarantee the schema even if something got dropped
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = []  # add empty

    df = df[EXPECTED_COLS]

    # Sort / slice safely
    if df.empty:
        return df, empty_df(), empty_df(), (sess_label, win_start, win_end)

    df = df.sort_values("CHG %", ascending=False, na_position="last")

    leaders = df[df["CHG %"] > 0].copy()
    laggards = df[df["CHG %"] < 0].copy().sort_values("CHG %", ascending=True)

    return df, leaders, laggards, (sess_label, win_start, win_end)

# -------------- UI --------------

st.set_page_config(page_title="Unified Movers — Auto Session (Polygon)", layout="wide")

st.sidebar.header("Options")
include_otc = st.sidebar.checkbox("Include OTC", value=False)
topn = st.sidebar.slider("Show Top N by |% Change|", min_value=20, max_value=200, value=TOPN_DEFAULT, step=5)
refresh = st.sidebar.button("Refresh")

now_et = et_now()
sess_lbl, ws, we = session_window(now_et)

st.title("⚡ Unified Movers — Auto Session (Polygon)")
st.write(
    f"**Now (ET)**: {now_et.strftime('%Y-%m-%d %H:%M:%S')} • "
    f"**Session**: {sess_lbl} • "
    f"**Window**: {ws.strftime('%H:%M')}–{we.strftime('%H:%M')} ET"
)

with st.spinner("Building movers…"):
    df_all, leaders, laggards, (sess_lbl, ws, we) = build_movers(
        MIN_PRICE_DEFAULT,
        MIN_WINDOW_VOL_DEFAULT,
        include_otc=include_otc,
    )

# Handle no data gracefully
if df_all.empty:
    st.warning(
        "No rows matched your filters/time window. "
        "This can happen especially early in Pre-Market or late in After-Hours.\n\n"
        "Tips:\n"
        "- Try enabling **Include OTC**.\n"
        "- Try increasing the time window (wait a bit) or lowering the volume threshold inside the code "
        "(defaults are $5+ and ≥ 2M shares within the window).\n"
        "- Market holidays/half-days also impact availability."
    )
else:
    # Top N by absolute % change for the headline lists
    leaders_n = leaders.head(topn).copy()
    laggards_n = laggards.head(topn).copy()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Leaders")
        st.dataframe(
            leaders_n.style.format({
                "Price": "{:.2f}",
                "Prev Close": "{:.2f}",
                "CHG": "{:.2f}",
                "CHG %": "{:.2f}",
                "Volume": "{:,}",
            }),
            use_container_width=True,
            height=480,
        )
    with col2:
        st.subheader("Laggards")
        st.dataframe(
            laggards_n.style.format({
                "Price": "{:.2f}",
                "Prev Close": "{:.2f}",
                "CHG": "{:.2f}",
                "CHG %": "{:.2f}",
                "Volume": "{:,}",
            }),
            use_container_width=True,
            height=480,
        )

    st.markdown("---")
    with st.expander("Full universe (filtered)", expanded=False):
        st.dataframe(
            df_all.style.format({
                "Price": "{:.2f}",
                "Prev Close": "{:.2f}",
                "CHG": "{:.2f}",
                "CHG %": "{:.2f}",
                "Volume": "{:,}",
            }),
            use_container_width=True,
            height=520,
        )

st.caption(
    "Price = last price inside the detected session window versus **yesterday's adjusted close**. "
    "Volume is the **sum of minute bars** inside the session window. "
    "Links open **Finviz** for deeper context. "
    "Defaults: $5+ & ≥2M shares in window; adjust in code if needed."
)
