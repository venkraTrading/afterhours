# streamlit run multi_session_gap_rsi.py
import os
import time
from datetime import datetime, date, timedelta
import pytz
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -------------------------
# Config & helpers
# -------------------------
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
if not POLYGON_API_KEY:
    st.stop()

SESSION_TZ = pytz.timezone("America/New_York")

def now_et():
    return datetime.now(SESSION_TZ)

def today_et_datestr():
    return now_et().strftime("%Y-%m-%d")

def et_iso(ts: datetime) -> str:
    """Return ISO timestamp (without timezone) for Polygon aggs query."""
    return ts.strftime("%Y-%m-%d")

def session_label(ts: datetime) -> str:
    t = ts.time()
    pre_start  = datetime.combine(ts.date(), datetime.strptime("04:00","%H:%M").time()).time()
    rth_start  = datetime.combine(ts.date(), datetime.strptime("09:30","%H:%M").time()).time()
    rth_end    = datetime.combine(ts.date(), datetime.strptime("16:00","%H:%M").time()).time()
    ah_end     = datetime.combine(ts.date(), datetime.strptime("20:00","%H:%M").time()).time()

    if pre_start <= t < rth_start: return "Pre-Market"
    if rth_start <= t < rth_end:   return "Regular"
    if rth_end   <= t < ah_end:    return "After-Hours"
    return "Closed"

def finviz_link(symbol: str) -> str:
    return f"https://finviz.com/quote.ashx?t={symbol.upper()}"

def human_int(x: float) -> str:
    if pd.isna(x): return "-"
    if x >= 1e9: return f"{x/1e9:.2f}B"
    if x >= 1e6: return f"{x/1e6:.2f}M"
    if x >= 1e3: return f"{x/1e3:.2f}K"
    return f"{x:.0f}"

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0.0)
    loss  = -delta.clip(upper=0.0)

    # Wilder’s smoothing
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

# -------------------------
# Polygon calls
# -------------------------
def get_1m_bars_today(symbol: str) -> pd.DataFrame:
    """
    Get today's 1-min bars from 04:00 ET to now.
    """
    today = today_et_datestr()
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/minute/{today}/{today}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": POLYGON_API_KEY,
    }
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"[{symbol}] {r.status_code} {r.text}")

    js = r.json()
    if js.get("status") != "OK" or "results" not in js:
        return pd.DataFrame(columns=["t","o","h","l","c","v"])

    df = pd.DataFrame(js["results"])
    # Convert Unix ms to timezone-aware ET datetimes
    df["dt"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(SESSION_TZ)
    # Keep from 04:00 ET to now
    start = df["dt"].dt.normalize().iloc[-1] + pd.Timedelta(hours=4)
    end   = now_et()
    df = df[(df["dt"] >= start) & (df["dt"] <= end)]
    # Rename to standard fields
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    return df[["dt","open","high","low","close","volume"]].reset_index(drop=True)

# -------------------------
# Core compute per symbol
# -------------------------
def analyze_symbol(symbol: str,
                   min_price: float,
                   min_rvol: float,
                   up_threshold: float,
                   rsi_low: float,
                   rsi_high: float) -> dict | None:
    """
    Returns a dict of computed metrics if it passes filters, else None.
    """
    try:
        df = get_1m_bars_today(symbol)
        if df.empty:
            return None
    except Exception as e:
        return None

    # last row is "now"
    last = df.iloc[-1]
    last_idx = len(df) - 1

    # Basic price filter
    last_price = float(last["close"])
    if last_price < min_price:
        return None

    # Compute RSI(14) on 1m closes
    df["rsi"] = rsi_wilder(df["close"], 14)
    last_rsi = float(df["rsi"].iloc[-1])
    if not (rsi_low <= last_rsi <= rsi_high):
        return None

    # Compute 1m RVOL: last volume ÷ SMA(20) 1m volume
    if len(df) < 21:
        return None
    sma20_vol = df["volume"].tail(21)[:-1].mean()  # prior 20 bars
    rvol1m = float(last["volume"]) / (sma20_vol if sma20_vol > 0 else np.nan)
    if not np.isfinite(rvol1m) or rvol1m < min_rvol:
        return None

    # Prior 1/5/15-minute closes (if available)
    def pct_vs_ago(minutes: int) -> float | None:
        rows_back = minutes  # with 1m bars, N minutes = N rows
        idx = last_idx - rows_back
        if idx >= 0:
            ref_close = float(df["close"].iloc[idx])
            if ref_close > 0:
                return (last_price / ref_close - 1) * 100.0
        return None

    chg_1m  = pct_vs_ago(1)
    chg_5m  = pct_vs_ago(5)
    chg_15m = pct_vs_ago(15)

    # If none of them is positive vs threshold -> discard
    ups = [x for x in [chg_1m, chg_5m, chg_15m] if x is not None]
    if not ups or max(ups) < up_threshold:
        return None

    # Build result row
    row = {
        "Symbol": symbol.upper(),
        "Price": round(last_price, 4),
        "% vs 1m": None if chg_1m is None else round(chg_1m, 2),
        "% vs 5m": None if chg_5m is None else round(chg_5m, 2),
        "% vs 15m": None if chg_15m is None else round(chg_15m, 2),
        "RSI(14)": round(last_rsi, 1),
        "RVOL(1m)": round(rvol1m, 2),
        "Time": last["dt"].strftime("%H:%M:%S"),
        "Session": session_label(last["dt"]),
        "Finviz": finviz_link(symbol),
        "1m Vol": int(last["volume"]),
    }
    return row

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Multi-Session Gap + RSI Screener", page_icon="⚡", layout="wide")
st.title("⚡ Multi-Session Gap Screener (Polygon)")

colA, colB, colC, colD = st.columns([1,1,1,1])

with colA:
    st.markdown("**Universe (comma-separated)**")
    symbols_in = st.text_area(
        "e.g., AAPL, MSFT, NVDA, TSLA …",
        value="AAPL, NVDA, TSLA, AMD, MSFT, META, AMZN, SMCI, AVGO, GOOGL",
        height=80,
        label_visibility="collapsed"
    )

with colB:
    min_price = st.number_input("Min last price ($)", min_value=0.0, value=5.0, step=0.5)
    min_rvol  = st.number_input("Min RVOL(1m)", min_value=0.5, value=1.50, step=0.1)

with colC:
    up_threshold = st.number_input("Min % up vs any of 1/5/15m", min_value=0.0, value=0.0, step=0.25)
    rsi_low  = st.number_input("RSI lower bound", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
    rsi_high = st.number_input("RSI upper bound", min_value=0.0, max_value=100.0, value=80.0, step=1.0)

with colD:
    top_n = st.slider("Show Top N (by max % among 1/5/15m)", min_value=10, max_value=200, value=60, step=10)
    poll = st.checkbox("Auto-refresh (15s)", value=False)

session_now = session_label(now_et())
st.caption(f"**Now (ET):** {now_et().strftime('%Y-%m-%d %H:%M:%S')}  •  **Session:** {session_now}")

# Clean symbol list
symbols = [s.strip().upper() for s in symbols_in.split(",") if s.strip()]
symbols = list(dict.fromkeys(symbols))  # unique, keep order

if not symbols:
    st.info("Enter at least one symbol.")
    st.stop()

btn = st.button("Run Screener")

if poll and not btn:
    # Auto-refresh every 15 seconds
    st.experimental_rerun()

if btn or poll:
    rows = []
    prog = st.progress(0.0, text="Fetching & computing …")
    for i, sym in enumerate(symbols, start=1):
        row = analyze_symbol(
            sym,
            min_price=min_price,
            min_rvol=min_rvol,
            up_threshold=up_threshold,
            rsi_low=rsi_low,
            rsi_high=rsi_high
        )
        if row:
            rows.append(row)
        # gentle pacing to avoid rate limits
        time.sleep(0.10)
        prog.progress(i/len(symbols), text=f"[{i}/{len(symbols)}] {sym}")

    prog.empty()

    if not rows:
        st.warning("No matches based on your filters right now. Try relaxing RVOL/RSI/Up thresholds or adjust universe.")
        st.stop()

    df = pd.DataFrame(rows)

    # Sort by max of the three % changes
    df["Max %"] = df[["% vs 1m","% vs 5m","% vs 15m"]].max(axis=1, skipna=True)
    df = df.sort_values(["Max %","RVOL(1m)"], ascending=[False, False]).head(top_n)

    # Pretty columns
    show = df[[
        "Symbol","Session","Time","Price",
        "% vs 1m","% vs 5m","% vs 15m",
        "RSI(14)","RVOL(1m)","1m Vol","Finviz"
    ]].copy()

    # Add clickable Finviz link
    show["Finviz"] = show["Finviz"].apply(lambda url: f"[Open]({url})")

    # Format
    st.subheader("Results")
    st.dataframe(
        show,
        hide_index=True,
        use_container_width=True
    )
