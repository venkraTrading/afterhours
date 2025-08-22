# afterhours_movers_live.py
# Enhanced Unified Movers (Polygon) — auto session with robust deployment support

import os
import math
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import pytz
import streamlit as st
from typing import Tuple, Optional, List, Dict, Any
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json

# -------------- Config --------------

def get_api_key() -> str:
    """Get API key with fallback options for deployment"""
    # Try environment variable first
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    
    # Try Streamlit secrets
    if not api_key:
        try:
            api_key = st.secrets.get("POLYGON_API_KEY", "").strip()
        except:
            pass
    
    # If still no key, use session state or sidebar input
    if not api_key:
        if 'polygon_api_key' not in st.session_state:
            st.session_state.polygon_api_key = ""
        
        if not st.session_state.polygon_api_key:
            with st.sidebar:
                st.error("⚠️ Polygon.io API Key Required")
                api_key_input = st.text_input(
                    "Enter your Polygon.io API Key:",
                    type="password",
                    help="Get your free API key from polygon.io",
                    key="api_key_input"
                )
                if api_key_input:
                    st.session_state.polygon_api_key = api_key_input
                    st.rerun()
                else:
                    st.info("You can get a free API key from [polygon.io](https://polygon.io)")
                    st.stop()
        
        api_key = st.session_state.polygon_api_key
    
    return api_key

ET = pytz.timezone("America/New_York")

MIN_PRICE_DEFAULT = 5.0        # $5+
MIN_WINDOW_VOL_DEFAULT = 1_000_000  # Reduced to 1M for better results
TOPN_DEFAULT = 60
UNIVERSE_CAP = 800            # Reduced for better performance
MAX_CONCURRENT_REQUESTS = 10   # Rate limiting

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

    if pre_start <= now_et < pre_end:
        return "Pre-Market", pre_start, pre_end
    elif rth_start <= now_et < rth_end:
        return "Regular", rth_start, rth_end
    elif ah_start <= now_et < ah_end:
        return "After-Hours", ah_start, ah_end
    else:
        # Outside trading hours - determine which session to show
        if now_et.time() < pre_start.time():
            # Before pre-market, show previous day's after hours
            prev_day = (now_et - timedelta(days=1)).date()
            prev_ah_start = ET.localize(datetime(prev_day.year, prev_day.month, prev_day.day, 16, 0))
            prev_ah_end = ET.localize(datetime(prev_day.year, prev_day.month, prev_day.day, 20, 0))
            return "After-Hours (Prev)", prev_ah_start, prev_ah_end
        else:
            # After hours ended, show today's after hours
            return "After-Hours (Ended)", ah_start, ah_end

def et_to_unix_ms(dt_et: datetime) -> int:
    """Convert timezone-aware ET datetime to unix ms."""
    return int(dt_et.timestamp() * 1000)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def prev_trading_date():
    """Get previous trading date with caching"""
    try:
        api_key = get_api_key()
        ms = requests.get(
            "https://api.polygon.io/v1/marketstatus/now",
            params={"apiKey": api_key},
            timeout=10,
        )
        if ms.status_code == 200:
            data = ms.json()
            # Use the date info from market status if available
            pass
    except Exception as e:
        st.warning(f"Could not fetch market status: {e}")

    # Fallback: yesterday in ET
    y_et = et_now() - timedelta(days=1)
    return y_et.strftime("%Y-%m-%d")

def poly_get(url, **params):
    """Make API call with enhanced error handling"""
    params = dict(params or {})
    api_key = get_api_key()
    params["apiKey"] = api_key
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            
            if r.status_code == 429:  # Rate limited
                wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                st.warning(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            r.raise_for_status()
            return r.json()
            
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                st.error(f"Timeout after {max_retries} attempts")
                raise
            time.sleep(1)
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                st.error(f"API request failed: {e}")
                raise
            time.sleep(1)
    
    return {}

@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_grouped(date_str, market="stocks"):
    """Grouped bars for RTH — used to build a universe of symbols."""
    try:
        data = poly_get(
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/{market}/{date_str}",
            adjusted="true",
        )
        if data.get("results"):
            return data["results"]
    except Exception as e:
        st.error(f"Error fetching grouped data: {e}")
        return []
    return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
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
    Enhanced with better error handling and performance.
    """
    try:
        start_date = start_et.strftime('%Y-%m-%d')
        end_date = end_et.strftime('%Y-%m-%d')
        
        data = poly_get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}",
            adjusted="true",
            sort="asc",
            limit=50000,
        )
        
        results = data.get("results", [])
        if not results:
            return None, 0

        start_ms = et_to_unix_ms(start_et)
        end_ms = et_to_unix_ms(end_et)

        # Filter bars within the time window
        bars = [b for b in results if start_ms <= b.get("t", 0) < end_ms]
        if not bars:
            return None, 0

        last_price = bars[-1].get("c")
        tot_vol = sum(b.get("v", 0) for b in bars)
        return last_price, tot_vol
        
    except Exception as e:
        # Don't spam errors for individual symbols
        return None, 0

def finviz_link(symbol: str) -> str:
    return f"https://finviz.com/quote.ashx?t={symbol}"

def tradingview_link(symbol: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol={symbol}"

# Expected dataframe columns
EXPECTED_COLS = [
    "Symbol",
    "Price",
    "Prev Close",
    "CHG",
    "CHG %",
    "Volume",
    "Last (ET)",
    "Src",
    "Charts",
]

def empty_df():
    """Return an empty dataframe with expected schema."""
    df = pd.DataFrame({c: [] for c in EXPECTED_COLS})
    return df[EXPECTED_COLS]

# -------------- Enhanced Builder with Progress --------------

def build_movers_with_progress(min_price=MIN_PRICE_DEFAULT,
                              min_window_vol=MIN_WINDOW_VOL_DEFAULT,
                              include_otc=False):
    """
    Build movers with progress bar and better error handling
    """
    now = et_now()
    sess_label, win_start, win_end = session_window(now)

    # Show current session info
    st.info(f"🕐 **Current Session**: {sess_label} | **Window**: {win_start.strftime('%H:%M')}–{win_end.strftime('%H:%M')} ET")

    # Get universe
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    progress_text.text("📊 Fetching universe of stocks...")
    y = prev_trading_date()
    grouped = fetch_grouped(y, market="stocks")
    
    if include_otc:
        progress_text.text("📊 Adding OTC stocks...")
        grouped += fetch_grouped(y, market="otc")

    if not grouped:
        st.error("❌ Could not fetch stock universe. Please check your API key and try again.")
        return empty_df(), empty_df(), empty_df(), (sess_label, win_start, win_end)

    # Filter universe
    progress_text.text("🔍 Filtering universe...")
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

    if not universe:
        st.warning("⚠️ No stocks found matching price criteria.")
        return empty_df(), empty_df(), empty_df(), (sess_label, win_start, win_end)

    # Process symbols with progress tracking
    rows = []
    total_symbols = len(universe)
    batch_size = 50  # Process in batches to manage memory
    
    for batch_start in range(0, total_symbols, batch_size):
        batch_end = min(batch_start + batch_size, total_symbols)
        batch = universe[batch_start:batch_end]
        
        progress_text.text(f"📈 Processing symbols {batch_start + 1}-{batch_end} of {total_symbols}...")
        progress = (batch_end) / total_symbols
        progress_bar.progress(progress)
        
        for i, sym in enumerate(batch):
            try:
                # Rate limiting: small delay every few requests
                if (batch_start + i) % 10 == 0 and (batch_start + i) > 0:
                    time.sleep(0.3)  # Reduced delay
                
                pclose = fetch_prev_close(sym)
                if not pclose or pclose <= 0:
                    continue

                last, wvol = fetch_minute_window(sym, win_start, win_end)
                if last is None or wvol < min_window_vol:
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
                    "Charts": f"[TV]({tradingview_link(sym)}) | [FV]({finviz_link(sym)})",
                })
                
            except Exception as e:
                # Skip problematic symbols silently
                continue

    # Clean up progress indicators
    progress_text.empty()
    progress_bar.empty()

    if not rows:
        st.warning("⚠️ No movers found matching your criteria. Try lowering the volume threshold or including OTC.")
        return empty_df(), empty_df(), empty_df(), (sess_label, win_start, win_end)

    # Create dataframe
    df = pd.DataFrame(rows)
    
    # Ensure schema
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = []

    df = df[EXPECTED_COLS]

    if df.empty:
        return df, empty_df(), empty_df(), (sess_label, win_start, win_end)

    # Sort and split
    df = df.sort_values("CHG %", ascending=False, na_position="last")
    leaders = df[df["CHG %"] > 0].copy()
    laggards = df[df["CHG %"] < 0].copy().sort_values("CHG %", ascending=True)

    return df, leaders, laggards, (sess_label, win_start, win_end)

# -------------- Enhanced UI --------------

st.set_page_config(
    page_title="Enhanced Movers — Auto Session (Polygon)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# API Key status
api_key = get_api_key()
if api_key:
    st.sidebar.success("✅ API Key Configured")
else:
    st.sidebar.error("❌ API Key Missing")

# Settings
include_otc = st.sidebar.checkbox("Include OTC", value=False, help="Include over-the-counter stocks")
min_price = st.sidebar.slider("Min Price ($)", min_value=1.0, max_value=50.0, value=MIN_PRICE_DEFAULT, step=0.5)
min_volume = st.sidebar.slider("Min Volume (M)", min_value=0.5, max_value=10.0, value=MIN_WINDOW_VOL_DEFAULT/1_000_000, step=0.5) * 1_000_000
topn = st.sidebar.slider("Show Top N", min_value=10, max_value=100, value=TOPN_DEFAULT, step=5)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (60s)", value=False)
refresh = st.sidebar.button("🔄 Refresh Now", type="primary")

# Display current time and session info
now_et = et_now()
sess_lbl, ws, we = session_window(now_et)

# Main title and info
st.title("⚡ Enhanced Market Movers — Live Session Detection")

# Time and session display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Current Time (ET)", now_et.strftime('%H:%M:%S'))
with col2:
    st.metric("Current Session", sess_lbl)
with col3:
    st.metric("Session Window", f"{ws.strftime('%H:%M')}–{we.strftime('%H:%M')}")

# Build movers data
if api_key:
    with st.spinner("🔄 Building movers data..."):
        df_all, leaders, laggards, (sess_lbl, ws, we) = build_movers_with_progress(
            min_price=min_price,
            min_window_vol=min_volume,
            include_otc=include_otc,
        )

    # Display results
    if df_all.empty:
        st.warning(
            "📊 **No movers found matching your criteria.**\n\n"
            "**Suggestions:**\n"
            "- Lower the minimum price or volume thresholds\n"
            "- Enable 'Include OTC' for more symbols\n"
            "- Check if markets are currently active\n"
            "- Verify your API key has sufficient quota"
        )
    else:
        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Movers", len(df_all))
        with col2:
            st.metric("Leaders", len(leaders))
        with col3:
            st.metric("Laggards", len(laggards))
        with col4:
            if not df_all.empty:
                avg_vol = df_all['Volume'].mean()
                st.metric("Avg Volume", f"{avg_vol:,.0f}")

        # Top movers tables
        leaders_n = leaders.head(topn).copy()
        laggards_n = laggards.head(topn).copy()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🚀 Top Gainers")
            if not leaders_n.empty:
                st.dataframe(
                    leaders_n.style.format({
                        "Price": "{:.2f}",
                        "Prev Close": "{:.2f}",
                        "CHG": "{:.2f}",
                        "CHG %": "{:.2f}",
                        "Volume": "{:,}",
                    }),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info("No gainers found")

        with col2:
            st.subheader("📉 Top Losers")
            if not laggards_n.empty:
                st.dataframe(
                    laggards_n.style.format({
                        "Price": "{:.2f}",
                        "Prev Close": "{:.2f}",
                        "CHG": "{:.2f}",
                        "CHG %": "{:.2f}",
                        "Volume": "{:,}",
                    }),
                    use_container_width=True,
                    height=400,
                )
            else:
                st.info("No losers found")

        # Full data expandable section
        st.markdown("---")
        with st.expander("📋 Full Dataset", expanded=False):
            st.dataframe(
                df_all.style.format({
                    "Price": "{:.2f}",
                    "Prev Close": "{:.2f}",
                    "CHG": "{:.2f}",
                    "CHG %": "{:.2f}",
                    "Volume": "{:,}",
                }),
                use_container_width=True,
                height=500,
            )

else:
    st.error("⚠️ Please configure your Polygon.io API key to continue.")

# Footer
st.markdown("---")
st.caption(
    "📊 **Data Source**: Polygon.io • "
    "💰 **Price**: Last trade in session window vs previous close • "
    "📈 **Volume**: Sum of minute bars in session window • "
    "🔗 **Charts**: TV = TradingView, FV = Finviz"
)

# Auto-refresh logic
if auto_refresh and api_key:
    time.sleep(60)
    st.rerun()
