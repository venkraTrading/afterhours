# afterhours_movers_live.py
# Simplified and robust Polygon.io market movers for Streamlit

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz
import streamlit as st

# -------------- Configuration --------------

def get_api_key():
    """Get API key with multiple fallback options"""
    # Try Streamlit secrets first
    try:
        return st.secrets["POLYGON_API_KEY"]
    except:
        pass
    
    # Try environment variable
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if api_key:
        return api_key
    
    # Fallback to user input
    return None

# Constants
ET = pytz.timezone("America/New_York")
MIN_PRICE_DEFAULT = 5.0
MIN_VOLUME_DEFAULT = 1_000_000
UNIVERSE_CAP = 500  # Reduced for reliability

# -------------- Helper Functions --------------

def get_et_now():
    """Get current time in Eastern timezone"""
    return datetime.now(ET)

def get_session_info(now_et):
    """Determine current market session"""
    current_time = now_et.time()
    date = now_et.date()
    
    # Define session times
    pre_start = datetime.combine(date, datetime.min.time().replace(hour=4)).replace(tzinfo=ET)
    market_open = datetime.combine(date, datetime.min.time().replace(hour=9, minute=30)).replace(tzinfo=ET)
    market_close = datetime.combine(date, datetime.min.time().replace(hour=16)).replace(tzinfo=ET)
    after_end = datetime.combine(date, datetime.min.time().replace(hour=20)).replace(tzinfo=ET)
    
    if pre_start.time() <= current_time < market_open.time():
        return "Pre-Market", pre_start, market_open
    elif market_open.time() <= current_time < market_close.time():
        return "Regular Hours", market_open, market_close
    elif market_close.time() <= current_time < after_end.time():
        return "After Hours", market_close, after_end
    else:
        return "Market Closed", market_close, after_end

def make_api_request(url, params=None, max_retries=3):
    """Make API request with retry logic"""
    params = params or {}
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 429:  # Rate limited
                wait_time = min(2 ** attempt, 10)
                time.sleep(wait_time)
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(1)
    
    return None

def get_stock_universe(api_key, date_str):
    """Get list of stocks from grouped daily bars"""
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}"
    params = {"apiKey": api_key, "adjusted": "true"}
    
    data = make_api_request(url, params)
    if not data or not data.get("results"):
        return []
    
    # Filter and limit universe
    stocks = []
    for result in data["results"]:
        symbol = result.get("T")
        close_price = result.get("c", 0)
        
        if symbol and close_price >= MIN_PRICE_DEFAULT:
            stocks.append(symbol)
            
        if len(stocks) >= UNIVERSE_CAP:
            break
    
    return stocks

def get_prev_close(api_key, symbol):
    """Get previous close price for a symbol"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev"
    params = {"apiKey": api_key, "adjusted": "true"}
    
    data = make_api_request(url, params)
    if data and data.get("results"):
        return data["results"][0].get("c")
    return None

def get_session_data(api_key, symbol, start_time, end_time):
    """Get minute data for a session window"""
    start_date = start_time.strftime("%Y-%m-%d")
    end_date = end_time.strftime("%Y-%m-%d")
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
    params = {
        "apiKey": api_key,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000
    }
    
    data = make_api_request(url, params)
    if not data or not data.get("results"):
        return None, 0
    
    # Filter bars within session window
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    session_bars = [
        bar for bar in data["results"] 
        if start_ms <= bar.get("t", 0) < end_ms
    ]
    
    if not session_bars:
        return None, 0
    
    last_price = session_bars[-1].get("c")
    total_volume = sum(bar.get("v", 0) for bar in session_bars)
    
    return last_price, total_volume

# -------------- Main App --------------

def main():
    st.set_page_config(
        page_title="Market Movers - Live",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Live Market Movers")
    st.markdown("Real-time market data powered by Polygon.io")
    
    # Get API key
    api_key = get_api_key()
    
    if not api_key:
        st.error("🔑 **Polygon.io API Key Required**")
        st.markdown("""
        **Setup Instructions:**
        1. Get your free API key from [polygon.io](https://polygon.io)
        2. Add it to Streamlit Cloud secrets as `POLYGON_API_KEY`
        3. Or set environment variable `POLYGON_API_KEY`
        """)
        
        # Manual input as fallback
        manual_key = st.text_input("Or enter your API key here:", type="password")
        if manual_key:
            api_key = manual_key
        else:
            st.stop()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    include_otc = st.sidebar.checkbox("Include OTC", value=False)
    min_volume = st.sidebar.slider("Min Volume (M)", 0.5, 10.0, 1.0, 0.5) * 1_000_000
    top_n = st.sidebar.slider("Show Top N", 10, 50, 20, 5)
    
    if st.sidebar.button("🔄 Refresh", type="primary"):
        st.rerun()
    
    # Get current session info
    now_et = get_et_now()
    session_name, session_start, session_end = get_session_info(now_et)
    
    # Display current info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Time (ET)", now_et.strftime("%H:%M:%S"))
    with col2:
        st.metric("Session", session_name)
    with col3:
        st.metric("Window", f"{session_start.strftime('%H:%M')}-{session_end.strftime('%H:%M')}")
    
    st.markdown("---")
    
    # Get data
    with st.spinner("📊 Fetching market data..."):
        # Get yesterday's date for universe
        yesterday = (now_et - timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Get stock universe
        progress_text = st.empty()
        progress_text.text("Getting stock universe...")
        
        universe = get_stock_universe(api_key, yesterday)
        if not universe:
            st.error("❌ Could not fetch stock universe. Please check your API key.")
            st.stop()
        
        progress_text.text(f"Processing {len(universe)} stocks...")
        
        # Process stocks
        movers_data = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(universe):
            # Update progress
            progress = (i + 1) / len(universe)
            progress_bar.progress(progress)
            
            # Rate limiting
            if i > 0 and i % 10 == 0:
                time.sleep(0.5)
            
            try:
                # Get previous close
                prev_close = get_prev_close(api_key, symbol)
                if not prev_close:
                    continue
                
                # Get session data
                current_price, volume = get_session_data(api_key, symbol, session_start, session_end)
                if not current_price or volume < min_volume:
                    continue
                
                # Calculate change
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100
                
                movers_data.append({
                    "Symbol": symbol,
                    "Price": round(current_price, 2),
                    "Prev Close": round(prev_close, 2),
                    "Change": round(change, 2),
                    "Change %": round(change_pct, 2),
                    "Volume": int(volume),
                    "Chart": f"[View](https://finviz.com/quote.ashx?t={symbol})"
                })
                
            except Exception:
                continue
        
        # Clean up progress indicators
        progress_text.empty()
        progress_bar.empty()
    
    # Display results
    if not movers_data:
        st.warning("⚠️ No movers found matching criteria. Try lowering volume threshold.")
        st.stop()
    
    # Create DataFrame
    df = pd.DataFrame(movers_data)
    df = df.sort_values("Change %", ascending=False)
    
    # Split into gainers and losers
    gainers = df[df["Change %"] > 0].head(top_n)
    losers = df[df["Change %"] < 0].head(top_n)
    
    # Display summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Movers", len(df))
    with col2:
        st.metric("Gainers", len(gainers))
    with col3:
        st.metric("Losers", len(losers))
    
    # Display tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Top Gainers")
        if len(gainers) > 0:
            st.dataframe(
                gainers,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No gainers found")
    
    with col2:
        st.subheader("📉 Top Losers") 
        if len(losers) > 0:
            st.dataframe(
                losers.sort_values("Change %", ascending=True),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No losers found")
    
    # Full data
    with st.expander("📋 All Movers", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.caption(f"📊 Data from {session_name} session • Last updated: {now_et.strftime('%H:%M:%S')} ET")

if __name__ == "__main__":
    main()
