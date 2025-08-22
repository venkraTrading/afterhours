#!/usr/bin/env python3
# pages/1_Live_News.py
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import pandas as pd  # not strictly required, but handy if you expand
import requests
import streamlit as st
from pytz import timezone as pytz_tz

# ─────────────────────────── Setup ───────────────────────────
st.set_page_config(page_title="📰 Live Headlines (Polygon)", page_icon="📰", layout="wide")

API_KEY = os.getenv("POLYGON_API_KEY") or st.secrets.get("POLYGON_API_KEY")
if not API_KEY:
    st.error("Missing POLYGON_API_KEY. Add it under **Settings → Secrets** or export it in the environment.")
    st.stop()

NY = pytz_tz("America/New_York")
NEWS_API = "https://api.polygon.io/v2/reference/news"

st.title("📰 Live Headlines")
st.caption("Near-real-time headlines from Polygon News, tailored for trading. Polls the API and de-duplicates as you watch.")

# ───────────────────────── Controls ──────────────────────────
with st.sidebar:
    st.subheader("Filters")

    ticker_filter_on = st.toggle("Filter by tickers", value=False, help="Limit to comma-separated symbols below.")
    ticker_input = st.text_input("Tickers (comma-separated)", value="AAPL, NVDA, TSLA").upper().replace(" ", "")
    tickers: List[str] = [t for t in ticker_input.split(",") if t] if ticker_filter_on else []

    include_kw = st.text_input("Must include keyword(s)", value="", help="Comma-separated; match any.")
    exclude_kw = st.text_input("Exclude keyword(s)", value="", help="Comma-separated; filter out if any match.")

    pub_allow_on = st.toggle("Publisher whitelist", value=False)
    pub_allow = st.text_input("Publishers (comma-separated)", value="Bloomberg, Reuters, MarketWatch") if pub_allow_on else ""

    min_sent = st.select_slider(
        "Min sentiment (overall)", options=["Any", "Neutral", "Positive"], value="Any",
        help="Uses Polygon’s overall sentiment when available."
    )

    st.subheader("Time & Refresh")
    lookback_min = st.slider("Lookback minutes", 5, 180, 45)
    refresh_sec = st.slider("Auto-refresh (sec)", 3, 30, 7)

include_terms = [s.strip().lower() for s in include_kw.split(",") if s.strip()]
exclude_terms = [s.strip().lower() for s in exclude_kw.split(",") if s.strip()]
allow_publishers = [s.strip().lower() for s in pub_allow.split(",") if s.strip()] if pub_allow_on else []

# ───────────────────── Helpers & Fetchers ────────────────────
def contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def rel_time(utc_iso: str) -> str:
    try:
        dt = datetime.fromisoformat(utc_iso.replace("Z", "")).replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        s = int(delta.total_seconds())
        if s < 60:
            return f"{s}s ago"
        m = s // 60
        if m < 60:
            return f"{m}m ago"
        h = m // 60
        return f"{h}h {m % 60}m ago"
    except Exception:
        return utc_iso

def badge(text: str, color: str) -> str:
    return (
        "<span style='"
        f"background:{color}; color:#fff; padding:2px 8px; border-radius:10px; font-size:0.75rem;"
        f"'>{text}</span>"
    )

def quick_links(symbols: List[str], url: str) -> str:
    first = symbols[0] if symbols else ""
    finviz = f"https://finviz.com/quote.ashx?t={first}" if first else None
    yahoo = f"https://finance.yahoo.com/quote/{first}" if first else None
    parts = [f"<a href='{url}' target='_blank'>Article</a>"]
    if finviz:
        parts.append(f"<a href='{finviz}' target='_blank'>Finviz</a>")
    if yahoo:
        parts.append(f"<a href='{yahoo}' target='_blank'>Yahoo</a>")
    return " • ".join(parts)

def extract_sentiment_label(record: dict) -> str:
    """
    Try to extract a single human-readable sentiment label from Polygon news.
    Handles cases where 'insights' is a dict, a list, or missing.
    Returns '' (empty) when nothing sensible is found.
    """
    insights = record.get("insights")

    # A) insights is a dict
    if isinstance(insights, dict):
        label = insights.get("overall_sentiment_label") or insights.get("overall_sentiment")
        if isinstance(label, str):
            return label

    # B) insights is a list
    if isinstance(insights, list) and insights:
        for item in insights:
            if isinstance(item, dict):
                label = (
                    item.get("overall_sentiment_label")
                    or item.get("overall_sentiment")
                    or item.get("sentiment")
                )
                if isinstance(label, str) and label:
                    return label

    # C) sometimes label might be at top level (rare)
    for k in ("overall_sentiment_label", "overall_sentiment"):
        v = record.get(k)
        if isinstance(v, str) and v:
            return v

    return ""

def label_color_css(label: str) -> str:
    """Pick a badge color for the sentiment label."""
    low = label.lower()
    if low.startswith("pos") or "bull" in low:
        return "#2e7d32"  # green
    if low.startswith("neg") or "bear" in low:
        return "#c62828"  # red
    if "neutral" in low or "somewhat" in low:
        return "#6a1b9a"  # purple-ish
    return "#6a1b9a"      # fallback

def meets_sentiment(r: Dict[str, Any]) -> bool:
    if min_sent == "Any":
        return True
    label = extract_sentiment_label(r).lower()
    if not label:
        # Treat unknown as not-positive for Positive; okay for Neutral.
        return min_sent != "Positive"
    if min_sent == "Neutral":
        return True  # any known label passes Neutral+
    if min_sent == "Positive":
        return ("pos" in label) or ("bull" in label)
    return True

@st.cache_data(ttl=0, show_spinner=False)
def fetch_news(since_iso: str, tickers: List[str], limit=50) -> List[Dict[str, Any]]:
    params = {
        "order": "desc",
        "limit": limit,
        "published_utc.gte": since_iso,
        "apiKey": API_KEY
    }
    if tickers:
        params["ticker"] = ",".join(tickers[:50])
    r = requests.get(NEWS_API, params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("results", []) or []

# ───────────────────── State & Time Window ───────────────────
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()

since_dt = datetime.now(timezone.utc) - timedelta(minutes=lookback_min)
since_iso = since_dt.isoformat()

# ─────────────────────────── Fetch ───────────────────────────
try:
    results = fetch_news(since_iso, tickers, limit=50)
except requests.HTTPError as e:
    st.error(f"HTTP {e.response.status_code}: {e.response.text[:300]}")
    st.stop()
except Exception as e:
    st.error(f"Fetch error: {e}")
    st.stop()

# Filter + Dedupe
clean: List[Dict[str, Any]] = []
new_count = 0

for r in results:
    rid = r.get("id") or r.get("article_url")
    if rid in st.session_state.seen_ids:
        continue

    title = (r.get("title") or "").strip()
    desc = (r.get("description") or "").strip()
    body = f"{title}\n{desc}"

    if include_terms and not contains_any(body, include_terms):
        continue
    if exclude_terms and contains_any(body, exclude_terms):
        continue
    if allow_publishers:
        pubname = (r.get("publisher") or {}).get("name", "").lower()
        if pubname not in allow_publishers:
            continue
    if not meets_sentiment(r):
        continue

    clean.append(r)
    new_count += 1

# Mark seen
for r in clean:
    rid = r.get("id") or r.get("article_url")
    st.session_state.seen_ids.add(rid)

# ────────────────────── Header Metrics ───────────────────────
left, mid, right = st.columns([1, 1, 2])
with left:
    st.metric("New since refresh", new_count)
with mid:
    st.metric("Total (lookback window)", len(clean))
with right:
    now_et = datetime.now(NY).strftime("%m/%d/%Y %H:%M:%S ET")
    st.caption(f"Last update: {now_et}")

# ───────────────────────── Render Cards ──────────────────────
if not clean:
    st.info("No headlines matched your filters in the current lookback window.")
else:
    for r in clean:
        title = r.get("title") or ""
        url = r.get("article_url") or r.get("amp_url") or "#"
        pub = (r.get("publisher") or {}).get("name") or "Unknown"
        tks = r.get("tickers") or []
        when = r.get("published_utc") or ""
        img = r.get("image_url")

        raw_label = extract_sentiment_label(r)
        sent_label = raw_label.title() if isinstance(raw_label, str) and raw_label else "—"
        sent_color = label_color_css(sent_label)

        # Card layout
        ch = st.container()
        with ch:
            c1, c2 = st.columns([6, 2])
            with c1:
                st.markdown(f"### {title}")
                info_row = []
                info_row.append(badge(pub, "#546e7a"))
                info_row.append(badge(sent_label, sent_color))
                info_row.append(badge(rel_time(when), "#37474f"))
                if tks:
                    tick_badge = badge(" ".join([f"${t}" for t in tks[:6]]), "#263238")
                    info_row.append(tick_badge)
                st.markdown(" ".join(info_row), unsafe_allow_html=True)

                st.markdown(quick_links(tks, url), unsafe_allow_html=True)

                if r.get("description"):
                    st.markdown(
                        f"<div style='color:#607d8b; margin-top:6px'>{r['description']}</div>",
                        unsafe_allow_html=True,
                    )
            with c2:
                if img:
                    st.image(img, use_column_width=True)

        st.divider()

# ───────────────────── Auto-Refresh (non-blocking) ───────────
st.autorefresh(interval=refresh_sec * 1000, key="news_autorefresh")
