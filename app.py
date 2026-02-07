"""Streamlit web UI for the intraday trading analyzer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os

import streamlit as st

from intraday_analyzer.analyzer import IntradayAnalyzer
from intraday_analyzer.models import parse_input

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Intraday Trading Analyzer",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")


def _load_example(name: str) -> dict:
    path = os.path.join(EXAMPLES_DIR, name)
    with open(path) as f:
        return json.load(f)


def _color_for_action(action: str) -> str:
    return {"Buy": "green", "Sell": "red", "Hold": "orange"}.get(action, "gray")


# ---------------------------------------------------------------------------
# Sidebar — input mode selector
# ---------------------------------------------------------------------------

st.sidebar.title("Input Mode")
input_mode = st.sidebar.radio(
    "Choose how to provide data:",
    ["Interactive Form", "JSON Editor", "Load Example"],
)

# ---------------------------------------------------------------------------
# Build input data based on mode
# ---------------------------------------------------------------------------

input_data = None

if input_mode == "Load Example":
    examples = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".json")]
    if not examples:
        st.sidebar.warning("No example files found in examples/")
    else:
        choice = st.sidebar.selectbox("Select example", sorted(examples))
        input_data = _load_example(choice)
        st.sidebar.success(f"Loaded {choice}")
        st.sidebar.json(input_data)

elif input_mode == "JSON Editor":
    default_json = json.dumps(_load_example("bullish_aapl.json"), indent=2)
    raw_text = st.sidebar.text_area(
        "Paste or edit JSON input:",
        value=default_json,
        height=500,
    )
    try:
        input_data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        st.sidebar.error(f"Invalid JSON: {e}")

elif input_mode == "Interactive Form":
    st.sidebar.markdown("---")

    ticker = st.sidebar.text_input("Ticker", value="AAPL")
    current_price = st.sidebar.number_input("Current Price", value=185.00, step=0.01)

    # --- Price Summary ---
    st.sidebar.subheader("Price Summary")
    col_ps1, col_ps2 = st.sidebar.columns(2)
    ps_open = col_ps1.number_input("Open", value=183.50, step=0.01)
    ps_high = col_ps2.number_input("High", value=185.80, step=0.01)
    ps_low = col_ps1.number_input("Low", value=183.20, step=0.01)
    ps_close = col_ps2.number_input("Close", value=185.00, step=0.01)
    ps_volume = st.sidebar.number_input("Volume", value=52_000_000, step=100_000)
    ps_vwap = st.sidebar.number_input("VWAP", value=184.50, step=0.01)
    ps_avg_vol = st.sidebar.number_input("Avg Volume (optional)", value=40_000_000, step=100_000)

    # --- Support / Resistance ---
    st.sidebar.subheader("Support / Resistance")
    supports_str = st.sidebar.text_input("Supports (comma-separated)", "183.00, 181.50, 180.00")
    resistances_str = st.sidebar.text_input("Resistances (comma-separated)", "186.50, 188.00, 190.00")

    def _parse_levels(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    # --- News ---
    st.sidebar.subheader("News Sentiment")
    news_impact = st.sidebar.selectbox("Impact", ["None", "Low", "Medium", "High"])
    news_sentiment = st.sidebar.selectbox("Sentiment", ["Neutral", "Bullish", "Bearish"])
    news_headlines = st.sidebar.text_input("Headlines (comma-separated)", "")

    # --- Indicators (main area tabs) ---
    input_data = None  # will be built after form submit

# ---------------------------------------------------------------------------
# Interactive form — indicator inputs in main area
# ---------------------------------------------------------------------------

if input_mode == "Interactive Form":
    st.title("Intraday Trading Analyzer")
    st.markdown("Fill in the technical indicators for each timeframe, then click **Analyze**.")

    tf_tabs = st.tabs(["1h Indicators", "15m Indicators", "1m Indicators"])
    tf_values = {}

    defaults = {
        "1h": {"ema9": 185.50, "ema21": 184.00, "ema50": 182.00, "rsi14": 62.0,
               "macd_val": 0.50, "macd_sig": 0.05, "macd_hist": 0.45, "atr14": 1.20},
        "15m": {"ema9": 185.20, "ema21": 184.80, "ema50": 0.0, "rsi14": 58.0,
                "macd_val": 0.20, "macd_sig": 0.05, "macd_hist": 0.15, "atr14": 0.60},
        "1m": {"ema9": 185.05, "ema21": 184.95, "ema50": 0.0, "rsi14": 42.0,
               "macd_val": 0.03, "macd_sig": 0.01, "macd_hist": 0.02, "atr14": 0.10},
    }

    for tab, tf_key in zip(tf_tabs, ["1h", "15m", "1m"]):
        d = defaults[tf_key]
        with tab:
            c1, c2, c3, c4 = st.columns(4)
            ema9 = c1.number_input(f"EMA9 ({tf_key})", value=d["ema9"], step=0.01, key=f"{tf_key}_ema9")
            ema21 = c2.number_input(f"EMA21 ({tf_key})", value=d["ema21"], step=0.01, key=f"{tf_key}_ema21")
            ema50 = c3.number_input(f"EMA50 ({tf_key})", value=d["ema50"], step=0.01, key=f"{tf_key}_ema50")
            rsi14 = c4.number_input(f"RSI14 ({tf_key})", value=d["rsi14"], min_value=0.0, max_value=100.0, step=0.1, key=f"{tf_key}_rsi14")

            c5, c6, c7, c8 = st.columns(4)
            macd_val = c5.number_input(f"MACD value ({tf_key})", value=d["macd_val"], step=0.01, format="%.4f", key=f"{tf_key}_macd_v")
            macd_sig = c6.number_input(f"MACD signal ({tf_key})", value=d["macd_sig"], step=0.01, format="%.4f", key=f"{tf_key}_macd_s")
            macd_hist = c7.number_input(f"MACD hist ({tf_key})", value=d["macd_hist"], step=0.01, format="%.4f", key=f"{tf_key}_macd_h")
            atr14 = c8.number_input(f"ATR14 ({tf_key})", value=d["atr14"], step=0.01, key=f"{tf_key}_atr14")

            ind = {"ema9": ema9, "ema21": ema21, "rsi14": rsi14,
                   "macd": {"value": macd_val, "signal": macd_sig, "histogram": macd_hist},
                   "atr14": atr14}
            if ema50 != 0.0:
                ind["ema50"] = ema50
            tf_values[tf_key] = ind

    if st.button("Analyze", type="primary", use_container_width=True):
        input_data = {
            "ticker": ticker,
            "current_price": current_price,
            "indicators": tf_values,
            "price_summary": {
                "open": ps_open, "high": ps_high, "low": ps_low, "close": ps_close,
                "volume": int(ps_volume), "vwap": ps_vwap, "avg_volume": int(ps_avg_vol),
            },
            "support_resistance": {
                "supports": _parse_levels(supports_str),
                "resistances": _parse_levels(resistances_str),
            },
            "news_sentiment": {
                "impact": news_impact,
                "sentiment": news_sentiment,
                "headlines": [h.strip() for h in news_headlines.split(",") if h.strip()],
            },
        }

# ---------------------------------------------------------------------------
# Run analysis and display results
# ---------------------------------------------------------------------------

if input_mode != "Interactive Form":
    st.title("Intraday Trading Analyzer")

if input_data is not None:
    try:
        inp = parse_input(input_data)
        analyzer = IntradayAnalyzer()
        signal = analyzer.analyze(inp)
        result = signal.to_dict()

        # --- Header banner ---
        action = result["action"]
        color = _color_for_action(action)
        conf = result["confidence"]

        st.markdown("---")
        col_action, col_conf, col_rr, col_ticker = st.columns(4)
        col_ticker.metric("Ticker", result["ticker"])
        col_action.markdown(
            f"### :{color}[{action}]"
        )
        col_conf.metric("Confidence", f"{conf}%")
        col_rr.metric("Risk / Reward", result["risk_reward"] if result["risk_reward"] else "N/A")

        # --- Trade levels ---
        if action != "Hold":
            st.markdown("#### Trade Levels")
            lc1, lc2, lc3, lc4, lc5 = st.columns(5)
            lc1.metric("Entry", f"${result['entry']:.2f}")
            lc2.metric("Stop", f"${result['stop']:.2f}")
            lc3.metric("Target 1", f"${result['target1']:.2f}")
            lc4.metric("Target 2", f"${result['target2']:.2f}" if result["target2"] else "N/A")
            lc5.metric("Position %", f"{result['position_size_pct']}%")

        # --- Meta ---
        st.markdown("#### Multi-Timeframe Summary")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.info(f"**1h Trend:** {result['meta']['hourly_trend']}")
        mc2.info(f"**News Impact:** {result['meta']['news_impact']}")
        mc3.info(f"**News Conflict:** {'Yes' if result['meta']['news_sentiment_conflict'] else 'No'}")
        mc4.info(f"**15m Setup:** confirmed" if action != "Hold" else f"**15m Setup:** not confirmed")

        st.caption(f"15m detail: {result['meta']['fifteen_min_setup']}")
        st.caption(f"1m detail: {result['meta']['one_min_timing']}")

        # --- Reasons & Invalidations ---
        r_col, i_col = st.columns(2)
        with r_col:
            st.markdown("#### Reasons")
            for r in result["reasons"]:
                st.markdown(f"- {r}")
        with i_col:
            st.markdown("#### Invalidations")
            if result["invalidations"]:
                for inv in result["invalidations"]:
                    st.markdown(f"- {inv}")
            else:
                st.markdown("_None (Hold signal)_")

        # --- Raw JSON ---
        with st.expander("Raw JSON Output"):
            st.json(result)

    except Exception as e:
        st.error(f"Analysis error: {e}")

elif input_mode != "Interactive Form":
    st.info("Select an example or paste JSON to begin.")
