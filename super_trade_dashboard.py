import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Super Trade Dashboard",
    layout="wide",
    page_icon="📈"
)

# -----------------------------
# Header
# -----------------------------
st.title("📈 Super Trade Dashboard")
st.caption("Paper trading dashboard • Educational use only")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")

symbol = st.sidebar.selectbox(
    "Select Symbol",
    ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"]
)

days = st.sidebar.slider(
    "Days of history",
    min_value=7,
    max_value=180,
    value=60
)

starting_balance = st.sidebar.number_input(
    "Starting Balance ($)",
    min_value=1000,
    max_value=1_000_000,
    value=25_000,
    step=1000
)
st.sidebar.subheader("📐 Factors")

show_sma = st.sidebar.checkbox("SMA (20)", value=True)
show_ema = st.sidebar.checkbox("EMA (20)")
show_rsi = st.sidebar.checkbox("RSI (14)")
show_macd = st.sidebar.checkbox("MACD")

# -----------------------------
# Generate Sample Price Data
# ----------------------------- 
#--------------------------
# FACTOR CALCULATIONS
# -------------------------

# Simple Moving Average
def sma(series, period=20):
    return series.rolling(period).mean()

# Exponential Moving Average
def ema(series, period=20):
    return series.ewm(span=period, adjust=False).mean()

# RSI
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# MACD
def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

np.random.seed(42)

dates = pd.date_range(
    end=datetime.today(),
    periods=days,
    freq="D"
)

price = np.cumsum(np.random.randn(days)) + 100
volume = np.random.randint(1_000_000, 5_000_000, size=days)

df = pd.DataFrame({
    "Date": dates,
    "Price": price,
    "Volume": volume
})

# -----------------------------
# Metrics Row
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Symbol", symbol)
col2.metric("Last Price", f"${price[-1]:.2f}")
col3.metric("Day Change", f"{price[-1] - price[-2]:.2f}")
col4.metric("Balance", f"${starting_balance:,.0f}")

# -----------------------------
# Price Chart
# -----------------------------
st.subheader("Price Chart")

price_fig = px.line(
    df,
    x="Date",
    y="Price",
    title=f"{symbol} Price History",
    markers=True
)

price_fig.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(price_fig, use_container_width=True)

# -----------------------------
# Volume Chart
# -----------------------------
st.subheader("Volume")

vol_fig = px.bar(
    df,
    x="Date",
    y="Volume",
    title=f"{symbol} Volume"
)

vol_fig.update_layout(
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(vol_fig, use_container_width=True)

# -----------------------------
# Trade Log (Mock)
# -----------------------------
st.subheader("Trade Log")

trades = pd.DataFrame({
    "Date": [
        datetime.today() - timedelta(days=5),
        datetime.today() - timedelta(days=2)
    ],
    "Symbol": [symbol, symbol],
    "Side": ["BUY", "SELL"],
    "Price": [price[-6], price[-3]],
    "Quantity": [10, 10],
    "PnL": ["—", f"${(price[-3] - price[-6]) * 10:.2f}"]
})

st.dataframe(trades, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ This is a simulated dashboard. "
    "No real trades are executed."
)
