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
# ===============================
# FACTOR SIGNALS
# ===============================

def sma_signal(price, sma):
    return 1 if price > sma else -1

def ema_signal(ema_fast, ema_slow):
    return 1 if ema_fast > ema_slow else -1

def rsi_signal(rsi):
    if rsi < 30:
        return 1
    elif rsi > 70:
        return -1
    return 0

def macd_signal(macd, signal):
    return 1 if macd > signal else -1

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
# ==========================
# INDICATOR CALCULATIONS
# ==========================

df["SMA_20"] = sma(df["Price"], 20)
df["EMA_20"] = ema(df["Price"], 20)
df["EMA_50"] = ema(df["Price"], 50)
df["RSI_14"] = rsi(df["Price"], 14)
# ==========================
# FACTOR SIGNALS
# ==========================

df["SMA_signal"] = df.apply(
    lambda x: sma_signal(x["Price"], x["SMA_20"]) if show_sma else 0,
    axis=1
)

df["EMA_signal"] = df.apply(
    lambda x: ema_signal(x["EMA_20"], x["EMA_50"]) if show_ema else 0,
    axis=1
)

df["RSI_signal"] = df["RSI_14"].apply(
    lambda x: rsi_signal(x) if show_rsi else 0
)
# ==========================
# FACTOR WEIGHTS
# ==========================

weights = {
    "SMA_signal": 1.0,
    "EMA_signal": 1.0,
    "RSI_signal": 1.0
}

df["Factor_Score"] = (
    df["SMA_signal"] * weights["SMA_signal"] +
    df["EMA_signal"] * weights["EMA_signal"] +
    df["RSI_signal"] * weights["RSI_signal"]
)
st.subheader("📊 Strategy Signal")

latest_score = df["Factor_Score"].iloc[-1]

if latest_score > 0:
    st.success(f"BULLISH 📈 (Score: {latest_score})")
elif latest_score < 0:
    st.error(f"BEARISH 📉 (Score: {latest_score})")
else:
    st.info("NEUTRAL ⚖️")
# =========================
# STEP 5: TRADE SIGNALS
# =========================

df["signal"] = 0
df.loc[df["Factor_Score"] > 0, "signal"] = 1
df.loc[df["Factor_Score"] < 0, "signal"] = -1

df["signal_shift"] = df["signal"].shift(1)

df["entry"] = (df["signal"] == 1) & (df["signal_shift"] != 1)
df["exit"]  = (df["signal"] == -1) & (df["signal_shift"] != -1)

# === PAPER TRADING ENGINE ===

balance = starting_balance
position = 0
entry_price = 0.0
equity_curve = []
trade_log = []

for i, row in df.iterrows():
    price = row["Close"]

    # BUY
    if row["entry"] and position == 0:
        position = balance / price
        entry_price = price
        balance = 0

        trade_log.append({
            "Date": row["Date"],
            "Type": "BUY",
            "Price": price
        })

    # SELL
    elif row["exit"] and position > 0:
        balance = position * price
        pnl = balance - (position * entry_price)
        position = 0

        trade_log.append({
            "Date": row["Date"],
            "Type": "SELL",
            "Price": price,
            "PnL": pnl
        })

    equity_curve.append(balance if position == 0 else position * price)

df["Equity"] = equity_curve
trades = pd.DataFrame(trade_log)

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
# === TRADE MARKERS (ENTRIES / EXITS) ===

entries = df[df["entry"]]
exits  = df[df["exit"]]

price_fig.add_scatter(
    x=entries["Date"],
    y=entries["Price"],
    mode="markers",
    marker=dict(symbol="triangle-up", size=14),
    name="Buy Entry"
)

price_fig.add_scatter(
    x=exits["Date"],
    y=exits["Price"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=14),
    name="Sell Exit"
)


price_fig.add_scatter(
    x=exits["Date"],
    y=exits["Price"],
    mode="markers",
    marker=dict(symbol="triangle-down", size=14),
    name="Sell Exit"
)

st.plotly_chart(price_fig, use_container_width=True)

# ==============================
# STEP 4: FACTOR SCORE PLOT
# ==============================

st.subheader("Factor Score")

if "factor_score" in df.columns and df["factor_score"].notna().any():

    score_fig = px.line(
        df,
        x="Date",
        y="factor_score",
        title="Composite Factor Score",
        markers=True
    )

    score_fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(score_fig, use_container_width=True)

else:
    st.info("Enable at least one factor to display Factor Score.")


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
