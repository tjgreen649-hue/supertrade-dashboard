import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pytz
from datetime import datetime, time, timedelta

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
    ["AAPL", "MSFT", "TSLA", "NVDA", "SPY", "QQQ" ]
)
# ===== Display Toggles =====
st.sidebar.subheader("Display")
show_volume = st.sidebar.checkbox("Show Volume", value=True)

days = st.sidebar.slider(
    "Days of history",
    min_value=7,
    max_value=180,
    value=60
)
st.sidebar.subheader("Timeframes")

TIMEFRAMES = {
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "60m",
    "1d": "1d"
}

enabled_timeframes = {
    tf: st.sidebar.checkbox(tf, value=(tf == "5m"))
    for tf in TIMEFRAMES
}

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

# =========================
# MARKET SESSION LOGIC (NYSE)
# ========================= 
import pytz
from datetime import datetime, time

ny_tz = pytz.timezone("America/New_York")
now_ny = datetime.now(ny_tz).time()

MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

in_market = MARKET_OPEN <= now_ny <= MARKET_CLOSE

# ===============================
# FACTOR SIGNALS
# ===============================
# =========================
# MULTI-TIMEFRAME BIAS
# =========================
def timeframe_bias(close_series: pd.Series) -> int:
    if len(close_series) < 50:
        return 0

    show_volume = st.checkbox("Show Volume", value=True)

    sma_20 = close_series.rolling(20).mean()
    sma_50 = close_series.rolling(50).mean()

    if sma_20.iloc[-1] > sma_50.iloc[-1]:
        return 1   # Bullish bias
    elif sma_20.iloc[-1] < sma_50.iloc[-1]:
        return -1  # Bearish bias
    return 0

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
# --- FIX: Set Date as timezone-aware index ---
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.set_index("Date")

# ==========================
# INDICATOR CALCULATIONS
# ==========================

df["SMA_20"] = sma(df["Price"], 20)
df["EMA_20"] = ema(df["Price"], 20)
df["EMA_50"] = ema(df["Price"], 50)
df["RSI_14"] = rsi(df["Price"], 14)

bias = timeframe_bias(df["Price"])
from datetime import time
import pytz

ny_tz = pytz.timezone("America/New_York")

df["NY_Time"] = df.index.tz_convert(ny_tz)
df["NY_Date"] = df["NY_Time"].dt.date
df["NY_Clock"] = df["NY_Time"].dt.time

ENTRY_START = time(9, 0)
ENTRY_END   = time(15, 30)

df["Valid_Trade_Window"] = df["NY_Clock"].between(
    ENTRY_START,
    ENTRY_END
)

# =====================
# COLOR SCALE FOR SUPER TRADES
# =====================

def supertrade_color(confidence):
    if confidence < 0.5:
        return "rgba(255,255,255,0.2)"  # weak / faded
    elif confidence < 0.75:
        return "rgba(255,165,0,0.6)"    # orange
    elif confidence < 0.9:
        return "rgba(255,140,0,0.85)"   # deep orange
    else:
        return "rgba(255,200,0,1.0)"    # golden orange

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
    df["RSI_signal"] * weights["RSI_signal"] +
    bias
)
df["Confidence"] = df["Factor_Score"].abs() / (
    sum(weights.values()) + abs(bias)
)

df["Confidence"] = df["Confidence"].clip(0, 1)

# =====================
# FACTOR SCORE
# =====================
( df["Factor_Score"] = (
    df["SMA_signal"] * weights["SMA_signal"] +
    df["EMA_signal"] * weights["EMA_signal"] +
    df["RSI_signal"] * weights["RSI_signal"]
)

# =========================
# TRADE TYPE CLASSIFICATION
# =========================
df["Trade_Type"] = np.where (
    df["Profit_Confidence"] >= 0.5,
    "SUPER",
    "NORMAL"
)

df["SMA_signal"] * weights["SMA_signal"] ( +
    df["EMA_signal"] * weights["EMA_signal"] +
    df["RSI_signal"] * weights["RSI_signal"]
)
# =====================
# PROFIT CONFIDENCE SCORE (0.0 → 1.0)
# =====================

df["Profit_Confidence"] = (
df["Factor_Score"].abs() / 4
).clip(0, 1)

df["Trade_Type"] = np.where(
    df["Profit_Confidence"] >= 0.5,
    "SUPER",
    "NORMAL"
)

# =====================
# SUPER TRADE FLAG
# =====================
# Trade Type Classification
df["Trade_Type"] = np.where(
    df["Profit_Confidence"] >= 0.5,
    "SUPER",
    "NORMAL"
)

df["Volume_Confirm"] = (
    df["Volume"] > df["Volume"].rolling(20).mean()
)
df["Glow"] = np.where(  
    (df["Trade_Type"] == "SUPER") & (df["Volume_Confirm"]),
     "gold",
     "transparent"
)

df["Super_Trade"] = (
df["Profit_Confidence"] >= 0.5 
).clip(0, 1)
# ======================
# TRADE ACTION
# ======================

df["Trade_Action"] = "HOLD"

df.loc[
    (df["Factor_Score"] >= 1.5) & (df["Super_Trade"]),
    "Trade_Action"
] = "BUY"

df.loc[
    (df["Factor_Score"] <= -1.5) & (df["Super_Trade"]),
    "Trade_Action"
] = "SELL"
df["Profit_Window"] = "None"

df.loc[
    (df["Trade_Action"] != "HOLD") &
    (df["Profit_Confidence"] >= 0.75),
    "Profit_Window"
] = "High (1–3 candles)"

df.loc[
    (df["Trade_Action"] != "HOLD") &
    (df["Profit_Confidence"] < 0.75),
    "Profit_Window"
] = "Moderate (3–6 candles)"

# =====================
# FINAL TRADE SIGNAL
# =====================

def trade_signal(score):
    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    else:
        return "HOLD"

df["Trade_Signal"] = df["Factor_Score"].apply(trade_signal)

st.subheader("📊 Strategy Signal")

latest_score = df["Factor_Score"].iloc[-1]

if latest_score > 0:
    st.success(f"BULLISH 📈 (Score: {latest_score})")
elif latest_score < 0:
    st.error(f"BEARISH 📉 (Score: {latest_score})")
else:
    st.info("NEUTRAL ⚖️")
# =========================
# Get In or Out  SIGNALS
# =========================

df["signal"] = 0
df.loc[df["Factor_Score"] > 0, "signal"] = 1
df.loc[df["Factor_Score"] < 0, "signal"] = -1

df["signal_shift"] = df["signal"].shift(1)

df["entry"] = (df["signal"] == 1) & (df["signal_shift"] != 1)
df["exit"]  = (df["signal"] == -1) & (df["signal_shift"] != -1)
def classify_trade(row):
    if row["Factor_Score"] >= 80 and row["Volume_Confirm"]:
        return "SUPER"
    elif row["Trade_Action"] in ["BUY", "SELL"]:
        return "NORMAL"
    return "NONE"

df["Trade_Type"] = df.apply(classify_trade, axis=1)

# === PAPER TRADING ENGINE ===

balance = starting_balance
position = 0
entry_price = 0.0
equity_curve = []
trade_log = []

for i, row in df.iterrows():
    price = row["Price"]

    # BUY
    if row["entry"] and position == 0:
        position = balance / price
        entry_price = price
        balance = 0

        trade_log.append({
            "Date": row.name.tz_convert("America/New_York"),
            "Type": "BUY",
            "Price": price
        })

    # SELL
    elif row["exit"] and position > 0:
        balance = position * price
        pnl = balance - (position * entry_price)
        position = 0

        trade_log.append({
            "Date": row.name.tz_convert("America/New_York"),

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
col2.metric("Last Price", f"${price:.2f}")
if len(df) >= 2:
    day_change = df["Price"].iloc[-1] - df["Price"].iloc[-2]
else:
    day_change = 0.0

col3.metric("Day Change", f"{day_change:.2f}")
col4.metric("Balance", f"${starting_balance:,.0f}")
df["Candle_Color"] = np.where(df["Price"] >= df["Price"], "green", "red")

df["Volume_Color"] = np.select(
    [
        df["Trade_Action"] == "BUY",
        df["Trade_Action"] == "SELL"
    ],
    ["lime", "red"],
    default="gray"
)

# -----------------------------
# Price Chart
# -----------------------------
st.subheader("Price Chart")
def supertrade_color(conf):
    if conf < 0.5:
        return "gray"
    r = int(255)
    g = int(165 + (90 * conf))  # orange → gold
    b = int(0)
    return f"rgb({r},{g},{b})"
df["Candle_Color"] = df.apply(
    lambda r: supertrade_color(r["Profit_Confidence"])
    if r["Super_Trade"] else "gray",
    axis=1
)

# === PRICE CHART ===
price_fig = px.line(
    df,
    x=df.index,
    y="Price",
    title=f"{symbol} Price History",
    markers=True
)

price_fig.update_layout(
    xaxis_title="Date / Time (NYSE)",
    yaxis_title="Price",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)

# === TRADE MARKERS (ENTRIES / EXITS) ===
# === TRADE MARKERS (ENTRIES / EXITS) ===
entries = df[df["entry"] == 1]
exits   = df[df["exit"] == 1]

if not entries.empty:
    price_fig.add_scatter(
        x=entries.index,
        y=entries["Price"],
        mode="markers",
        name="Entry",
        marker=dict(symbol="triangle-up", size=12, color="green")
    )

if not exits.empty:
    price_fig.add_scatter(
        x=exits.index,
        y=exits["Price"],
        mode="markers",
        name="Exit",
        marker=dict(symbol="triangle-down", size=12, color="red")
    )
price_fig.update_layout(
    xaxis_title="Date / Time (NYSE)",
    yaxis_title="Price",
    height=400,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(price_fig, use_container_width=True)

# ==============================
# FACTOR SCORE PLOT
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

# =========================
# MARKET STRUCTURE + ORDER BLOCKS
# =========================

# Market Structure Break (simple swing logic)
df["swing_high"] = df["Price"] > df["Price"].shift(1)
df["swing_low"] = df["Price"] < df["Price"].shift(1)

df["bull_msb"] = (
    df["swing_high"] &
    (df["Price"] > df["Price"].rolling(5).max().shift(1))
)

# Bullish Order Block (last down candle before impulse)
df["bull_ob_low"] = df["Price"].shift(1)
df["bull_ob_high"] = df["Price"].shift(1)

# =========================
# BUYERS VS SELLERS (Volume Pressure)
# =========================

buy_vol = df["Volume"].where(df["Price"] > df["Price"], 0)
sell_vol = df["Volume"].where(df["Price"] < df["Price"], 0)

df["buyers_pct"] = 100 * buy_vol / (buy_vol + sell_vol)
df["buyers_pct"] = df["buyers_pct"].fillna(50)
price = df.loc[i, "Price"]
buyers_pct = df.loc[i, "buyers_pct"]
# --- SAFETY CHECK ---
df = df.copy()

if "Volume" not in df.columns:
    st.error(f"Volume column missing. Columns found: {list(df.columns)}")
    st.stop()

# Ensure datetime index
if not isinstance(df.index, pd.DatetimeIndex):
    df.index = pd.to_datetime(df.index)

# ==========================
# Volume Chart (SAFE)
# ==========================
st.subheader("Volume")

if show_volume and "Volume" in df.columns and not df.empty:
    vol_fig = px.bar(
    df,
    x=df.index,
    y="Volume",
    color="Vol_Color",
    color_discrete_map={"green": "green", "red": "red"},
    title=f"{symbol} Volume"
)
    
    vol_fig.update_layout(
        xaxis_title="Date / Time (NYSE)",
        yaxis_title="Volume",
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(vol_fig, use_container_width=True)
else:
    st.info("Volume hidden or data unavailable.")

# -----------------------------
# Trade Log (Mock)
# -----------------------------
st.subheader("Trade Log")

# =========================
# SAFE TRADE LOG DATA
# ==========================
# SIGNAL-BASED TRADE ENGINE
# ==========================

position = None
entry_price = None
balance = st.session_state.get("balance", starting_balance)
trades = []

for i in range(10, len(df)):

    price = df["Price"].iloc[i]

    # BUY SIGNAL
    if (
        position is None
        and df["bull_msb"].iloc[i]
        and df["bull_ob_low"].iloc[i] <= price <= df["bull_ob_high"].iloc[i]
        and buyers_pct > 60
    ):
        position = "LONG"
        entry_price = price
        trades.append({
            "Date": df.index[i],
            "Type": "BUY",
            "Price": price,
            "PnL": 0
        })

    # SELL SIGNAL
    elif (
        position == "LONG"
        and (
            df["bear_msb"].iloc[i]
            or sellers_pct > 60
        )
    ):
        pnl = (price - entry_price) * 10
        balance += pnl

        trades.append({
            "Date": df.index[i],
            "Type": "SELL",
            "Price": price,
            "PnL": pnl
        })

        position = None
        entry_price = Price

st.session_state["balance"] = balance
exit_price = price
if entry_price is not None:
    pnl_value = exit_price - entry_price
else:
    pnl_value = 0.0


trade_log = pd.DataFrame({
    "Date": ["Entry", "Exit"],
    "Price": [entry_price, exit_price],
    "PnL": [
        "-",
        f"${pnl_value:.2f}"
    ]
})

st.dataframe(trade_log, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ This is a simulated dashboard. "
    "No real trades are executed."
)
