import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.markdown(
    "<div style='margin-top: 40px;'></div>",
    unsafe_allow_html=True
)
st.set_page_config(
    page_title="S&P 500 Prediction Dashboard",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

header[data-testid="stHeader"] {
    display: none;
}

/* ===== Background Gradient (FIXED) ===== */
.stApp {
    background: linear-gradient(180deg, #09a6d6, #003140, #001c26);
    color: #ffffff;
}

/* ===== Remove default padding ===== */
.block-container {
    padding-top: 2rem;
}

/* ===== Tabs Styling (FIXED VISIBILITY) ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 30px;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 600;
    color: #e0f7fa;  /* light teal text */
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #00e5ff;
}

/* ===== Glass Card ===== */
.glass-card {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 22px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.25);
}

/* ===== Title ===== */
.title {
    font-size: 44px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #d9f3f4;
}

/* ===== Metrics ===== */
.metric {
    font-size: 26px;
    font-weight: bold;
}

.metric-label {
    font-size: 14px;
    color: #e0f7fa;
}

/* Divider */
.divider {
    margin: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.2);
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">📊 S&P 500 Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">LSTM vs XGBoost | Time Series Forecasting</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------- TABS --------------------
tabs = st.tabs([
    "🏠 Overview",
    "📊 EDA",
    "🤖 Models",
    "⚖️ Comparison",
    "🔮 Forecast"
])

# =========================================================
# 🏠 TAB 1: LANDING PAGE
# =========================================================
with tabs[0]:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    # ---------------- DATA ----------------
    file_id = 'https://drive.google.com/file/d/1s5rg6SG7IoZ4poTNqAKXdh_tJT_S44-w/view?usp=sharing'
    url = f'https://google.com{file_id}'
    stock_data = pd.read_csv(url)
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    # ---------------- METRICS ----------------
    col1, col2, col3, col4 = st.columns(4)

    current_price = stock_data.sort_values('date').iloc[-1]['close']
    daily_change = stock_data.sort_values('date').iloc[-1]['close'] - stock_data.sort_values('date').iloc[-2]['close']
    volatility = stock_data['close'].pct_change().std()
    formatted_sum = "{:.2e}".format(stock_data['volume'].sum())

    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Current Price</div>
            <div class="metric">{round(current_price,2)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Daily Change</div>
            <div class="metric">{round(daily_change,2)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Volatility</div>
            <div class="metric">{round(volatility,4)}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-label">Volume</div>
            <div class="metric">{formatted_sum}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- TREEMAP ----------------
    latest = stock_data.sort_values('date').groupby('Name').tail(1)
    prev = stock_data.sort_values('date').groupby('Name').nth(-2).reset_index()

    merged = latest.merge(prev[['Name', 'close']], on='Name', suffixes=('', '_prev'))
    merged['change'] = ((merged['close'] - merged['close_prev']) / merged['close_prev']) * 100

    fig_tree = px.treemap(
        merged,
        path=['Name'],
        values='volume',
        color='change',
        color_continuous_scale=["#8B0000", "#111111", "#00C853"],
        color_continuous_midpoint=0
    )

    fig_tree.update_layout(
        paper_bgcolor="#0b1f24",
        plot_bgcolor="#0b1f24",
        font_color="white",
        margin=dict(t=40, l=10, r=10, b=10)
    )

    # ---------------- MAIN LAYOUT ----------------
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Market Heatmap</div>
            <div class="divider"></div>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_tree, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Model Snapshot</div>
            <div class="divider"></div>
            <p>LSTM Accuracy: 72%</p>
            <p>XGBoost Accuracy: 76%</p>
            <p><b>Best Model:</b> XGBoost</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- STOCK SELECTION + CANDLE ----------------
    stock_list = stock_data['Name'].unique()
    selected_stock = st.selectbox("Select Stock", stock_list)

    grouped = stock_data.groupby('Name')
    data1 = grouped.get_group(selected_stock)

    fig_candle = go.Figure(data=[go.Candlestick(
        x=data1['date'],
        open=data1['open'],
        high=data1['high'],
        low=data1['low'],
        close=data1['close']
    )])

    fig_candle.update_layout(
        paper_bgcolor="#0b1f24",
        plot_bgcolor="#0b1f24",
        font_color="white"
    )

    st.markdown("""
    <div class="glass-card">
        <div class="metric-label">Candlestick Chart</div>
        <div class="divider"></div>
    </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- BOTTOM SECTION ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Top Sector</div>
            <div class="metric">Technology</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Worst Sector</div>
            <div class="metric">Energy</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Market Sentiment</div>
            <div class="metric">Bullish 📈</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# EMPTY TABS
# =========================================================
with tabs[1]:
    st.empty()

with tabs[2]:
    st.empty()

with tabs[3]:
    st.empty()

with tabs[4]:
    st.empty()
