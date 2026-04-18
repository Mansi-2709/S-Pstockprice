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

/* ===== Background Gradient (FIXED) ===== */
.stApp {
    background: linear-gradient(135deg, #09a6d6, #003140, #001c26);
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

    # -------- Top Metrics --------
    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("Current Price", "4,589.32"),
        ("Daily Change", "+0.82%"),
        ("Volatility", "1.23%"),
        ("Volume", "3.1B")
    ]

    for col, (label, value) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">{label}</div>
                <div class="metric">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- Main Section --------
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Price Trend (S&P 500)</div>
            <div class="divider"></div>
            <div style="height:320px; display:flex; align-items:center; justify-content:center;">
                Chart will be displayed here
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Model Snapshot</div>
            <div class="divider"></div>
            <p>LSTM Accuracy: <b>72%</b></p>
            <p>XGBoost Accuracy: <b>76%</b></p>
            <p><b>Best Model:</b> XGBoost</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- Bottom Cards --------
    col1, col2, col3 = st.columns(3)

    bottom_data = [
        ("Top Sector", "Technology"),
        ("Worst Sector", "Energy"),
        ("Market Sentiment", "Bullish 📈")
    ]

    for col, (label, value) in zip([col1, col2, col3], bottom_data):
        with col:
            st.markdown(f"""
            <div class="glass-card">
                <div class="metric-label">{label}</div>
                <div class="metric">{value}</div>
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
