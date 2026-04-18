import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="S&P 500 Prediction Dashboard",
    layout="wide"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* Glassmorphism Card */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 20px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Titles */
.title {
    font-size: 40px;
    font-weight: 700;
}

.subtitle {
    font-size: 18px;
    color: #cfd8dc;
}

/* Metric Styling */
.metric {
    font-size: 22px;
    font-weight: bold;
}

.metric-label {
    font-size: 14px;
    color: #b0bec5;
}

/* Divider */
.divider {
    margin-top: 10px;
    margin-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown('<div class="title">📈 S&P 500 Prediction Dashboard</div>', unsafe_allow_html=True)
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
# 🏠 TAB 1: LANDING PAGE (ONLY THIS FILLED)
# =========================================================
with tabs[0]:

    # -------- Top Metrics Row --------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Current Price</div>
            <div class="metric">4,589.32</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Daily Change</div>
            <div class="metric">+0.82%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Volatility</div>
            <div class="metric">1.23%</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Volume</div>
            <div class="metric">3.1B</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- Main Chart Placeholder --------
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Price Trend (S&P 500)</div>
            <div class="divider"></div>
            <div style="height:300px; display:flex; align-items:center; justify-content:center;">
                Chart will be displayed here
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <div class="metric-label">Model Snapshot</div>
            <div class="divider"></div>
            <p>LSTM Accuracy: 72%</p>
            <p>XGBoost Accuracy: 76%</p>
            <p>Best Model: XGBoost</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------- Bottom Section --------
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
# EMPTY TABS (PLACEHOLDERS)
# =========================================================
with tabs[1]:
    st.empty()

with tabs[2]:
    st.empty()

with tabs[3]:
    st.empty()

with tabs[4]:
    st.empty()
