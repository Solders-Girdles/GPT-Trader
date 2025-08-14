#!/usr/bin/env python
"""
ML-Enhanced Trading Dashboard
Streamlit application for monitoring ML predictions and portfolio optimization
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import ML dashboard pages
from ml_pages import render_ml_predictions_page, render_portfolio_optimization_page

# Configure Streamlit page
st.set_page_config(
    page_title="GPT-Trader ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #e0e5ec 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-active {
        color: #00cc88;
        font-weight: bold;
    }
    .status-inactive {
        color: #ff4444;
        font-weight: bold;
    }
    .profit { color: #00cc88; }
    .loss { color: #ff4444; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_overview_page():
    """Render overview dashboard page"""
    st.markdown('<h1 class="main-header">🤖 GPT-Trader ML Dashboard</h1>', unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Portfolio Value", "$125,432", "+5.2%")
    with col2:
        st.metric("Today's P&L", "+$1,245", "+1.0%")
    with col3:
        st.metric("Active Positions", "6", "")
    with col4:
        st.metric("Win Rate", "58.3%", "+2.1%")
    with col5:
        st.metric("Sharpe Ratio", "1.24", "+0.08")

    # System status
    st.subheader("📊 System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ML Models")
        models = {
            "Regime Detector": "Active",
            "Strategy Selector": "Active",
            "Portfolio Optimizer": "Active",
        }
        for model, status in models.items():
            status_class = "status-active" if status == "Active" else "status-inactive"
            st.markdown(
                f'<div class="metric-card">📈 {model}: <span class="{status_class}">{status}</span></div>',
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("### Trading Status")
        trading_status = {
            "Auto-Trading": "Enabled",
            "Risk Limits": "Normal",
            "Rebalancing": "Scheduled",
        }
        for item, status in trading_status.items():
            color = "#00cc88" if status in ["Enabled", "Normal", "Scheduled"] else "#ff9900"
            st.markdown(
                f'<div class="metric-card">⚡ {item}: <span style="color: {color}">{status}</span></div>',
                unsafe_allow_html=True,
            )

    with col3:
        st.markdown("### Market Conditions")
        market_info = {"Market Regime": "Bull Quiet", "VIX Level": "14.2 (Low)", "Trend": "Upward"}
        for item, value in market_info.items():
            st.markdown(
                f'<div class="metric-card">📊 {item}: <b>{value}</b></div>', unsafe_allow_html=True
            )

    # Recent activity
    st.subheader("📈 Recent Activity")

    activity_data = pd.DataFrame(
        {
            "Time": pd.date_range(end=datetime.now(), periods=10, freq="H").strftime("%H:%M"),
            "Event": [
                "Regime Change",
                "Strategy Switch",
                "Trade Executed",
                "Rebalancing",
                "Risk Alert",
                "Trade Executed",
                "Model Update",
                "Trade Executed",
                "Strategy Switch",
                "Market Close",
            ],
            "Details": [
                "Bull Quiet → Bull Volatile",
                "Momentum → Trend Following",
                "Buy AAPL 100 shares",
                "5 positions adjusted",
                "Volatility spike detected",
                "Sell MSFT 50 shares",
                "Regime detector retrained",
                "Buy GOOGL 25 shares",
                "Trend Following → Mean Reversion",
                "Trading halted",
            ],
            "Impact": [
                "Medium",
                "High",
                "Low",
                "Medium",
                "High",
                "Low",
                "Low",
                "Low",
                "Medium",
                "N/A",
            ],
        }
    )

    st.dataframe(activity_data, use_container_width=True, height=300)

    # Performance chart
    st.subheader("📊 Performance Overview")

    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
    portfolio_values = 100000 * (1 + np.random.randn(30).cumsum() * 0.01)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values,
            mode="lines",
            name="Portfolio Value",
            line=dict(color="#1f77b4", width=3),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.1)",
        )
    )

    fig.update_layout(
        height=400,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_live_trading_page():
    """Render live trading monitor page"""
    st.title("🔴 Live Trading Monitor")

    # Trading controls
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⏸️ Pause Trading", type="secondary", use_container_width=True):
            st.warning("Trading paused")

    with col2:
        if st.button("🔄 Force Rebalance", type="secondary", use_container_width=True):
            st.info("Rebalancing initiated")

    with col3:
        if st.button("📊 Update Models", type="secondary", use_container_width=True):
            st.info("Model update scheduled")

    with col4:
        if st.button("🛑 Emergency Stop", type="primary", use_container_width=True):
            st.error("Emergency stop activated!")

    # Live positions
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Active Positions")

        positions_df = pd.DataFrame(
            {
                "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"],
                "Quantity": [100, 50, 75, 40, 60, 30],
                "Entry Price": [150.25, 380.50, 140.75, 180.00, 490.25, 890.50],
                "Current Price": [152.30, 385.20, 142.10, 178.50, 495.60, 905.30],
                "P&L": [205, 235, 101.25, -60, 321, 444],
                "P&L %": [1.36, 1.23, 0.96, -0.83, 1.09, 1.66],
            }
        )

        # Color code P&L
        def color_pnl(val):
            color = "green" if val > 0 else "red"
            return f"color: {color}"

        st.dataframe(
            positions_df.style.applymap(color_pnl, subset=["P&L", "P&L %"]).format(
                {
                    "Entry Price": "${:.2f}",
                    "Current Price": "${:.2f}",
                    "P&L": "${:.2f}",
                    "P&L %": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

    with col2:
        st.subheader("🎯 Next Actions")

        actions = [
            ("Buy Signal", "TSLA", "Strong", "🟢"),
            ("Sell Signal", "META", "Moderate", "🟡"),
            ("Stop Loss", "AMZN", "Triggered", "🔴"),
            ("Rebalance", "Portfolio", "In 2 hours", "🔵"),
        ]

        for action, symbol, strength, emoji in actions:
            st.markdown(
                f"""
            <div class="metric-card">
                {emoji} <b>{action}</b><br>
                <span style="font-size: 0.9em">{symbol} - {strength}</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Order book
    st.subheader("📋 Recent Orders")

    orders_df = pd.DataFrame(
        {
            "Time": pd.date_range(end=datetime.now(), periods=8, freq="30min").strftime("%H:%M:%S"),
            "Symbol": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "AAPL"],
            "Side": ["BUY", "BUY", "SELL", "BUY", "SELL", "SELL", "BUY", "SELL"],
            "Quantity": [100, 50, 25, 30, 40, 20, 75, 50],
            "Price": [152.30, 385.20, 142.10, 905.30, 495.60, 178.50, 245.80, 152.50],
            "Status": [
                "FILLED",
                "FILLED",
                "FILLED",
                "PENDING",
                "FILLED",
                "CANCELLED",
                "FILLED",
                "PENDING",
            ],
        }
    )

    # Color code status
    def color_status(val):
        colors = {"FILLED": "green", "PENDING": "orange", "CANCELLED": "red"}
        return f'color: {colors.get(val, "black")}'

    st.dataframe(
        orders_df.style.applymap(color_status, subset=["Status"]).format({"Price": "${:.2f}"}),
        use_container_width=True,
        height=300,
    )


def main():
    """Main dashboard application"""

    # Sidebar navigation
    st.sidebar.title("🤖 Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["Overview", "ML Predictions", "Portfolio Optimization", "Live Trading"],
        index=0,
    )

    # Render selected page
    if page == "Overview":
        render_overview_page()
    elif page == "ML Predictions":
        render_ml_predictions_page()
    elif page == "Portfolio Optimization":
        render_portfolio_optimization_page()
    elif page == "Live Trading":
        render_live_trading_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")

    # Settings controls
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        refresh_rate = st.sidebar.select_slider(
            "Refresh Rate", options=[5, 10, 30, 60], value=30, format_func=lambda x: f"{x} seconds"
        )

    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.metric("Total Trades Today", "42")
    st.sidebar.metric("Success Rate", "71.4%")
    st.sidebar.metric("Avg Trade Time", "2.3 min")

    # Version info
    st.sidebar.markdown("---")
    st.sidebar.caption("GPT-Trader v2.0.0")
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


if __name__ == "__main__":
    main()
