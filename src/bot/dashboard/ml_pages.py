"""
ML-specific dashboard pages for predictions and optimization
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_ml_predictions_page():
    """Render ML predictions dashboard page"""
    st.title("🤖 ML Predictions")

    # Page layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.subheader("📊 Market Regime")
        render_regime_detection()

    with col2:
        st.subheader("🎯 Strategy Selection")
        render_strategy_selection()

    with col3:
        st.subheader("📈 Confidence Metrics")
        render_confidence_metrics()

    # Historical predictions chart
    st.subheader("📈 Historical Predictions")
    render_prediction_history()

    # Feature importance
    st.subheader("🔍 Feature Analysis")
    render_feature_importance()


def render_regime_detection():
    """Render market regime detection panel"""

    # Mock data for demonstration
    current_regime = "Bull Quiet"
    confidence = 0.87

    # Regime indicator
    regime_colors = {
        "Bull Quiet": "#00cc88",
        "Bull Volatile": "#00ff00",
        "Bear Quiet": "#ff9900",
        "Bear Volatile": "#ff0000",
        "Sideways": "#999999",
    }

    color = regime_colors.get(current_regime, "#666666")

    st.markdown(
        f"""
    <div style='text-align: center; padding: 20px; background: {color}20; border-radius: 10px;'>
        <h2 style='color: {color}; margin: 0;'>{current_regime}</h2>
        <p style='margin: 5px 0;'>Confidence: {confidence:.1%}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Regime probabilities
    st.markdown("**Regime Probabilities:**")
    regimes = ["Bull Quiet", "Bull Volatile", "Bear Quiet", "Bear Volatile", "Sideways"]
    probabilities = [0.65, 0.15, 0.10, 0.05, 0.05]

    for regime, prob in zip(regimes, probabilities, strict=False):
        st.progress(prob)
        st.caption(f"{regime}: {prob:.1%}")

    # Transition matrix
    with st.expander("Transition Matrix"):
        transition_matrix = pd.DataFrame(
            np.random.rand(5, 5) * 0.2 + np.eye(5) * 0.8, index=regimes, columns=regimes
        )
        st.dataframe(transition_matrix.style.format("{:.2%}"))


def render_strategy_selection():
    """Render strategy selection panel"""

    # Current strategy selection
    strategies = {
        "Trend Following": 0.45,
        "Mean Reversion": 0.25,
        "Momentum": 0.15,
        "Breakout": 0.10,
        "Range Trading": 0.05,
    }

    # Create donut chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(strategies.keys()),
                values=list(strategies.values()),
                hole=0.4,
                marker_colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            )
        ]
    )

    fig.update_layout(height=300, showlegend=True, margin=dict(t=0, b=0, l=0, r=0))

    st.plotly_chart(fig, use_container_width=True)

    # Strategy performance table
    st.markdown("**Strategy Performance (Last 30 Days):**")

    perf_data = pd.DataFrame(
        {
            "Strategy": list(strategies.keys()),
            "Weight": list(strategies.values()),
            "Return": [0.052, 0.031, 0.045, 0.023, 0.018],
            "Sharpe": [1.2, 0.9, 1.1, 0.7, 0.6],
            "Win Rate": [0.58, 0.52, 0.55, 0.48, 0.51],
        }
    )

    st.dataframe(
        perf_data.style.format(
            {"Weight": "{:.1%}", "Return": "{:.1%}", "Sharpe": "{:.2f}", "Win Rate": "{:.1%}"}
        ),
        use_container_width=True,
    )


def render_confidence_metrics():
    """Render confidence metrics panel"""

    # Key metrics
    metrics = {
        "Model Accuracy": 0.82,
        "Prediction Confidence": 0.75,
        "Feature Quality": 0.91,
        "Data Freshness": 0.98,
    }

    for metric, value in metrics.items():
        st.metric(metric, f"{value:.1%}")

        # Color-coded progress bar
        color = "#00cc88" if value > 0.7 else "#ff9900" if value > 0.5 else "#ff0000"
        st.markdown(
            f"""
        <div style='background: #f0f0f0; border-radius: 5px; overflow: hidden;'>
            <div style='background: {color}; width: {value*100}%; height: 5px;'></div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.text("")  # Spacing

    # Model status
    st.markdown("**Model Status:**")

    models = {
        "Regime Detector": ("Active", "2024-01-15"),
        "Strategy Selector": ("Active", "2024-01-15"),
        "Portfolio Optimizer": ("Active", "2024-01-14"),
    }

    for model, (status, updated) in models.items():
        status_color = "#00cc88" if status == "Active" else "#ff0000"
        st.markdown(
            f"• {model}: <span style='color: {status_color}'>{status}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"  Last updated: {updated}")


def render_prediction_history():
    """Render historical predictions chart"""

    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq="D")

    # Create traces for each prediction type
    fig = go.Figure()

    # Regime predictions (as background colors)
    regime_data = np.random.choice([0, 1, 2, 3, 4], size=30, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    regime_names = ["Bull Quiet", "Bull Volatile", "Bear Quiet", "Bear Volatile", "Sideways"]
    colors = ["#00cc88", "#00ff00", "#ff9900", "#ff0000", "#999999"]

    for i, (regime, color) in enumerate(zip(regime_names, colors, strict=False)):
        mask = regime_data == i
        if mask.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[mask],
                    y=[1] * mask.sum(),
                    mode="markers",
                    name=regime,
                    marker=dict(size=15, color=color, symbol="square"),
                    yaxis="y2",
                )
            )

    # Strategy selection
    strategy_scores = np.random.randn(30).cumsum() + 50
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strategy_scores,
            mode="lines",
            name="Strategy Score",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # Confidence bands
    confidence = np.random.uniform(0.6, 0.9, 30)
    upper_band = strategy_scores + confidence * 5
    lower_band = strategy_scores - confidence * 5

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=upper_band,
            mode="lines",
            name="Upper Confidence",
            line=dict(color="rgba(31, 119, 180, 0.2)"),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=lower_band,
            mode="lines",
            name="Lower Confidence",
            line=dict(color="rgba(31, 119, 180, 0.2)"),
            fill="tonexty",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=400,
        title="ML Predictions Over Time",
        xaxis_title="Date",
        yaxis_title="Strategy Score",
        yaxis2=dict(title="Regime", overlaying="y", side="right", showticklabels=False),
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_feature_importance():
    """Render feature importance analysis"""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top Features - Regime Detection:**")

        regime_features = {
            "Volatility (20d)": 0.18,
            "Returns (5d)": 0.15,
            "Volume Ratio": 0.12,
            "RSI": 0.10,
            "MACD Signal": 0.08,
            "Trend Strength": 0.07,
            "Market Beta": 0.06,
            "Others": 0.24,
        }

        fig = px.bar(
            x=list(regime_features.values()),
            y=list(regime_features.keys()),
            orientation="h",
            color=list(regime_features.values()),
            color_continuous_scale="Blues",
        )

        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Importance",
            yaxis_title="",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Top Features - Strategy Selection:**")

        strategy_features = {
            "RSI": 0.20,
            "Trend Strength": 0.16,
            "ADX": 0.14,
            "Bollinger %": 0.11,
            "Volume": 0.09,
            "ATR": 0.08,
            "Momentum": 0.07,
            "Others": 0.15,
        }

        fig = px.bar(
            x=list(strategy_features.values()),
            y=list(strategy_features.keys()),
            orientation="h",
            color=list(strategy_features.values()),
            color_continuous_scale="Greens",
        )

        fig.update_layout(
            height=300,
            showlegend=False,
            xaxis_title="Importance",
            yaxis_title="",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)


def render_portfolio_optimization_page():
    """Render portfolio optimization dashboard page"""
    st.title("📊 Portfolio Optimization")

    # Optimization controls
    with st.sidebar:
        st.subheader("Optimization Settings")

        objective = st.selectbox(
            "Objective", ["Max Sharpe", "Min Risk", "Max Return", "Risk Parity"]
        )

        risk_level = st.slider("Risk Tolerance", 0.0, 1.0, 0.5)

        constraints = st.multiselect(
            "Constraints",
            ["Long Only", "Max Position 30%", "Min Position 5%", "Sector Limits"],
            default=["Long Only", "Max Position 30%"],
        )

        rebalance_freq = st.selectbox(
            "Rebalance Frequency", ["Daily", "Weekly", "Monthly", "Quarterly"], index=2
        )

        if st.button("🔄 Re-optimize", type="primary"):
            st.success("Portfolio re-optimized!")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📈 Efficient Frontier")
        render_efficient_frontier()

    with col2:
        st.subheader("🎯 Current Allocation")
        render_current_allocation()

    # Performance metrics
    st.subheader("📊 Portfolio Metrics")
    render_portfolio_metrics()

    # Rebalancing analysis
    st.subheader("🔄 Rebalancing Analysis")
    render_rebalancing_analysis()


def render_efficient_frontier():
    """Render efficient frontier chart"""

    # Generate sample efficient frontier
    n_portfolios = 50
    returns = np.linspace(0.05, 0.25, n_portfolios)
    risks = np.sqrt(returns * 0.5) + np.random.normal(0, 0.01, n_portfolios)

    # Current portfolio
    current_return = 0.15
    current_risk = 0.18

    # Optimal portfolio
    optimal_idx = np.argmax(returns / risks)

    fig = go.Figure()

    # Efficient frontier
    fig.add_trace(
        go.Scatter(
            x=risks,
            y=returns,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # Random portfolios
    random_risks = np.random.uniform(0.1, 0.35, 200)
    random_returns = np.random.uniform(0.02, 0.20, 200)

    fig.add_trace(
        go.Scatter(
            x=random_risks,
            y=random_returns,
            mode="markers",
            name="Possible Portfolios",
            marker=dict(size=3, color="lightgray", opacity=0.5),
        )
    )

    # Current portfolio
    fig.add_trace(
        go.Scatter(
            x=[current_risk],
            y=[current_return],
            mode="markers",
            name="Current Portfolio",
            marker=dict(size=15, color="#ff7f0e", symbol="star"),
        )
    )

    # Optimal portfolio
    fig.add_trace(
        go.Scatter(
            x=[risks[optimal_idx]],
            y=[returns[optimal_idx]],
            mode="markers",
            name="Optimal Portfolio",
            marker=dict(size=15, color="#2ca02c", symbol="diamond"),
        )
    )

    fig.update_layout(
        height=400,
        xaxis_title="Risk (Volatility)",
        yaxis_title="Expected Return",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_current_allocation():
    """Render current portfolio allocation"""

    # Sample allocation
    allocation = {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "GOOGL": 0.18,
        "AMZN": 0.15,
        "META": 0.12,
        "NVDA": 0.10,
    }

    # Create pie chart
    fig = go.Figure(
        data=[
            go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                hole=0.3,
                marker_colors=px.colors.qualitative.Set3,
            )
        ]
    )

    fig.update_layout(height=350, showlegend=True, margin=dict(t=0, b=0, l=0, r=0))

    st.plotly_chart(fig, use_container_width=True)

    # Allocation table
    st.markdown("**Position Details:**")

    positions_df = pd.DataFrame(
        {
            "Symbol": list(allocation.keys()),
            "Weight": list(allocation.values()),
            "Value": [w * 100000 for w in allocation.values()],
            "Change": np.random.uniform(-0.05, 0.05, len(allocation)),
        }
    )

    st.dataframe(
        positions_df.style.format({"Weight": "{:.1%}", "Value": "${:,.0f}", "Change": "{:+.2%}"}),
        use_container_width=True,
    )


def render_portfolio_metrics():
    """Render portfolio performance metrics"""

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Expected Return", "14.5%", "+2.3%")
        st.metric("Volatility", "18.2%", "-1.5%")

    with col2:
        st.metric("Sharpe Ratio", "1.24", "+0.15")
        st.metric("Sortino Ratio", "1.85", "+0.22")

    with col3:
        st.metric("Max Drawdown", "-8.5%", "")
        st.metric("Value at Risk", "-3.2%", "")

    with col4:
        st.metric("Beta", "0.95", "-0.05")
        st.metric("Alpha", "2.8%", "+0.5%")

    # Historical performance chart
    st.markdown("**Portfolio Performance:**")

    dates = pd.date_range(end=datetime.now(), periods=90, freq="D")
    portfolio_value = 100000 * (1 + np.random.randn(90).cumsum() * 0.01)
    benchmark_value = 100000 * (1 + np.random.randn(90).cumsum() * 0.008)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_value,
            mode="lines",
            name="Portfolio",
            line=dict(color="#1f77b4", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=benchmark_value,
            mode="lines",
            name="Benchmark (S&P 500)",
            line=dict(color="#ff7f0e", width=1, dash="dash"),
        )
    )

    fig.update_layout(
        height=300,
        xaxis_title="",
        yaxis_title="Portfolio Value ($)",
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def render_rebalancing_analysis():
    """Render rebalancing analysis panel"""

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Rebalancing Triggers:**")

        triggers = {
            "Weight Deviation": {"status": "Active", "threshold": "5%", "current": "7.2%"},
            "Time-Based": {"status": "Pending", "threshold": "30 days", "current": "22 days"},
            "Volatility": {"status": "Inactive", "threshold": "25%", "current": "18%"},
            "Drawdown": {"status": "Inactive", "threshold": "10%", "current": "3.5%"},
        }

        for trigger, details in triggers.items():
            status_color = (
                "#00cc88"
                if details["status"] == "Active"
                else "#ff9900"
                if details["status"] == "Pending"
                else "#999"
            )

            st.markdown(
                f"""
            <div style='padding: 10px; margin: 5px 0; background: #f5f5f5; border-radius: 5px;'>
                <b>{trigger}</b> <span style='color: {status_color}'>● {details["status"]}</span><br>
                <small>Threshold: {details["threshold"]} | Current: {details["current"]}</small>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("**Cost-Benefit Analysis:**")

        # Sample rebalancing analysis
        analysis = {
            "Proposed Trades": 8,
            "Total Turnover": "$25,000",
            "Estimated Cost": "$125",
            "Expected Benefit": "$450",
            "Cost-Benefit Ratio": 3.6,
            "Decision": "REBALANCE",
        }

        for key, value in analysis.items():
            if key == "Decision":
                color = "#00cc88" if value == "REBALANCE" else "#ff0000"
                st.markdown(
                    f"**{key}:** <span style='color: {color}'>{value}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"**{key}:** {value}")

        # Rebalancing history
        st.markdown("**Recent Rebalances:**")

        history_df = pd.DataFrame(
            {
                "Date": pd.date_range(end=datetime.now(), periods=5, freq="30D").strftime(
                    "%Y-%m-%d"
                ),
                "Trades": [6, 8, 5, 7, 9],
                "Cost": [95, 125, 78, 110, 145],
                "Turnover": [0.18, 0.25, 0.15, 0.22, 0.28],
            }
        )

        st.dataframe(
            history_df.style.format({"Cost": "${:.0f}", "Turnover": "{:.1%}"}),
            use_container_width=True,
        )


def render_ml_dashboard():
    """Main function to render ML dashboard pages"""

    # Page selection
    page = st.sidebar.selectbox("Select Page", ["ML Predictions", "Portfolio Optimization"])

    if page == "ML Predictions":
        render_ml_predictions_page()
    else:
        render_portfolio_optimization_page()
