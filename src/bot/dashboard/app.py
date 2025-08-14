"""
GPT-Trader Streamlit Dashboard
Simplified web-based dashboard for monitoring and controlling the trading system
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import json
import sqlite3
from typing import Dict, List, Any, Optional

# Configure Streamlit page
st.set_page_config(
    page_title="GPT-Trader Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .status-active { color: #00cc88; }
    .status-inactive { color: #ff4444; }
    .profit { color: #00cc88; }
    .loss { color: #ff4444; }
</style>
""", unsafe_allow_html=True)


class DashboardData:
    """Data provider for dashboard"""
    
    def __init__(self, db_path: str = "data/unified.db"):
        self.db_path = Path(db_path)
        
    def get_connection(self):
        """Get database connection"""
        if self.db_path.exists():
            return sqlite3.connect(str(self.db_path))
        return None
    
    def get_portfolio_overview(self) -> Dict[str, Any]:
        """Get portfolio overview metrics"""
        # Mock data for now - will connect to real database
        return {
            "total_value": 105234.56,
            "daily_pnl": 1234.56,
            "daily_pnl_pct": 1.18,
            "total_positions": 5,
            "cash_available": 45678.90,
            "margin_used": 59555.66,
            "active_strategies": 3,
            "total_trades_today": 12
        }
    
    def get_positions(self) -> pd.DataFrame:
        """Get current positions"""
        # Mock data for now
        data = {
            "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "Quantity": [100, 50, 30, 40, 75],
            "Entry Price": [150.00, 350.00, 140.00, 180.00, 250.00],
            "Current Price": [155.50, 365.00, 138.50, 185.00, 245.00],
            "P&L": [550.00, 750.00, -45.00, 200.00, -375.00],
            "P&L %": [3.67, 4.29, -1.07, 2.78, -2.00],
            "Strategy": ["trend_breakout", "demo_ma", "trend_breakout", "demo_ma", "trend_breakout"]
        }
        return pd.DataFrame(data)
    
    def get_strategy_performance(self) -> pd.DataFrame:
        """Get strategy performance metrics"""
        # Mock data for now
        data = {
            "Strategy": ["trend_breakout", "demo_ma", "mean_reversion"],
            "Total Return": [12.5, 8.3, -2.1],
            "Sharpe Ratio": [1.45, 1.12, -0.34],
            "Win Rate": [58.2, 52.1, 45.3],
            "Avg Win": [2.34, 1.89, 1.45],
            "Avg Loss": [-1.12, -1.45, -1.78],
            "Max Drawdown": [-5.4, -7.2, -12.3],
            "Status": ["Active", "Active", "Paused"]
        }
        return pd.DataFrame(data)
    
    def get_recent_trades(self) -> pd.DataFrame:
        """Get recent trade history"""
        # Mock data for now
        times = pd.date_range(end=datetime.now(), periods=10, freq='H')
        data = {
            "Time": times,
            "Symbol": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"] * 2,
            "Side": ["BUY", "SELL"] * 5,
            "Quantity": [100, 50, 30, 40, 75, 100, 50, 30, 40, 75],
            "Price": [155.50, 365.00, 138.50, 185.00, 245.00, 156.00, 364.50, 139.00, 184.50, 246.00],
            "Strategy": ["trend_breakout", "demo_ma"] * 5
        }
        return pd.DataFrame(data)
    
    def get_performance_history(self, days: int = 30) -> pd.DataFrame:
        """Get historical performance data"""
        # Mock data for now
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        cumulative_return = pd.Series(range(days)).apply(lambda x: 100000 * (1 + 0.001 * x + 0.002 * pd.Series(range(days)).sample(1).values[0]))
        
        data = {
            "Date": dates,
            "Portfolio Value": cumulative_return,
            "Daily Return": cumulative_return.pct_change() * 100
        }
        return pd.DataFrame(data)


def render_portfolio_overview(data_provider: DashboardData):
    """Render portfolio overview section"""
    st.header("📊 Portfolio Overview")
    
    metrics = data_provider.get_portfolio_overview()
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Portfolio Value",
            f"${metrics['total_value']:,.2f}",
            f"{metrics['daily_pnl_pct']:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Daily P&L",
            f"${metrics['daily_pnl']:+,.2f}",
            f"{metrics['daily_pnl_pct']:+.2f}%"
        )
    
    with col3:
        st.metric(
            "Active Positions",
            metrics['total_positions'],
            f"{metrics['total_trades_today']} trades today"
        )
    
    with col4:
        st.metric(
            "Cash Available",
            f"${metrics['cash_available']:,.2f}",
            f"Margin: ${metrics['margin_used']:,.2f}"
        )
    
    # Performance chart
    st.subheader("📈 Performance History")
    perf_data = data_provider.get_performance_history()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perf_data['Date'],
        y=perf_data['Portfolio Value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00cc88', width=2)
    ))
    
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_positions(data_provider: DashboardData):
    """Render current positions section"""
    st.header("📦 Current Positions")
    
    positions_df = data_provider.get_positions()
    
    # Format the dataframe for display
    formatted_df = positions_df.copy()
    formatted_df['P&L'] = formatted_df['P&L'].apply(lambda x: f"${x:+,.2f}")
    formatted_df['P&L %'] = formatted_df['P&L %'].apply(lambda x: f"{x:+.2f}%")
    formatted_df['Entry Price'] = formatted_df['Entry Price'].apply(lambda x: f"${x:.2f}")
    formatted_df['Current Price'] = formatted_df['Current Price'].apply(lambda x: f"${x:.2f}")
    
    # Color code P&L
    def highlight_pnl(val):
        if '+' in str(val):
            return 'color: #00cc88'
        elif '-' in str(val):
            return 'color: #ff4444'
        return ''
    
    styled_df = formatted_df.style.applymap(highlight_pnl, subset=['P&L', 'P&L %'])
    st.dataframe(styled_df, use_container_width=True, height=300)


def render_strategy_performance(data_provider: DashboardData):
    """Render strategy performance section"""
    st.header("🎯 Strategy Performance")
    
    strat_df = data_provider.get_strategy_performance()
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Performance Metrics", "Comparison Charts"])
    
    with tab1:
        # Format the dataframe
        formatted_df = strat_df.copy()
        formatted_df['Total Return'] = formatted_df['Total Return'].apply(lambda x: f"{x:+.2f}%")
        formatted_df['Sharpe Ratio'] = formatted_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
        formatted_df['Win Rate'] = formatted_df['Win Rate'].apply(lambda x: f"{x:.1f}%")
        formatted_df['Max Drawdown'] = formatted_df['Max Drawdown'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(formatted_df, use_container_width=True)
    
    with tab2:
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(strat_df, x='Strategy', y='Total Return', 
                        title='Total Returns by Strategy',
                        color='Total Return',
                        color_continuous_scale=['red', 'yellow', 'green'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(strat_df, x='Win Rate', y='Sharpe Ratio', 
                           size='Total Return', color='Strategy',
                           title='Risk-Adjusted Performance',
                           hover_data=['Max Drawdown'])
            st.plotly_chart(fig, use_container_width=True)


def render_recent_trades(data_provider: DashboardData):
    """Render recent trades section"""
    st.header("📝 Recent Trades")
    
    trades_df = data_provider.get_recent_trades()
    
    # Format the dataframe
    formatted_df = trades_df.copy()
    formatted_df['Time'] = formatted_df['Time'].dt.strftime('%Y-%m-%d %H:%M')
    formatted_df['Price'] = formatted_df['Price'].apply(lambda x: f"${x:.2f}")
    
    # Add side coloring
    def color_side(val):
        if val == 'BUY':
            return 'background-color: #e8f5e9'
        else:
            return 'background-color: #ffebee'
    
    styled_df = formatted_df.style.applymap(color_side, subset=['Side'])
    st.dataframe(styled_df, use_container_width=True, height=400)


def render_risk_metrics(data_provider: DashboardData):
    """Render risk metrics section"""
    st.header("⚠️ Risk Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("VaR (95%)", "$2,345", "-5.2%")
        st.metric("Portfolio Beta", "1.12", "+0.03")
    
    with col2:
        st.metric("Current Drawdown", "-3.4%", "Max: -7.8%")
        st.metric("Sharpe Ratio (30d)", "1.34", "+0.12")
    
    with col3:
        st.metric("Correlation", "0.65", "-0.05")
        st.metric("Volatility (30d)", "18.5%", "+2.1%")
    
    # Risk alerts
    st.subheader("🚨 Active Risk Alerts")
    alerts = [
        {"Level": "⚠️ Warning", "Message": "TSLA position exceeds 15% of portfolio", "Time": "10 min ago"},
        {"Level": "ℹ️ Info", "Message": "Market volatility increased by 20%", "Time": "1 hour ago"},
    ]
    
    for alert in alerts:
        st.info(f"{alert['Level']}: {alert['Message']} ({alert['Time']})")


def render_system_health(data_provider: DashboardData):
    """Render system health monitoring"""
    st.header("💚 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Data Feed", "Connected", "Latency: 12ms")
    
    with col2:
        st.metric("Execution", "Active", "Orders: 0 pending")
    
    with col3:
        st.metric("Risk Monitor", "Running", "Checks: 156 today")
    
    with col4:
        st.metric("Database", "Healthy", "Size: 45 MB")
    
    # Component status
    components = {
        "Strategy Engine": "🟢 Running",
        "Portfolio Manager": "🟢 Active",
        "Risk Monitor": "🟢 Active",
        "Data Pipeline": "🟢 Connected",
        "Order Executor": "🟡 Idle",
        "Alert System": "🟢 Active"
    }
    
    st.subheader("Component Status")
    cols = st.columns(3)
    for i, (component, status) in enumerate(components.items()):
        with cols[i % 3]:
            st.write(f"{component}: {status}")


def main():
    """Main dashboard application"""
    st.title("🚀 GPT-Trader Dashboard")
    
    # Initialize data provider
    data_provider = DashboardData()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["Portfolio Overview", "Positions", "Strategy Performance", 
         "Recent Trades", "Risk Metrics", "System Health"]
    )
    
    # Auto-refresh option
    st.sidebar.divider()
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
    if auto_refresh:
        st.rerun()
    
    # Render selected page
    if page == "Portfolio Overview":
        render_portfolio_overview(data_provider)
    elif page == "Positions":
        render_positions(data_provider)
    elif page == "Strategy Performance":
        render_strategy_performance(data_provider)
    elif page == "Recent Trades":
        render_recent_trades(data_provider)
    elif page == "Risk Metrics":
        render_risk_metrics(data_provider)
    elif page == "System Health":
        render_system_health(data_provider)
    
    # Footer
    st.sidebar.divider()
    st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption("GPT-Trader v2.0 - Autonomous Portfolio Management")


if __name__ == "__main__":
    main()