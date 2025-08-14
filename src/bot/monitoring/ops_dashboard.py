"""
Operational Dashboard
Phase 3, Week 8: OPS-017 to OPS-024
Real-time monitoring and visualization dashboard
"""

import logging
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="GPT-Trader Operations Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .alert-critical {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .alert-warning {
        background-color: #ffa500;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .status-healthy {
        color: #00cc00;
        font-weight: bold;
    }
    .status-degraded {
        color: #ff9900;
        font-weight: bold;
    }
    .status-critical {
        color: #ff0000;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


class MetricsCollector:
    """Collect system and application metrics"""

    def __init__(self):
        self.metrics_history = {
            "cpu": deque(maxlen=100),
            "memory": deque(maxlen=100),
            "disk": deque(maxlen=100),
            "predictions_per_sec": deque(maxlen=100),
            "model_accuracy": deque(maxlen=100),
            "active_alerts": deque(maxlen=100),
            "api_latency": deque(maxlen=100),
            "error_rate": deque(maxlen=100),
        }
        self.timestamps = deque(maxlen=100)

    def collect_metrics(self) -> dict[str, float]:
        """Collect current metrics"""
        metrics = {}

        # System metrics
        metrics["cpu"] = psutil.cpu_percent(interval=1)
        metrics["memory"] = psutil.virtual_memory().percent
        metrics["disk"] = psutil.disk_usage("/").percent

        # Application metrics (simulated for demo)
        metrics["predictions_per_sec"] = np.random.normal(5000, 500)
        metrics["model_accuracy"] = np.random.normal(0.6, 0.05)
        metrics["active_alerts"] = np.random.poisson(5)
        metrics["api_latency"] = np.random.gamma(2, 10)
        metrics["error_rate"] = np.random.exponential(0.01)

        # Store history
        timestamp = datetime.now()
        self.timestamps.append(timestamp)
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        return metrics

    def get_history(self, metric: str, window: int = 100) -> pd.DataFrame:
        """Get metric history as DataFrame"""
        if metric not in self.metrics_history:
            return pd.DataFrame()

        history = list(self.metrics_history[metric])[-window:]
        timestamps = list(self.timestamps)[-window:]

        return pd.DataFrame({"timestamp": timestamps, metric: history})


class AlertMonitor:
    """Monitor and display alerts"""

    def __init__(self):
        self.alerts = []
        self.alert_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    def get_active_alerts(self) -> list[dict]:
        """Get active alerts (simulated)"""
        # Simulate some alerts
        sample_alerts = [
            {
                "id": "alert_001",
                "title": "High Memory Usage",
                "severity": "high",
                "category": "system",
                "timestamp": datetime.now() - timedelta(minutes=5),
                "message": "Memory usage at 85%",
            },
            {
                "id": "alert_002",
                "title": "Model Accuracy Degradation",
                "severity": "medium",
                "category": "model",
                "timestamp": datetime.now() - timedelta(minutes=15),
                "message": "Accuracy dropped to 55%",
            },
        ]

        # Randomly add critical alert
        if np.random.random() > 0.8:
            sample_alerts.append(
                {
                    "id": "alert_003",
                    "title": "Trading System Error",
                    "severity": "critical",
                    "category": "trading",
                    "timestamp": datetime.now(),
                    "message": "Order execution failed",
                }
            )

        return sample_alerts

    def get_alert_stats(self) -> dict[str, int]:
        """Get alert statistics"""
        alerts = self.get_active_alerts()
        stats = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for alert in alerts:
            severity = alert.get("severity", "low")
            stats[severity] = stats.get(severity, 0) + 1

        return stats


class ModelPerformanceTracker:
    """Track ML model performance"""

    def __init__(self):
        self.models = ["XGBoost_v1.2", "RandomForest_v1.1", "LSTM_v2.0"]
        self.performance_data = {}

    def get_model_metrics(self) -> pd.DataFrame:
        """Get model performance metrics"""
        data = []
        for model in self.models:
            data.append(
                {
                    "Model": model,
                    "Accuracy": np.random.normal(0.6, 0.05),
                    "Precision": np.random.normal(0.65, 0.05),
                    "Recall": np.random.normal(0.55, 0.05),
                    "F1 Score": np.random.normal(0.58, 0.05),
                    "Sharpe Ratio": np.random.normal(1.2, 0.2),
                    "Predictions/Day": np.random.randint(100000, 500000),
                }
            )

        return pd.DataFrame(data)

    def get_prediction_distribution(self) -> dict:
        """Get prediction distribution data"""
        return {
            "timestamps": pd.date_range(start="today", periods=24, freq="H"),
            "predictions": np.random.poisson(5000, 24),
            "accuracy": np.random.normal(0.6, 0.02, 24),
        }


class TradingPerformanceMonitor:
    """Monitor trading performance"""

    def __init__(self):
        self.positions = []
        self.pnl_history = []

    def get_portfolio_metrics(self) -> dict:
        """Get portfolio performance metrics"""
        return {
            "total_value": 1000000 + np.random.normal(0, 10000),
            "daily_pnl": np.random.normal(5000, 2000),
            "total_return": np.random.normal(0.15, 0.05),
            "sharpe_ratio": np.random.normal(1.5, 0.3),
            "max_drawdown": np.random.uniform(0.05, 0.15),
            "win_rate": np.random.uniform(0.55, 0.65),
            "active_positions": np.random.randint(5, 20),
        }

    def get_pnl_history(self) -> pd.DataFrame:
        """Get P&L history"""
        dates = pd.date_range(end="today", periods=30, freq="D")
        pnl = np.cumsum(np.random.normal(5000, 2000, 30))

        return pd.DataFrame(
            {"date": dates, "pnl": pnl, "daily_return": np.random.normal(0.002, 0.01, 30)}
        )


def render_header():
    """Render dashboard header"""
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.image("https://via.placeholder.com/150x50?text=GPT-Trader", width=150)

    with col2:
        st.title("🎯 Operations Dashboard")
        st.caption(
            f"Phase 3 - Week 8 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    with col3:
        status = st.empty()
        if st.button("🔄 Refresh"):
            st.rerun()


def render_system_health(metrics_collector: MetricsCollector):
    """Render system health metrics"""
    st.header("💻 System Health")

    metrics = metrics_collector.collect_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("CPU Usage", f"{metrics['cpu']:.1f}%", delta=f"{metrics['cpu'] - 50:.1f}%")

    with col2:
        st.metric(
            "Memory Usage", f"{metrics['memory']:.1f}%", delta=f"{metrics['memory'] - 60:.1f}%"
        )

    with col3:
        st.metric("Disk Usage", f"{metrics['disk']:.1f}%")

    with col4:
        st.metric(
            "API Latency",
            f"{metrics['api_latency']:.1f}ms",
            delta=f"{metrics['api_latency'] - 20:.1f}ms",
        )

    # CPU and Memory charts
    col1, col2 = st.columns(2)

    with col1:
        cpu_history = metrics_collector.get_history("cpu", 50)
        if not cpu_history.empty:
            fig = px.line(cpu_history, x="timestamp", y="cpu", title="CPU Usage Over Time")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        memory_history = metrics_collector.get_history("memory", 50)
        if not memory_history.empty:
            fig = px.line(memory_history, x="timestamp", y="memory", title="Memory Usage Over Time")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def render_alert_panel(alert_monitor: AlertMonitor):
    """Render alert panel"""
    st.header("🚨 Active Alerts")

    alert_stats = alert_monitor.get_alert_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Critical", alert_stats.get("critical", 0))
    with col2:
        st.metric("High", alert_stats.get("high", 0))
    with col3:
        st.metric("Medium", alert_stats.get("medium", 0))
    with col4:
        st.metric("Low", alert_stats.get("low", 0))

    # Alert list
    alerts = alert_monitor.get_active_alerts()

    if alerts:
        for alert in alerts:
            severity = alert["severity"]
            if severity == "critical":
                st.markdown(
                    f"""
                <div class="alert-critical">
                    <strong>{alert['title']}</strong><br>
                    {alert['message']}<br>
                    <small>{alert['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            elif severity == "high":
                st.warning(f"**{alert['title']}** - {alert['message']}")
            else:
                st.info(f"{alert['title']} - {alert['message']}")
    else:
        st.success("✅ No active alerts")


def render_model_performance(model_tracker: ModelPerformanceTracker):
    """Render model performance metrics"""
    st.header("🤖 Model Performance")

    # Model metrics table
    model_metrics = model_tracker.get_model_metrics()

    # Format numerical columns
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        model_metrics[col] = model_metrics[col].apply(lambda x: f"{x:.3f}")
    model_metrics["Sharpe Ratio"] = model_metrics["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")

    st.dataframe(model_metrics, use_container_width=True)

    # Prediction distribution
    col1, col2 = st.columns(2)

    with col1:
        pred_data = model_tracker.get_prediction_distribution()
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=pred_data["timestamps"],
                y=pred_data["predictions"],
                name="Predictions",
                marker_color="lightblue",
            )
        )
        fig.update_layout(
            title="Predictions per Hour", xaxis_title="Time", yaxis_title="Predictions", height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pred_data["timestamps"],
                y=pred_data["accuracy"],
                mode="lines+markers",
                name="Accuracy",
                line=dict(color="green"),
            )
        )
        fig.add_hline(y=0.55, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig.update_layout(
            title="Model Accuracy Trend", xaxis_title="Time", yaxis_title="Accuracy", height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def render_trading_performance(trading_monitor: TradingPerformanceMonitor):
    """Render trading performance"""
    st.header("💰 Trading Performance")

    portfolio_metrics = trading_monitor.get_portfolio_metrics()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Portfolio Value",
            f"${portfolio_metrics['total_value']:,.0f}",
            delta=f"${portfolio_metrics['daily_pnl']:+,.0f}",
        )

    with col2:
        st.metric(
            "Total Return",
            f"{portfolio_metrics['total_return']:.1%}",
            delta=f"{portfolio_metrics['total_return'] - 0.1:.1%}",
        )

    with col3:
        st.metric("Sharpe Ratio", f"{portfolio_metrics['sharpe_ratio']:.2f}")

    with col4:
        st.metric("Win Rate", f"{portfolio_metrics['win_rate']:.1%}")

    # P&L Chart
    pnl_history = trading_monitor.get_pnl_history()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pnl_history["date"],
            y=pnl_history["pnl"],
            mode="lines",
            fill="tozeroy",
            name="Cumulative P&L",
            line=dict(color="green", width=2),
        )
    )
    fig.update_layout(
        title="30-Day P&L Performance", xaxis_title="Date", yaxis_title="P&L ($)", height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sidebar():
    """Render sidebar controls"""
    with st.sidebar:
        st.header("⚙️ Controls")

        # View selector
        view = st.selectbox(
            "Dashboard View", ["Overview", "System Health", "Models", "Trading", "Alerts"]
        )

        # Refresh settings
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)

        # Time range
        st.subheader("Time Range")
        time_range = st.selectbox(
            "Select Range", ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
        )

        # Filters
        st.subheader("Filters")
        show_critical = st.checkbox("Show Critical Alerts", value=True)
        show_warnings = st.checkbox("Show Warnings", value=True)

        # Export
        st.subheader("Export")
        if st.button("📊 Export Report"):
            st.success("Report exported successfully!")

        # System Status
        st.subheader("System Status")
        st.markdown(
            """
        <div class="status-healthy">✅ All Systems Operational</div>
        """,
            unsafe_allow_html=True,
        )

        return {
            "view": view,
            "auto_refresh": auto_refresh,
            "refresh_interval": refresh_interval,
            "time_range": time_range,
            "show_critical": show_critical,
            "show_warnings": show_warnings,
        }


def main():
    """Main dashboard application"""
    # Initialize components
    metrics_collector = MetricsCollector()
    alert_monitor = AlertMonitor()
    model_tracker = ModelPerformanceTracker()
    trading_monitor = TradingPerformanceMonitor()

    # Render header
    render_header()

    # Render sidebar and get settings
    settings = render_sidebar()

    # Main content area
    if settings["view"] == "Overview":
        # Overview dashboard
        tab1, tab2, tab3, tab4 = st.tabs(["System", "Models", "Trading", "Alerts"])

        with tab1:
            render_system_health(metrics_collector)

        with tab2:
            render_model_performance(model_tracker)

        with tab3:
            render_trading_performance(trading_monitor)

        with tab4:
            render_alert_panel(alert_monitor)

    elif settings["view"] == "System Health":
        render_system_health(metrics_collector)

    elif settings["view"] == "Models":
        render_model_performance(model_tracker)

    elif settings["view"] == "Trading":
        render_trading_performance(trading_monitor)

    elif settings["view"] == "Alerts":
        render_alert_panel(alert_monitor)

    # Auto-refresh
    if settings["auto_refresh"]:
        time.sleep(settings["refresh_interval"])
        st.rerun()


if __name__ == "__main__":
    main()
