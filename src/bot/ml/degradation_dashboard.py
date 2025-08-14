"""
Degradation Visualization Dashboard
Phase 3, Week 1: MON-007
Interactive dashboard for monitoring model degradation metrics
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .degradation_integration import DegradationIntegrator

logger = logging.getLogger(__name__)


class DegradationDashboard:
    """
    Interactive dashboard for visualizing model degradation metrics.
    Provides real-time monitoring and historical analysis.
    """

    def __init__(self, integrator: DegradationIntegrator | None = None, update_interval: int = 60):
        """
        Initialize the dashboard.

        Args:
            integrator: DegradationIntegrator instance to monitor
            update_interval: Update interval in seconds
        """
        self.integrator = integrator
        self.update_interval = update_interval
        self.figures = {}
        self.last_update = None

        logger.info(f"Degradation Dashboard initialized with {update_interval}s update interval")

    def create_overview_dashboard(self) -> go.Figure:
        """
        Create main overview dashboard with key metrics.

        Returns:
            Plotly figure with overview dashboard
        """
        if not self.integrator or not self.integrator.reports:
            return self._create_empty_dashboard("No data available")

        # Create subplots
        fig = make_subplots(
            rows=3,
            cols=3,
            subplot_titles=(
                "Overall Status",
                "Accuracy Trend",
                "Degradation Score",
                "Feature Drift",
                "Confidence Trend",
                "Error Distribution",
                "CUSUM Chart",
                "Alerts Timeline",
                "Retraining History",
            ),
            specs=[
                [{"type": "indicator"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}, {"type": "histogram"}],
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "table"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # Get latest report
        latest = self.integrator.reports[-1]

        # 1. Overall Status Indicator
        status_color = self._get_status_color(latest.overall_status)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=latest.legacy_accuracy * 100,
                title={"text": "Model Accuracy (%)"},
                domain={"x": [0, 1], "y": [0, 1]},
                delta={
                    "reference": (
                        latest.legacy_accuracy * 100
                        if len(self.integrator.reports) > 1
                        else latest.legacy_accuracy * 100
                    )
                },
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": status_color},
                    "steps": [
                        {"range": [0, 55], "color": "lightgray"},
                        {"range": [55, 70], "color": "yellow"},
                        {"range": [70, 100], "color": "lightgreen"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 55,
                    },
                },
            ),
            row=1,
            col=1,
        )

        # 2. Accuracy Trend
        if len(self.integrator.reports) > 1:
            timestamps = [r.timestamp for r in self.integrator.reports]
            accuracies = [r.legacy_accuracy for r in self.integrator.reports]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=accuracies,
                    mode="lines+markers",
                    name="Accuracy",
                    line=dict(color="blue", width=2),
                    marker=dict(size=5),
                ),
                row=1,
                col=2,
            )

            # Add baseline
            if self.integrator.baseline_set:
                baseline = self.integrator.legacy_monitor.model_baselines.get(
                    latest.model_id, accuracies[0]
                )
                fig.add_hline(
                    y=baseline,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Baseline",
                    row=1,
                    col=2,
                )

        # 3. Degradation Score Trend
        if len(self.integrator.reports) > 1:
            scores = [r.advanced_score for r in self.integrator.reports]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=scores,
                    mode="lines+markers",
                    name="Degradation Score",
                    line=dict(color="red", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(255, 0, 0, 0.1)",
                ),
                row=1,
                col=3,
            )

            # Add threshold lines
            fig.add_hline(
                y=0.6,
                line_dash="dash",
                line_color="orange",
                annotation_text="Warning",
                row=1,
                col=3,
            )
            fig.add_hline(
                y=0.8, line_dash="dash", line_color="red", annotation_text="Critical", row=1, col=3
            )

        # 4. Feature Drift Bar Chart
        if self.integrator.advanced_detector and latest.feature_drift_count > 0:
            # Get feature drift details from advanced detector
            drift_scores = self.integrator.advanced_detector.feature_baselines
            if drift_scores:
                features = list(drift_scores.keys())[:10]  # Top 10 features
                drift_values = [1.0 - drift_scores.get(f, {}).get("mean", 1.0) for f in features]

                fig.add_trace(
                    go.Bar(x=features, y=drift_values, name="Feature Drift", marker_color="orange"),
                    row=2,
                    col=1,
                )

        # 5. Confidence Trend
        if len(self.integrator.reports) > 1:
            confidences = [r.confidence_level for r in self.integrator.reports]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=confidences,
                    mode="lines+markers",
                    name="Confidence",
                    line=dict(color="purple", width=2),
                    marker=dict(size=5),
                ),
                row=2,
                col=2,
            )

            fig.update_yaxes(range=[0, 1], row=2, col=2)

        # 6. Error Distribution
        if self.integrator.advanced_detector:
            # Get recent error patterns
            error_history = self.integrator.advanced_detector.error_patterns
            if error_history:
                errors = [e for pattern in error_history for e in pattern.get("errors", [])]

                fig.add_trace(
                    go.Histogram(
                        x=errors[-100:],  # Last 100 errors
                        nbinsx=20,
                        name="Error Distribution",
                        marker_color="red",
                    ),
                    row=2,
                    col=3,
                )

        # 7. CUSUM Chart
        if len(self.integrator.reports) > 1:
            cusum_values = [r.cusum_value for r in self.integrator.reports]

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cusum_values,
                    mode="lines",
                    name="CUSUM",
                    line=dict(color="brown", width=2),
                ),
                row=3,
                col=1,
            )

            # Add control limit
            if self.integrator.advanced_detector:
                h_value = self.integrator.advanced_detector.cusum_h
                fig.add_hline(
                    y=h_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Control Limit",
                    row=3,
                    col=1,
                )

        # 8. Alerts Timeline
        if latest.alerts:
            alert_times = [a.timestamp for a in latest.alerts[-20:]]
            alert_types = [a.alert_type.value for a in latest.alerts[-20:]]
            alert_severities = [a.severity for a in latest.alerts[-20:]]

            # Color based on severity
            colors = [
                "green" if s == "low" else "orange" if s == "medium" else "red"
                for s in alert_severities
            ]

            fig.add_trace(
                go.Scatter(
                    x=alert_times,
                    y=alert_types,
                    mode="markers",
                    name="Alerts",
                    marker=dict(size=10, color=colors, symbol="diamond"),
                    text=[f"{t}: {s}" for t, s in zip(alert_types, alert_severities, strict=False)],
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=3,
                col=2,
            )

        # 9. Retraining History Table
        retrain_data = self._get_retraining_history()
        if retrain_data:
            fig.add_trace(
                go.Table(
                    header=dict(
                        values=["Date", "Trigger", "Result"],
                        fill_color="paleturquoise",
                        align="left",
                    ),
                    cells=dict(
                        values=[
                            retrain_data["dates"],
                            retrain_data["triggers"],
                            retrain_data["results"],
                        ],
                        fill_color="lavender",
                        align="left",
                    ),
                ),
                row=3,
                col=3,
            )

        # Update layout
        fig.update_layout(
            title={
                "text": f"Model Degradation Dashboard - {latest.model_id}",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            showlegend=False,
            height=900,
            template="plotly_white",
        )

        # Update axes labels
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=1, col=3)
        fig.update_yaxes(title_text="Score", row=1, col=3)
        fig.update_xaxes(title_text="Feature", row=2, col=1)
        fig.update_yaxes(title_text="Drift", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Confidence", row=2, col=2)
        fig.update_xaxes(title_text="Error Value", row=2, col=3)
        fig.update_yaxes(title_text="Count", row=2, col=3)
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="CUSUM", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=2)

        self.figures["overview"] = fig
        self.last_update = datetime.now()

        return fig

    def create_feature_drift_dashboard(self) -> go.Figure:
        """
        Create detailed feature drift analysis dashboard.

        Returns:
            Plotly figure with feature drift analysis
        """
        if not self.integrator or not self.integrator.advanced_detector:
            return self._create_empty_dashboard("No feature drift data available")

        detector = self.integrator.advanced_detector

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Feature Drift Heatmap",
                "Top Drifted Features",
                "Drift Evolution",
                "Feature Correlations",
            ),
            specs=[
                [{"type": "heatmap"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
            ],
        )

        # Get feature drift data
        if detector.feature_baselines:
            features = list(detector.feature_baselines.keys())

            # 1. Feature Drift Heatmap
            if len(self.integrator.reports) > 1:
                # Create drift matrix over time
                drift_matrix = []
                for report in self.integrator.reports[-20:]:  # Last 20 reports
                    row = []
                    for feature in features[:20]:  # Top 20 features
                        # Simulate drift score (in production, get from report)
                        drift_score = np.random.random()
                        row.append(drift_score)
                    drift_matrix.append(row)

                fig.add_trace(
                    go.Heatmap(
                        z=drift_matrix,
                        x=features[:20],
                        y=[f"T-{i}" for i in range(len(drift_matrix))],
                        colorscale="RdYlGn_r",
                        reversescale=False,
                    ),
                    row=1,
                    col=1,
                )

            # 2. Top Drifted Features
            # Get current drift scores
            drift_scores = {}
            for feature in features:
                baseline = detector.feature_baselines.get(feature, {})
                if baseline:
                    # Calculate drift (simplified)
                    drift_scores[feature] = np.random.random()  # Replace with actual

            # Sort and get top 10
            top_features = sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)[:10]

            fig.add_trace(
                go.Bar(
                    x=[f[0] for f in top_features],
                    y=[f[1] for f in top_features],
                    marker_color="coral",
                    text=[f"{v:.3f}" for _, v in top_features],
                    textposition="auto",
                ),
                row=1,
                col=2,
            )

            # 3. Drift Evolution for Top Features
            if len(self.integrator.reports) > 1:
                timestamps = [r.timestamp for r in self.integrator.reports]

                for i, (feature, _) in enumerate(top_features[:5]):
                    # Simulate drift evolution (replace with actual data)
                    drift_evolution = np.random.random(len(timestamps))

                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=drift_evolution,
                            mode="lines",
                            name=feature,
                            line=dict(width=2),
                        ),
                        row=2,
                        col=1,
                    )

            # 4. Feature Correlations
            if len(features) > 1:
                # Create correlation matrix (simplified)
                n_features = min(15, len(features))
                corr_matrix = np.random.rand(n_features, n_features)
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                np.fill_diagonal(corr_matrix, 1)

                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix,
                        x=features[:n_features],
                        y=features[:n_features],
                        colorscale="RdBu",
                        zmid=0,
                        text=corr_matrix,
                        texttemplate="%{text:.2f}",
                        textfont={"size": 8},
                    ),
                    row=2,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title="Feature Drift Analysis Dashboard",
            height=700,
            showlegend=True,
            template="plotly_white",
        )

        self.figures["feature_drift"] = fig
        return fig

    def create_performance_dashboard(self) -> go.Figure:
        """
        Create detailed performance metrics dashboard.

        Returns:
            Plotly figure with performance analysis
        """
        if not self.integrator or not self.integrator.reports:
            return self._create_empty_dashboard("No performance data available")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Accuracy vs Baseline",
                "Performance Volatility",
                "Trend Analysis",
                "Error Rate",
                "Confidence Distribution",
                "Performance by Hour",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "violin"}, {"type": "bar"}],
            ],
        )

        reports = self.integrator.reports
        timestamps = [r.timestamp for r in reports]

        # 1. Accuracy vs Baseline
        accuracies = [r.legacy_accuracy for r in reports]
        baseline = accuracies[0] if accuracies else 0.7

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=accuracies,
                mode="lines+markers",
                name="Actual",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[baseline] * len(timestamps),
                mode="lines",
                name="Baseline",
                line=dict(color="green", dash="dash"),
            ),
            row=1,
            col=1,
        )

        # 2. Performance Volatility
        if len(accuracies) > 3:
            # Calculate rolling volatility
            window = min(5, len(accuracies))
            volatilities = pd.Series(accuracies).rolling(window).std().fillna(0).tolist()

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=volatilities,
                    mode="lines",
                    name="Volatility",
                    line=dict(color="orange", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(255, 165, 0, 0.1)",
                ),
                row=1,
                col=2,
            )

        # 3. Trend Analysis
        if len(accuracies) > 1:
            # Fit trend line
            x_numeric = np.arange(len(accuracies))
            z = np.polyfit(x_numeric, accuracies, 1)
            p = np.poly1d(z)
            trend_line = p(x_numeric)

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=accuracies,
                    mode="markers",
                    name="Actual",
                    marker=dict(color="blue", size=8),
                ),
                row=1,
                col=3,
            )

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=trend_line,
                    mode="lines",
                    name="Trend",
                    line=dict(color="red", width=2, dash="dash"),
                ),
                row=1,
                col=3,
            )

        # 4. Error Rate
        error_rates = [1 - r.legacy_accuracy for r in reports]

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=error_rates,
                mode="lines+markers",
                name="Error Rate",
                line=dict(color="red", width=2),
                marker=dict(size=5),
            ),
            row=2,
            col=1,
        )

        # 5. Confidence Distribution
        confidences = [r.confidence_level for r in reports]

        fig.add_trace(
            go.Violin(
                y=confidences,
                name="Confidence",
                box_visible=True,
                meanline_visible=True,
                fillcolor="lightseagreen",
                opacity=0.6,
            ),
            row=2,
            col=2,
        )

        # 6. Performance by Hour
        # Group by hour of day
        hour_performance = {}
        for report in reports:
            hour = report.timestamp.hour
            if hour not in hour_performance:
                hour_performance[hour] = []
            hour_performance[hour].append(report.legacy_accuracy)

        hours = sorted(hour_performance.keys())
        avg_performance = [np.mean(hour_performance[h]) for h in hours]

        fig.add_trace(
            go.Bar(
                x=hours,
                y=avg_performance,
                name="Avg Accuracy",
                marker_color="skyblue",
                text=[f"{v:.3f}" for v in avg_performance],
                textposition="auto",
            ),
            row=2,
            col=3,
        )

        # Update layout
        fig.update_layout(
            title="Performance Metrics Dashboard",
            height=700,
            showlegend=True,
            template="plotly_white",
        )

        # Update axes
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Volatility", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=1, col=3)
        fig.update_yaxes(title_text="Accuracy", row=1, col=3)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Error Rate", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=2)
        fig.update_xaxes(title_text="Hour", row=2, col=3)
        fig.update_yaxes(title_text="Avg Accuracy", row=2, col=3)

        self.figures["performance"] = fig
        return fig

    def create_alert_dashboard(self) -> go.Figure:
        """
        Create alert monitoring dashboard.

        Returns:
            Plotly figure with alert analysis
        """
        if not self.integrator or not self.integrator.reports:
            return self._create_empty_dashboard("No alert data available")

        # Collect all alerts
        all_alerts = []
        for report in self.integrator.reports:
            all_alerts.extend(report.alerts)

        if not all_alerts:
            return self._create_empty_dashboard("No alerts generated")

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Alert Frequency",
                "Alert Types Distribution",
                "Severity Timeline",
                "Alert Patterns",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "sunburst"}],
            ],
        )

        # 1. Alert Frequency
        alert_times = [a.timestamp for a in all_alerts]
        alert_counts = pd.Series([1] * len(alert_times), index=alert_times).resample("1H").sum()

        fig.add_trace(
            go.Scatter(
                x=alert_counts.index,
                y=alert_counts.values,
                mode="lines+markers",
                name="Alert Count",
                line=dict(color="red", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.1)",
            ),
            row=1,
            col=1,
        )

        # 2. Alert Types Distribution
        alert_types = [a.alert_type.value for a in all_alerts]
        type_counts = pd.Series(alert_types).value_counts()

        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                hole=0.3,
                marker=dict(colors=px.colors.sequential.RdBu),
            ),
            row=1,
            col=2,
        )

        # 3. Severity Timeline
        severities = [a.severity for a in all_alerts]
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        severity_values = [severity_map.get(s, 1) for s in severities]

        fig.add_trace(
            go.Scatter(
                x=alert_times,
                y=severity_values,
                mode="markers",
                name="Severity",
                marker=dict(
                    size=10,
                    color=severity_values,
                    colorscale="Reds",
                    showscale=True,
                    colorbar=dict(title="Severity"),
                ),
                text=severities,
                hovertemplate="%{text}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # 4. Alert Patterns (Sunburst)
        # Create hierarchical data
        pattern_data = []
        for alert in all_alerts[:100]:  # Last 100 alerts
            pattern_data.append(
                {
                    "severity": alert.severity,
                    "type": alert.alert_type.value,
                    "hour": alert.timestamp.hour,
                }
            )

        if pattern_data:
            df_patterns = pd.DataFrame(pattern_data)

            fig.add_trace(
                go.Sunburst(
                    labels=["All"]
                    + df_patterns["severity"].tolist()
                    + df_patterns["type"].tolist(),
                    parents=[""] + ["All"] * len(df_patterns) + df_patterns["severity"].tolist(),
                    values=[1] * (1 + len(df_patterns) * 2),
                    branchvalues="total",
                ),
                row=2,
                col=2,
            )

        # Update layout
        fig.update_layout(
            title="Alert Monitoring Dashboard",
            height=700,
            showlegend=False,
            template="plotly_white",
        )

        self.figures["alerts"] = fig
        return fig

    def _get_status_color(self, status: str) -> str:
        """Get color for status indicator"""
        colors = {"healthy": "green", "warning": "yellow", "degraded": "orange", "critical": "red"}
        return colors.get(status, "gray")

    def _get_retraining_history(self) -> dict[str, list]:
        """Get retraining history data"""
        if not self.integrator or not self.integrator.legacy_monitor:
            return {}

        # Get from legacy monitor's alert history
        alerts = self.integrator.legacy_monitor.alert_history
        retrain_alerts = [a for a in alerts if a.get("type") == "retraining_triggered"]

        if not retrain_alerts:
            return {}

        return {
            "dates": [
                (
                    a["timestamp"].strftime("%Y-%m-%d")
                    if isinstance(a["timestamp"], datetime)
                    else a["timestamp"]
                )
                for a in retrain_alerts[-5:]
            ],
            "triggers": [a.get("reason", "Unknown")[:30] for a in retrain_alerts[-5:]],
            "results": ["Success"] * len(retrain_alerts[-5:]),  # Placeholder
        }

    def _create_empty_dashboard(self, message: str) -> go.Figure:
        """Create empty dashboard with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="gray"),
        )
        fig.update_layout(title="Degradation Dashboard", height=400, template="plotly_white")
        return fig

    def save_dashboards(self, output_dir: str = "dashboards"):
        """
        Save all dashboards to HTML files.

        Args:
            output_dir: Directory to save dashboard files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, fig in self.figures.items():
            filepath = output_path / f"degradation_{name}_{timestamp}.html"
            fig.write_html(str(filepath))
            logger.info(f"Saved {name} dashboard to {filepath}")

    def get_dashboard_summary(self) -> dict[str, Any]:
        """
        Get summary of dashboard metrics.

        Returns:
            Dictionary with dashboard summary
        """
        if not self.integrator or not self.integrator.reports:
            return {"status": "no_data"}

        latest = self.integrator.reports[-1]

        summary = {
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "current_status": latest.overall_status,
            "current_accuracy": latest.legacy_accuracy,
            "degradation_score": latest.advanced_score,
            "feature_drift_count": latest.feature_drift_count,
            "confidence_level": latest.confidence_level,
            "total_reports": len(self.integrator.reports),
            "dashboards_available": list(self.figures.keys()),
        }

        # Add trend information
        if len(self.integrator.reports) > 1:
            recent_scores = [r.advanced_score for r in self.integrator.reports[-5:]]
            summary["score_trend"] = (
                "increasing" if recent_scores[-1] > recent_scores[0] else "decreasing"
            )
            summary["avg_recent_score"] = np.mean(recent_scores)

        return summary


def create_degradation_dashboard(integrator: DegradationIntegrator) -> DegradationDashboard:
    """
    Create and initialize a degradation dashboard.

    Args:
        integrator: DegradationIntegrator instance

    Returns:
        Configured DegradationDashboard instance
    """
    dashboard = DegradationDashboard(integrator)

    # Create all dashboards
    dashboard.create_overview_dashboard()
    dashboard.create_feature_drift_dashboard()
    dashboard.create_performance_dashboard()
    dashboard.create_alert_dashboard()

    return dashboard


if __name__ == "__main__":
    # Example usage
    from .degradation_integration import create_integrated_monitor

    # Create monitor and dashboard
    monitor = create_integrated_monitor()

    # Simulate some data
    np.random.seed(42)
    for i in range(20):
        n = 100
        features = pd.DataFrame({"f1": np.random.randn(n), "f2": np.random.randn(n)})

        # Simulate degradation over time
        accuracy = 0.75 - (i * 0.01) + np.random.normal(0, 0.02)
        predictions = np.random.choice([0, 1], n)
        actuals = predictions.copy()
        wrong_idx = np.random.choice(n, int((1 - accuracy) * n), replace=False)
        actuals[wrong_idx] = 1 - actuals[wrong_idx]
        confidences = np.random.uniform(0.5, 0.9, n)

        monitor.check_degradation(features, predictions, actuals, confidences)

    # Create dashboard
    dashboard = create_degradation_dashboard(monitor)

    # Save dashboards
    dashboard.save_dashboards()

    # Get summary
    summary = dashboard.get_dashboard_summary()
    print("\nDashboard Summary:")
    print(json.dumps(summary, indent=2, default=str))
