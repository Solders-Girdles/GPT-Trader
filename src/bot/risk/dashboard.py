"""Risk Management Dashboard.

This module provides a comprehensive risk management dashboard
for monitoring portfolio risk metrics and generating alerts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from bot.risk.config import RiskManagementConfig
from bot.risk.integration import RiskIntegration, AllocationResult
from bot.risk.utils import (
    calculate_portfolio_metrics,
    calculate_risk_adjusted_returns,
    format_risk_report,
)

logger = logging.getLogger(__name__)


class RiskAlert:
    """Risk alert representation."""

    def __init__(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metric_value: float = None,
        threshold: float = None,
        symbol: str = None,
        timestamp: datetime = None,
    ):
        self.alert_type = alert_type
        self.severity = severity  # LOW, MEDIUM, HIGH, CRITICAL
        self.message = message
        self.metric_value = metric_value
        self.threshold = threshold
        self.symbol = symbol
        self.timestamp = timestamp or datetime.now()
        self.alert_id = f"{self.alert_type}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type,
            "severity": self.severity,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        severity_icons = {"LOW": "ℹ️", "MEDIUM": "⚠️", "HIGH": "🔴", "CRITICAL": "🆘"}
        icon = severity_icons.get(self.severity, "⚠️")
        return f"{icon} {self.severity}: {self.message}"


class RiskDashboard:
    """Comprehensive risk management dashboard."""

    def __init__(self, risk_config: RiskManagementConfig, risk_integration: RiskIntegration):
        self.risk_config = risk_config
        self.risk_integration = risk_integration

        # Dashboard state
        self.alerts: List[RiskAlert] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.last_update = datetime.now()

        # Alert thresholds from config
        self.alert_thresholds = {
            "var_threshold": risk_config.monitoring.var_alert_threshold_pct,
            "drawdown_threshold": risk_config.monitoring.drawdown_alert_threshold_pct,
            "concentration_threshold": risk_config.monitoring.concentration_alert_threshold,
            "position_size_threshold": risk_config.position_limits.max_position_size_pct,
            "exposure_threshold": risk_config.portfolio_limits.max_gross_exposure_pct,
        }

        logger.info("Risk dashboard initialized")

    def update_dashboard(
        self,
        portfolio_value: float,
        positions: Dict[str, Dict[str, float]],
        market_data: Dict[str, pd.DataFrame] = None,
        current_pnl: float = 0.0,
    ) -> Dict[str, Any]:
        """Update dashboard with current portfolio state.

        Args:
            portfolio_value: Current portfolio value
            positions: Current position data
            market_data: Historical market data
            current_pnl: Current P&L

        Returns:
            Dashboard data dictionary
        """
        logger.info(f"Updating risk dashboard for portfolio value: ${portfolio_value:,.2f}")

        try:
            # Calculate current metrics
            portfolio_metrics = calculate_portfolio_metrics(positions, portfolio_value)

            # Check for new alerts
            new_alerts = self._check_risk_alerts(
                portfolio_metrics, positions, portfolio_value, current_pnl
            )
            self.alerts.extend(new_alerts)

            # Keep only recent alerts (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

            # Create dashboard data
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "current_pnl": current_pnl,
                "portfolio_metrics": portfolio_metrics,
                "position_count": len(
                    [p for p in positions.values() if p.get("position_value", 0) > 0]
                ),
                "alerts": {
                    "total": len(self.alerts),
                    "critical": len([a for a in self.alerts if a.severity == "CRITICAL"]),
                    "high": len([a for a in self.alerts if a.severity == "HIGH"]),
                    "medium": len([a for a in self.alerts if a.severity == "MEDIUM"]),
                    "recent": [alert.to_dict() for alert in self.alerts[-10:]],  # Last 10 alerts
                },
                "risk_limits": {
                    "max_position_size": self.risk_config.position_limits.max_position_size_pct,
                    "max_portfolio_exposure": self.risk_config.portfolio_limits.max_gross_exposure_pct,
                    "max_daily_loss": self.risk_config.portfolio_limits.max_daily_loss_pct,
                    "max_drawdown": self.risk_config.portfolio_limits.max_drawdown_pct,
                },
                "limit_utilization": self._calculate_limit_utilization(
                    portfolio_metrics, current_pnl, portfolio_value
                ),
            }

            # Add position details
            if positions:
                dashboard_data["positions"] = self._format_position_data(positions)

            # Add market data insights if available
            if market_data:
                dashboard_data["market_insights"] = self._calculate_market_insights(
                    market_data, positions
                )

            # Store metrics history
            self.metrics_history.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "portfolio_value": portfolio_value,
                    "metrics": portfolio_metrics,
                    "alert_count": len(self.alerts),
                }
            )

            # Keep last 1000 historical points
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            self.last_update = datetime.now()

            return dashboard_data

        except Exception as e:
            logger.error(f"Error updating risk dashboard: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
            }

    def _check_risk_alerts(
        self,
        portfolio_metrics: Dict[str, float],
        positions: Dict[str, Dict[str, float]],
        portfolio_value: float,
        current_pnl: float,
    ) -> List[RiskAlert]:
        """Check for risk limit violations and generate alerts."""
        alerts = []

        # Portfolio exposure alert
        exposure = portfolio_metrics.get("total_exposure", 0)
        if exposure > self.alert_thresholds["exposure_threshold"]:
            alerts.append(
                RiskAlert(
                    alert_type="PORTFOLIO_EXPOSURE",
                    severity="HIGH" if exposure > 0.98 else "MEDIUM",
                    message=f"Portfolio exposure ({exposure:.1%}) exceeds threshold ({self.alert_thresholds['exposure_threshold']:.1%})",
                    metric_value=exposure,
                    threshold=self.alert_thresholds["exposure_threshold"],
                )
            )

        # Position size alerts
        largest_position = portfolio_metrics.get("largest_position", 0)
        if largest_position > self.alert_thresholds["position_size_threshold"]:
            alerts.append(
                RiskAlert(
                    alert_type="POSITION_SIZE",
                    severity="HIGH" if largest_position > 0.15 else "MEDIUM",
                    message=f"Largest position ({largest_position:.1%}) exceeds limit ({self.alert_thresholds['position_size_threshold']:.1%})",
                    metric_value=largest_position,
                    threshold=self.alert_thresholds["position_size_threshold"],
                )
            )

        # Concentration alert
        concentration = portfolio_metrics.get("concentration_ratio", 0)
        if concentration > self.alert_thresholds["concentration_threshold"]:
            alerts.append(
                RiskAlert(
                    alert_type="CONCENTRATION",
                    severity="MEDIUM",
                    message=f"Portfolio concentration ({concentration:.3f}) is high (threshold: {self.alert_thresholds['concentration_threshold']:.3f})",
                    metric_value=concentration,
                    threshold=self.alert_thresholds["concentration_threshold"],
                )
            )

        # Daily loss alert
        daily_loss_pct = (
            abs(current_pnl) / portfolio_value if portfolio_value > 0 and current_pnl < 0 else 0
        )
        daily_loss_limit = self.risk_config.portfolio_limits.max_daily_loss_pct

        if daily_loss_pct > daily_loss_limit:
            severity = "CRITICAL" if daily_loss_pct > daily_loss_limit * 1.5 else "HIGH"
            alerts.append(
                RiskAlert(
                    alert_type="DAILY_LOSS",
                    severity=severity,
                    message=f"Daily loss ({daily_loss_pct:.1%}) exceeds limit ({daily_loss_limit:.1%})",
                    metric_value=daily_loss_pct,
                    threshold=daily_loss_limit,
                )
            )

        # Individual position alerts
        for symbol, pos_data in positions.items():
            pos_size_pct = pos_data.get("position_size_pct", 0)
            if pos_size_pct > self.alert_thresholds["position_size_threshold"]:
                alerts.append(
                    RiskAlert(
                        alert_type="INDIVIDUAL_POSITION",
                        severity="MEDIUM",
                        message=f"Position {symbol} size ({pos_size_pct:.1%}) exceeds limit",
                        metric_value=pos_size_pct,
                        threshold=self.alert_thresholds["position_size_threshold"],
                        symbol=symbol,
                    )
                )

        # Risk budget alert
        total_risk = portfolio_metrics.get("total_risk", 0)
        risk_limit = self.risk_config.portfolio_limits.max_portfolio_var_pct
        if total_risk > risk_limit:
            alerts.append(
                RiskAlert(
                    alert_type="RISK_BUDGET",
                    severity="HIGH",
                    message=f"Total portfolio risk ({total_risk:.1%}) exceeds VaR limit ({risk_limit:.1%})",
                    metric_value=total_risk,
                    threshold=risk_limit,
                )
            )

        return alerts

    def _calculate_limit_utilization(
        self, portfolio_metrics: Dict[str, float], current_pnl: float, portfolio_value: float
    ) -> Dict[str, float]:
        """Calculate how much of each risk limit is being utilized."""
        utilization = {}

        # Exposure utilization
        exposure = portfolio_metrics.get("total_exposure", 0)
        max_exposure = self.risk_config.portfolio_limits.max_gross_exposure_pct
        utilization["exposure"] = exposure / max_exposure if max_exposure > 0 else 0

        # Position size utilization
        largest_position = portfolio_metrics.get("largest_position", 0)
        max_position = self.risk_config.position_limits.max_position_size_pct
        utilization["position_size"] = largest_position / max_position if max_position > 0 else 0

        # Daily loss utilization
        daily_loss_pct = (
            abs(current_pnl) / portfolio_value if portfolio_value > 0 and current_pnl < 0 else 0
        )
        max_daily_loss = self.risk_config.portfolio_limits.max_daily_loss_pct
        utilization["daily_loss"] = daily_loss_pct / max_daily_loss if max_daily_loss > 0 else 0

        # Concentration utilization
        concentration = portfolio_metrics.get("concentration_ratio", 0)
        max_concentration = self.risk_config.portfolio_limits.max_concentration_ratio
        utilization["concentration"] = (
            concentration / max_concentration if max_concentration > 0 else 0
        )

        return utilization

    def _format_position_data(self, positions: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Format position data for dashboard display."""
        formatted_positions = []

        for symbol, pos_data in positions.items():
            if pos_data.get("position_value", 0) > 0:
                formatted_positions.append(
                    {
                        "symbol": symbol,
                        "value": pos_data.get("position_value", 0),
                        "size_pct": pos_data.get("position_size_pct", 0),
                        "pnl": pos_data.get("unrealized_pnl", 0),
                        "pnl_pct": pos_data.get("unrealized_pnl_pct", 0),
                        "risk": pos_data.get("total_risk", 0),
                        "stop_distance": pos_data.get("stop_distance", 0),
                    }
                )

        # Sort by position size (largest first)
        formatted_positions.sort(key=lambda x: x["size_pct"], reverse=True)

        return formatted_positions

    def _calculate_market_insights(
        self, market_data: Dict[str, pd.DataFrame], positions: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Calculate market insights from historical data."""
        insights = {"volatility_analysis": {}, "correlation_analysis": {}, "trend_analysis": {}}

        try:
            # Calculate volatilities for held positions
            for symbol in positions.keys():
                if symbol in market_data:
                    df = market_data[symbol]
                    if "Close" in df.columns and len(df) > 30:
                        returns = df["Close"].pct_change().dropna()
                        if len(returns) > 0:
                            volatility = returns.std() * (252**0.5)  # Annualized
                            insights["volatility_analysis"][symbol] = {
                                "volatility": float(volatility),
                                "percentile": None,  # Could add market percentile
                            }

            # Calculate portfolio correlation if multiple positions
            if len(positions) > 1:
                returns_data = {}
                for symbol in positions.keys():
                    if symbol in market_data:
                        df = market_data[symbol]
                        if "Close" in df.columns and len(df) > 30:
                            returns = df["Close"].pct_change().dropna()
                            if len(returns) >= 30:
                                returns_data[symbol] = returns.tail(30)

                if len(returns_data) > 1:
                    from bot.risk.utils import calculate_correlation_matrix

                    corr_matrix = calculate_correlation_matrix(returns_data)
                    if not corr_matrix.empty:
                        # Find highest correlation
                        max_corr = 0
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i + 1, len(corr_matrix.columns)):
                                corr = abs(corr_matrix.iloc[i, j])
                                if not pd.isna(corr):
                                    max_corr = max(max_corr, corr)

                        insights["correlation_analysis"] = {
                            "max_correlation": float(max_corr),
                            "avg_correlation": float(corr_matrix.mean().mean()),
                        }

        except Exception as e:
            logger.warning(f"Error calculating market insights: {e}")

        return insights

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        recent_alerts = [
            alert for alert in self.alerts if alert.timestamp > datetime.now() - timedelta(hours=1)
        ]

        return {
            "last_update": self.last_update.isoformat(),
            "total_alerts": len(self.alerts),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in self.alerts if a.severity == "CRITICAL"]),
            "risk_config_profile": self.risk_config.profile.value,
            "monitoring_enabled": self.risk_config.monitoring.real_time_monitoring,
            "circuit_breakers_enabled": self.risk_config.enable_circuit_breakers,
        }

    def generate_risk_report(
        self, positions: Dict[str, Dict[str, float]], portfolio_value: float
    ) -> str:
        """Generate formatted risk report."""
        portfolio_metrics = calculate_portfolio_metrics(positions, portfolio_value)

        # Get risk limits as dict
        risk_limits = {
            "max_position_size": self.risk_config.position_limits.max_position_size_pct,
            "max_portfolio_exposure": self.risk_config.portfolio_limits.max_gross_exposure_pct,
            "max_daily_loss": self.risk_config.portfolio_limits.max_daily_loss_pct,
            "max_drawdown": self.risk_config.portfolio_limits.max_drawdown_pct,
            "stop_loss_pct": self.risk_config.stop_loss.default_stop_loss_pct,
        }

        report = format_risk_report(portfolio_metrics, positions, risk_limits)

        # Add alerts section
        if self.alerts:
            report += "\n\nRECENT ALERTS:\n"
            report += "-" * 30 + "\n"
            for alert in self.alerts[-5:]:  # Last 5 alerts
                report += f"{alert}\n"

        return report

    def export_dashboard_data(self, filename: str = None) -> str:
        """Export dashboard data to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_dashboard_{timestamp}.json"

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "risk_config": self.risk_config.to_dict(),
            "alerts": [alert.to_dict() for alert in self.alerts],
            "metrics_history": self.metrics_history[-100:],  # Last 100 points
            "alert_thresholds": self.alert_thresholds,
        }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Dashboard data exported to {filename}")
        return filename

    def clear_alerts(self, alert_type: str = None, older_than_hours: int = None) -> int:
        """Clear alerts based on criteria.

        Args:
            alert_type: Clear alerts of specific type (optional)
            older_than_hours: Clear alerts older than specified hours (optional)

        Returns:
            Number of alerts cleared
        """
        initial_count = len(self.alerts)

        if older_than_hours:
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            self.alerts = [alert for alert in self.alerts if alert.timestamp > cutoff_time]

        if alert_type:
            self.alerts = [alert for alert in self.alerts if alert.alert_type != alert_type]

        cleared_count = initial_count - len(self.alerts)
        logger.info(f"Cleared {cleared_count} alerts")

        return cleared_count
