"""
Local alerting system for monitoring.

Complete isolation - no external dependencies.
"""

import uuid
from datetime import datetime, timedelta

from bot_v2.monitoring.alert_types import Alert, AlertSeverity
from bot_v2.monitoring.interfaces import MonitorConfig


class AlertManager:
    """Manages system alerts."""

    def __init__(self, config: MonitorConfig) -> None:
        """
        Initialize alert manager.

        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.max_history = 1000

        # Deduplication tracking
        self.recent_alerts: dict[str, datetime] = {}
        self.dedup_window_seconds = 300  # 5 minutes

    def create_alert(
        self, severity: AlertSeverity, component: str, message: str, details: dict | None = None
    ) -> Alert | None:
        """
        Create a new alert.

        Args:
            level: Alert severity level
            component: Component that triggered alert
            message: Alert message
            details: Additional details

        Returns:
            Alert object or None if deduplicated
        """
        # Check for duplicate alerts
        alert_key = f"{component}:{message}"
        if alert_key in self.recent_alerts:
            time_since = (datetime.now() - self.recent_alerts[alert_key]).total_seconds()
            if time_since < self.dedup_window_seconds:
                # Duplicate alert within window, skip
                return None

        # Create alert
        alert_id = str(uuid.uuid4())[:8]
        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=component,
            component=component,
            message=message,
            details=dict(details or {}),
            source="monitoring.alert_manager",
        )

        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Trim history
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Update deduplication tracking
        self.recent_alerts[alert_key] = datetime.now()

        # Send notification if enabled
        if self.config.enable_notifications:
            self._send_notification(alert)

        # Print alert
        self._print_alert(alert)

        return alert

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if acknowledged successfully
        """
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledge()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if resolved successfully
        """
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.mark_resolved()

            # Move to history only
            del self.alerts[alert_id]

            print(f"✅ Alert {alert_id} resolved")
            return True
        return False

    def get_all_alerts(self) -> list[Alert]:
        """Get all alerts (active and historical)."""
        return self.alert_history

    def get_active_alerts(self) -> list[Alert]:
        """Get active alerts only."""
        return list(self.alerts.values())

    def get_alerts_by_level(self, severity: AlertSeverity) -> list[Alert]:
        """
        Get alerts by severity level.

        Args:
            level: Alert level to filter by

        Returns:
            List of alerts with specified level
        """
        return [a for a in self.alerts.values() if a.severity == severity]

    def get_alerts_by_component(self, component: str) -> list[Alert]:
        """
        Get alerts by component.

        Args:
            component: Component to filter by

        Returns:
            List of alerts for specified component
        """
        return [a for a in self.alerts.values() if a.component == component]

    def clear_resolved_alerts(self) -> None:
        """Clear resolved alerts from history."""
        self.alert_history = [a for a in self.alert_history if a.is_active()]

    def get_alert_summary(self) -> dict:
        """Get summary of current alerts."""
        active_alerts = self.get_active_alerts()

        summary = {
            "total_active": len(active_alerts),
            "critical": len([a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]),
            "error": len([a for a in active_alerts if a.severity == AlertSeverity.ERROR]),
            "warning": len([a for a in active_alerts if a.severity == AlertSeverity.WARNING]),
            "info": len([a for a in active_alerts if a.severity == AlertSeverity.INFO]),
            "acknowledged": len([a for a in active_alerts if a.acknowledged]),
            "components_affected": list({a.component for a in active_alerts}),
        }

        return summary

    def _send_notification(self, alert: Alert) -> None:
        """
        Send alert notification.

        Args:
            alert: Alert to notify about
        """
        # In production, this would send to:
        # - Email
        # - Slack
        # - PagerDuty
        # - SMS
        # etc.

    def _print_alert(self, alert: Alert) -> None:
        """
        Print alert to console.

        Args:
            alert: Alert to print
        """
        # Color coding by level
        if alert.severity == AlertSeverity.CRITICAL:
            prefix = "🚨 CRITICAL"
        elif alert.severity == AlertSeverity.ERROR:
            prefix = "❌ ERROR"
        elif alert.severity == AlertSeverity.WARNING:
            prefix = "⚠️  WARNING"
        else:
            prefix = "ℹ️  INFO"

        print(f"\n{prefix} Alert [{alert.alert_id}]")
        print(f"Component: {alert.component}")
        print(f"Message: {alert.message}")
        if alert.details:
            print(f"Details: {alert.details}")
        print(f"Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    def cleanup_old_alerts(self) -> None:
        """Clean up old alerts based on retention policy."""
        if not self.config.retention_days:
            return

        cutoff = datetime.now() - timedelta(days=self.config.retention_days)

        # Remove old resolved alerts
        self.alert_history = [
            a for a in self.alert_history if a.is_active() or a.created_at > cutoff
        ]

        # Clean up deduplication tracking
        current_time = datetime.now()
        self.recent_alerts = {
            k: v
            for k, v in self.recent_alerts.items()
            if (current_time - v).total_seconds() < self.dedup_window_seconds
        }
