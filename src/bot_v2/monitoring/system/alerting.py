"""
Local alerting system for monitoring.

DEPRECATED: This module is deprecated in favor of bot_v2.monitoring.alerts_manager.AlertManager,
which provides the same functionality plus multi-channel dispatching and YAML configuration.

For new code, use:
    from bot_v2.monitoring.alerts_manager import AlertManager

This module is kept temporarily for backward compatibility with system/engine.py.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from datetime import datetime, timedelta

from bot_v2.monitoring.alerts import Alert, AlertLevel
from bot_v2.monitoring.interfaces import MonitorConfig

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages system alerts.

    DEPRECATED: Use bot_v2.monitoring.alerts_manager.AlertManager instead.
    """

    def __init__(self, config: MonitorConfig) -> None:
        warnings.warn(
            "bot_v2.monitoring.system.alerting.AlertManager is deprecated. "
            "Use bot_v2.monitoring.alerts_manager.AlertManager instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.config = config
        self.alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.max_history = 1000

        # Deduplication tracking
        self.recent_alerts: dict[str, datetime] = {}
        self.dedup_window_seconds = 300  # 5 minutes

    def create_alert(
        self, level: AlertLevel, component: str, message: str, details: dict | None = None
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

        # Create alert using canonical Alert structure
        alert_id = str(uuid.uuid4())[:8]
        alert = Alert(
            timestamp=datetime.now(),
            source=component,
            severity=level,
            title=f"{component} Alert",
            message=message,
            context=details or {},
            alert_id=alert_id,
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
        # Note: Canonical Alert doesn't have acknowledged field
        # This is a no-op for backward compatibility
        return alert_id in self.alerts

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if resolved successfully
        """
        if alert_id in self.alerts:
            # Remove from active alerts (history still kept)
            alert = self.alerts[alert_id]
            del self.alerts[alert_id]
            logger.info(
                "Alert resolved",
                extra={
                    "alert_id": alert_id,
                    "source": alert.source,
                    "severity": alert.severity.value,
                },
            )
            return True
        return False

    def get_all_alerts(self) -> list[Alert]:
        """Get all alerts (active and historical)."""
        return self.alert_history

    def get_active_alerts(self) -> list[Alert]:
        """Get active alerts only."""
        return list(self.alerts.values())

    def get_alerts_by_level(self, level: AlertLevel) -> list[Alert]:
        """
        Get alerts by severity level.

        Args:
            level: Alert level to filter by

        Returns:
            List of alerts with specified level
        """
        return [a for a in self.alerts.values() if a.severity == level]

    def get_alerts_by_component(self, component: str) -> list[Alert]:
        """
        Get alerts by component.

        Args:
            component: Component to filter by

        Returns:
            List of alerts for specified component
        """
        return [a for a in self.alerts.values() if a.source == component]

    def clear_resolved_alerts(self) -> None:
        """Clear resolved alerts from history."""
        # Note: canonical Alert doesn't have is_active(), filter by active alerts dict
        active_ids = set(self.alerts.keys())
        self.alert_history = [a for a in self.alert_history if a.alert_id in active_ids]

    def get_alert_summary(self) -> dict:
        """Get summary of current alerts."""
        active_alerts = self.get_active_alerts()

        summary = {
            "total_active": len(active_alerts),
            "critical": len([a for a in active_alerts if a.severity == AlertLevel.CRITICAL]),
            "error": len([a for a in active_alerts if a.severity == AlertLevel.ERROR]),
            "warning": len([a for a in active_alerts if a.severity == AlertLevel.WARNING]),
            "info": len([a for a in active_alerts if a.severity == AlertLevel.INFO]),
            "acknowledged": 0,  # Canonical Alert doesn't track acknowledged status
            "components_affected": list({a.source for a in active_alerts}),
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
        pass

    def _print_alert(self, alert: Alert) -> None:
        """
        Log alert using structured logging.

        Args:
            alert: Alert to log
        """
        # Map alert severity to logging level
        log_level_map = {
            AlertLevel.CRITICAL: logging.CRITICAL,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.INFO: logging.INFO,
        }
        log_level = log_level_map.get(alert.severity, logging.INFO)

        logger.log(
            log_level,
            "Alert created",
            extra={
                "alert_id": alert.alert_id,
                "source": alert.source,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "context": alert.context,
                "timestamp": alert.timestamp.isoformat(),
            },
        )

    def cleanup_old_alerts(self) -> None:
        """Clean up old alerts based on retention policy."""
        if not self.config.retention_days:
            return

        cutoff = datetime.now() - timedelta(days=self.config.retention_days)

        # Remove old resolved alerts (those not in active alerts dict)
        active_ids = set(self.alerts.keys())
        self.alert_history = [
            a for a in self.alert_history if a.alert_id in active_ids or a.timestamp > cutoff
        ]

        # Clean up deduplication tracking
        current_time = datetime.now()
        self.recent_alerts = {
            k: v
            for k, v in self.recent_alerts.items()
            if (current_time - v).total_seconds() < self.dedup_window_seconds
        }
