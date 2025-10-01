from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

try:
    import yaml  # type: ignore
except Exception:
    yaml = cast(Any, None)

from bot_v2.monitoring.alerts import Alert, AlertDispatcher, AlertLevel, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class AlertManager:
    """Unified alert manager with dispatching, history, and deduplication.

    Combines the functionality of:
    - AlertDispatcher for multi-channel routing (Slack, PagerDuty, Email, etc.)
    - Alert history and deduplication
    - YAML profile configuration support

    This replaces bot_v2.monitoring.system.alerting.AlertManager.
    """

    dispatcher: AlertDispatcher
    alert_history: list[Alert] = field(default_factory=list)
    max_history: int = 1000
    dedup_window_seconds: int = 300  # 5 minutes

    # Deduplication tracking
    _recent_alerts: dict[str, datetime] = field(default_factory=dict, init=False)

    @classmethod
    def from_profile_yaml(cls, path: str | Path = "config/profiles/canary.yaml") -> AlertManager:
        config: dict[str, Any] = {}
        if yaml is not None:
            try:
                path_obj = Path(path)
                with path_obj.open() as f:
                    loaded = yaml.safe_load(f) or {}
                data = loaded if isinstance(loaded, dict) else {}
                monitoring_section = data.get("monitoring")
                alerts_config = (
                    monitoring_section.get("alerts") if isinstance(monitoring_section, dict) else {}
                )
                # Map profile fields to dispatcher config format
                mapped: dict[str, Any] = {
                    "slack_webhook_url": None,
                    "slack_min_severity": "WARNING",
                    "pagerduty_api_key": None,
                    "pagerduty_min_severity": "ERROR",
                }
                if isinstance(alerts_config, dict):
                    chans = alerts_config.get("channels") or []
                    if not isinstance(chans, list):
                        chans = [chans]
                    for c in chans:
                        if not isinstance(c, Mapping):
                            continue
                        ctype = str(c.get("type") or "").lower()
                        level = str(c.get("level") or "WARNING").upper()
                        if ctype == "slack" and not mapped["slack_webhook_url"]:
                            mapped["slack_webhook_url"] = c.get("webhook_url")
                            mapped["slack_min_severity"] = level
                        elif ctype == "pagerduty" and not mapped["pagerduty_api_key"]:
                            mapped["pagerduty_api_key"] = c.get("api_key")
                            mapped["pagerduty_min_severity"] = level
                config = mapped
            except Exception as e:
                logger.debug(f"Failed to load alert config from YAML: {e}")
        return cls(dispatcher=AlertDispatcher.from_config(config))

    def create_alert(
        self,
        level: AlertLevel | str,
        source: str,
        message: str,
        title: str | None = None,
        context: dict[str, Any] | None = None,
        component: str | None = None,  # Backward compatibility
        details: dict[str, Any] | None = None,  # Backward compatibility
    ) -> Alert | None:
        """Create an alert with deduplication.

        Args:
            level: Alert severity level (AlertLevel enum or string)
            source: Alert source/component name
            message: Alert message
            title: Optional alert title (defaults to "{source} Alert")
            context: Additional context dictionary
            component: (Deprecated) Alias for source
            details: (Deprecated) Alias for context

        Returns:
            Alert object or None if deduplicated
        """
        # Backward compatibility: component -> source, details -> context
        if component is not None:
            source = component
        if details is not None:
            context = details

        # Convert string level to AlertLevel
        if isinstance(level, str):
            level = AlertLevel[level.upper()]

        # Check for duplicate alerts
        alert_key = f"{source}:{message}"
        if alert_key in self._recent_alerts:
            time_since = (datetime.now() - self._recent_alerts[alert_key]).total_seconds()
            if time_since < self.dedup_window_seconds:
                # Duplicate alert within window, skip
                return None

        # Create alert
        alert_id = str(uuid.uuid4())[:8]
        alert = Alert(
            timestamp=datetime.now(),
            source=source,
            severity=level,
            title=title or f"{source} Alert",
            message=message,
            context=context or {},
            alert_id=alert_id,
        )

        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        # Update deduplication tracking
        self._recent_alerts[alert_key] = datetime.now()

        return alert

    async def send_alert(
        self,
        level: str,
        title: str,
        message: str,
        metrics: dict[str, Any] | None = None,
        source: str = "perps_bot",
    ) -> None:
        """Send an alert through the dispatcher (async).

        This is the legacy async method for backward compatibility.
        For sync alert creation with deduplication, use create_alert().

        Args:
            level: Alert level as string
            title: Alert title
            message: Alert message
            metrics: Optional metrics/context dict
            source: Source component name
        """
        sev = AlertSeverity[level.upper()] if level else AlertSeverity.WARNING
        alert = Alert(
            timestamp=datetime.utcnow(),
            source=source,
            severity=sev,
            title=title,
            message=message,
            context=dict(metrics or {}),
        )
        await self.dispatcher.dispatch(alert)
        # Also add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

    def get_recent_alerts(
        self, count: int = 10, severity: AlertLevel | None = None, source: str | None = None
    ) -> list[Alert]:
        """Get recent alerts with optional filtering.

        Args:
            count: Number of recent alerts to return
            severity: Optional severity filter
            source: Optional source filter

        Returns:
            List of recent alerts
        """
        alerts = list(self.alert_history)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]

        return alerts[-count:]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of recent alerts.

        Returns:
            Dictionary with alert counts by severity and sources
        """
        recent = self.alert_history[-100:]  # Last 100 alerts

        summary: dict[str, Any] = {
            "total": len(recent),
            "by_severity": {
                "critical": len([a for a in recent if a.severity == AlertLevel.CRITICAL]),
                "error": len([a for a in recent if a.severity == AlertLevel.ERROR]),
                "warning": len([a for a in recent if a.severity == AlertLevel.WARNING]),
                "info": len([a for a in recent if a.severity == AlertLevel.INFO]),
            },
            "sources": list({a.source for a in recent}),
        }

        return summary

    def cleanup_old_alerts(self, retention_hours: int = 24) -> None:
        """Clean up old alerts from history.

        Args:
            retention_hours: Keep alerts newer than this many hours
        """
        cutoff = datetime.now() - timedelta(hours=retention_hours)
        self.alert_history = [a for a in self.alert_history if a.timestamp > cutoff]

        # Clean up deduplication tracking
        current_time = datetime.now()
        self._recent_alerts = {
            k: v
            for k, v in self._recent_alerts.items()
            if (current_time - v).total_seconds() < self.dedup_window_seconds
        }

    # Backward compatibility methods for system/alerting.py API
    def get_all_alerts(self) -> list[Alert]:
        """Get all alerts (alias for alert_history).

        For backward compatibility with monitoring.system.alerting.AlertManager.
        """
        return self.alert_history

    def get_active_alerts(self) -> list[Alert]:
        """Get recent active alerts.

        For backward compatibility with monitoring.system.alerting.AlertManager.
        Returns alerts from the last hour.
        """
        cutoff = datetime.now() - timedelta(hours=1)
        return [a for a in self.alert_history if a.timestamp > cutoff]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert.

        For backward compatibility with monitoring.system.alerting.AlertManager.
        The new AlertManager doesn't track acknowledgment state, so this is a no-op.

        Args:
            alert_id: Alert ID to acknowledge

        Returns:
            True if alert exists in history
        """
        return any(a.alert_id == alert_id for a in self.alert_history)

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        For backward compatibility with monitoring.system.alerting.AlertManager.
        The new AlertManager doesn't track resolution state, so this is a no-op.

        Args:
            alert_id: Alert ID to resolve

        Returns:
            True if alert exists in history
        """
        return any(a.alert_id == alert_id for a in self.alert_history)
