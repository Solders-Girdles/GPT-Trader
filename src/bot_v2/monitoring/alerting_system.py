from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, DefaultDict


class AlertLevel(Enum):
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class Alert:
    id: str
    level: AlertLevel
    category: str
    message: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    count: int = 1
    last_seen: datetime = field(default_factory=datetime.now)


@dataclass
class AlertRule:
    name: str
    condition: Callable[[dict[str, Any]], bool]
    level: AlertLevel
    category: str
    message_template: str
    cooldown_minutes: int = 5


class AlertingSystem:
    def __init__(self) -> None:
        self.alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.rules: list[AlertRule] = []
        self.handlers: list[Callable[[Alert], None]] = []
        self.suppression_rules: dict[str, datetime] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger("alerting")

        # Default handlers
        self._setup_default_handlers()
        # Default rules
        self._setup_default_rules()

    def register_rule(self, rule: AlertRule) -> None:
        """Register a new alert rule."""
        self.rules.append(rule)

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a new alert handler."""
        self.handlers.append(handler)

    def check_metric(
        self, metric_name: str, value: float, context: dict[str, Any] | None = None
    ) -> None:
        """Check a metric value against all registered rules."""
        ctx = dict(context or {})
        ctx["metric_name"] = metric_name
        ctx["value"] = value

        for rule in self.rules:
            try:
                if rule.condition(ctx):
                    self._trigger_alert(rule, ctx)
            except Exception as e:
                self.logger.error(f"Error checking rule {rule.name}: {e}")

    def trigger_alert(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Manually trigger an alert."""
        alert_id = f"{category}_{hash(message)}"

        with self._lock:
            # Check for deduplication
            if alert_id in self.alerts:
                existing = self.alerts[alert_id]
                existing.count += 1
                existing.last_seen = datetime.now()

                # Check suppression rules
                if self._should_suppress(existing):
                    return
            else:
                alert = Alert(
                    id=alert_id,
                    level=level,
                    category=category,
                    message=message,
                    timestamp=datetime.now(),
                    metadata=dict(metadata or {}),
                )
                self.alerts[alert_id] = alert

                # Add to history
                self.alert_history.append(alert)
                if len(self.alert_history) > 1000:
                    self.alert_history.pop(0)

                # Dispatch to handlers
                self._dispatch_alert(alert)

    def _trigger_alert(self, rule: AlertRule, context: dict[str, Any]) -> None:
        """Trigger an alert based on a rule match."""
        # Check cooldown
        rule_key = f"{rule.name}_{rule.category}"
        if rule_key in self.suppression_rules:
            last_alert = self.suppression_rules[rule_key]
            if datetime.now() - last_alert < timedelta(minutes=rule.cooldown_minutes):
                return

        # Format message
        message = rule.message_template.format(**context)

        # Trigger alert
        self.trigger_alert(rule.level, rule.category, message, context)

        # Update suppression
        self.suppression_rules[rule_key] = datetime.now()

    def _should_suppress(self, alert: Alert) -> bool:
        """Check if an alert should be suppressed due to frequency."""
        # Suppress if too many occurrences in short time
        if alert.count > 10:
            time_diff = datetime.now() - alert.timestamp
            if time_diff < timedelta(minutes=5):
                return alert.count % 10 != 0  # Only alert every 10th occurrence
        return False

    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch an alert to all registered handlers."""
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")

    def _setup_default_handlers(self) -> None:
        """Set up default alert handlers."""

        # Console handler
        def console_handler(alert: Alert) -> None:
            icon = (
                "ℹ️"
                if alert.level == AlertLevel.INFO
                else "⚠️" if alert.level == AlertLevel.WARNING else "❌"
            )
            self.logger.info(
                "%s [%s] %s: %s",
                icon,
                alert.level.name,
                alert.category,
                alert.message,
            )

        # Log handler
        def log_handler(alert: Alert) -> None:
            if alert.level == AlertLevel.CRITICAL:
                self.logger.critical(alert.message)
            elif alert.level == AlertLevel.ERROR:
                self.logger.error(alert.message)
            elif alert.level == AlertLevel.WARNING:
                self.logger.warning(alert.message)
            else:
                self.logger.info(alert.message)

        # File handler for critical alerts
        def file_handler(alert: Alert) -> None:
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
                try:
                    with open("critical_alerts.log", "a") as f:
                        f.write(
                            f"{alert.timestamp.isoformat()} [{alert.level.name}] {alert.category}: {alert.message}\n"
                        )
                except Exception as e:
                    self.logger.error(f"Error writing to alert file: {e}")

        self.register_handler(console_handler)
        self.register_handler(log_handler)
        self.register_handler(file_handler)

    def _setup_default_rules(self) -> None:
        """Set up default alert rules for common conditions."""
        # High CPU usage
        self.register_rule(
            AlertRule(
                name="high_cpu",
                condition=lambda ctx: ctx.get("metric_name") == "cpu_percent"
                and ctx.get("value", 0) > 80,
                level=AlertLevel.WARNING,
                category="resource",
                message_template="High CPU usage: {value:.1f}%",
            )
        )

        # High memory usage
        self.register_rule(
            AlertRule(
                name="high_memory",
                condition=lambda ctx: ctx.get("metric_name") == "memory_percent"
                and ctx.get("value", 0) > 85,
                level=AlertLevel.WARNING,
                category="resource",
                message_template="High memory usage: {value:.1f}%",
            )
        )

        # Critical memory usage
        self.register_rule(
            AlertRule(
                name="critical_memory",
                condition=lambda ctx: ctx.get("metric_name") == "memory_percent"
                and ctx.get("value", 0) > 95,
                level=AlertLevel.CRITICAL,
                category="resource",
                message_template="Critical memory usage: {value:.1f}%",
            )
        )

        # Workflow failures
        self.register_rule(
            AlertRule(
                name="workflow_failed",
                condition=lambda ctx: ctx.get("workflow_status") == "failed",
                level=AlertLevel.ERROR,
                category="workflow",
                message_template="Workflow {workflow_name} failed",
            )
        )

        # Trading errors
        self.register_rule(
            AlertRule(
                name="trading_error",
                condition=lambda ctx: ctx.get("category") == "trading"
                and ctx.get("error_count", 0) > 5,
                level=AlertLevel.CRITICAL,
                category="trading",
                message_template="Multiple trading errors detected: {error_count} errors",
            )
        )

        # Data pipeline issues
        self.register_rule(
            AlertRule(
                name="data_stale",
                condition=lambda ctx: ctx.get("metric_name") == "data_age_minutes"
                and ctx.get("value", 0) > 30,
                level=AlertLevel.WARNING,
                category="data",
                message_template="Stale data detected: {value:.1f} minutes old",
            )
        )

    def get_active_alerts(self) -> list[Alert]:
        """Get all currently active alerts."""
        with self._lock:
            return list(self.alerts.values())

    def clear_alert(self, alert_id: str) -> None:
        """Clear a specific alert by ID."""
        with self._lock:
            if alert_id in self.alerts:
                del self.alerts[alert_id]

    def clear_alerts_by_category(self, category: str) -> None:
        """Clear all alerts in a specific category."""
        with self._lock:
            to_remove = [
                alert_id for alert_id, alert in self.alerts.items() if alert.category == category
            ]
            for alert_id in to_remove:
                del self.alerts[alert_id]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get a summary of current alert status."""
        with self._lock:
            by_level: DefaultDict[str, int] = defaultdict(int)
            by_category: DefaultDict[str, int] = defaultdict(int)

            for alert in self.alerts.values():
                by_level[alert.level.name] += 1
                by_category[alert.category] += 1

            return {
                "total_active": len(self.alerts),
                "by_level": dict(by_level),
                "by_category": dict(by_category),
                "critical_alerts": [
                    a for a in self.alerts.values() if a.level == AlertLevel.CRITICAL
                ],
                "recent_alerts": self.alert_history[-10:] if self.alert_history else [],
            }

    def export_alert_history(self, filename: str | None = None) -> str:
        """Export alert history to JSON file."""
        if not filename:
            filename = f"alert_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        history_data: list[dict[str, Any]] = []
        for alert in self.alert_history:
            history_data.append(
                {
                    "id": alert.id,
                    "level": alert.level.name,
                    "category": alert.category,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "count": alert.count,
                    "metadata": alert.metadata,
                }
            )

        with open(filename, "w") as f:
            json.dump(history_data, f, indent=2)

        return filename
