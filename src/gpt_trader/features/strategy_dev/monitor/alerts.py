"""Alert system for performance monitoring.

Provides:
- AlertRule: Define conditions that trigger alerts
- AlertManager: Manage alert rules and notifications
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from gpt_trader.features.strategy_dev.monitor.metrics import PerformanceSnapshot

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertCondition(Enum):
    """Built-in alert conditions."""

    DRAWDOWN_EXCEEDED = "drawdown_exceeded"
    VOLATILITY_SPIKE = "volatility_spike"
    WIN_RATE_DECLINE = "win_rate_decline"
    REGIME_CHANGE = "regime_change"
    LOSS_STREAK = "loss_streak"
    EQUITY_BELOW = "equity_below"
    RETURN_THRESHOLD = "return_threshold"
    CUSTOM = "custom"


@dataclass
class AlertEvent:
    """Record of a triggered alert."""

    rule_name: str
    severity: AlertSeverity
    condition: AlertCondition
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    snapshot: PerformanceSnapshot | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "condition": self.condition.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """Rule that defines when an alert should trigger.

    Can be based on:
    - Built-in conditions (drawdown, volatility, etc.)
    - Custom condition functions
    """

    name: str
    condition: AlertCondition
    severity: AlertSeverity = AlertSeverity.WARNING
    threshold: float = 0.0
    enabled: bool = True

    # For custom conditions
    custom_check: Callable[[PerformanceSnapshot], bool] | None = None
    custom_message: Callable[[PerformanceSnapshot], str] | None = None

    # Cooldown to prevent alert spam
    cooldown_seconds: int = 300
    _last_triggered: datetime | None = field(default=None, repr=False)

    # For streak-based alerts
    streak_count: int = 3
    _current_streak: int = field(default=0, repr=False)

    def check(
        self, snapshot: PerformanceSnapshot, previous: PerformanceSnapshot | None = None
    ) -> AlertEvent | None:
        """Check if this rule triggers on the given snapshot.

        Args:
            snapshot: Current performance snapshot
            previous: Previous snapshot for comparison

        Returns:
            AlertEvent if triggered, None otherwise
        """
        if not self.enabled:
            return None

        # Check cooldown
        if self._last_triggered:
            elapsed = (datetime.now() - self._last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return None

        triggered = False
        message = ""

        if self.condition == AlertCondition.DRAWDOWN_EXCEEDED:
            if snapshot.drawdown >= self.threshold:
                triggered = True
                message = f"Drawdown {snapshot.drawdown:.2%} exceeds threshold {self.threshold:.2%}"

        elif self.condition == AlertCondition.VOLATILITY_SPIKE:
            if snapshot.volatility >= self.threshold:
                triggered = True
                message = (
                    f"Volatility {snapshot.volatility:.2%} exceeds threshold {self.threshold:.2%}"
                )

        elif self.condition == AlertCondition.WIN_RATE_DECLINE:
            if snapshot.win_rate < self.threshold:
                triggered = True
                message = f"Win rate {snapshot.win_rate:.2%} below threshold {self.threshold:.2%}"

        elif self.condition == AlertCondition.REGIME_CHANGE:
            if previous and snapshot.current_regime != previous.current_regime:
                triggered = True
                message = (
                    f"Regime changed from {previous.current_regime} to {snapshot.current_regime}"
                )

        elif self.condition == AlertCondition.LOSS_STREAK:
            # Track losing streak
            if snapshot.daily_return < 0:
                self._current_streak += 1
            else:
                self._current_streak = 0

            if self._current_streak >= self.streak_count:
                triggered = True
                message = f"Loss streak of {self._current_streak} days reached"
                self._current_streak = 0  # Reset after alert

        elif self.condition == AlertCondition.EQUITY_BELOW:
            if float(snapshot.equity) < self.threshold:
                triggered = True
                message = f"Equity {snapshot.equity} below threshold {self.threshold}"

        elif self.condition == AlertCondition.RETURN_THRESHOLD:
            if snapshot.total_return < self.threshold:
                triggered = True
                message = (
                    f"Total return {snapshot.total_return:.2%} below threshold {self.threshold:.2%}"
                )

        elif self.condition == AlertCondition.CUSTOM:
            if self.custom_check and self.custom_check(snapshot):
                triggered = True
                if self.custom_message:
                    message = self.custom_message(snapshot)
                else:
                    message = f"Custom condition '{self.name}' triggered"

        if triggered:
            self._last_triggered = datetime.now()
            return AlertEvent(
                rule_name=self.name,
                severity=self.severity,
                condition=self.condition,
                message=message,
                snapshot=snapshot,
            )

        return None

    def reset(self) -> None:
        """Reset the rule state."""
        self._last_triggered = None
        self._current_streak = 0


@dataclass
class AlertManager:
    """Manage alert rules and process notifications.

    Features:
    - Register and manage alert rules
    - Process snapshots against all rules
    - Track alert history
    - Notification callbacks
    """

    rules: dict[str, AlertRule] = field(default_factory=dict)
    history: list[AlertEvent] = field(default_factory=list)
    max_history: int = 1000
    _callbacks: list[Callable[[AlertEvent], None]] = field(default_factory=list)
    _previous_snapshot: PerformanceSnapshot | None = None

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: AlertRule to add
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.condition.value})")

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule.

        Args:
            name: Rule name to remove

        Returns:
            True if removed, False if not found
        """
        if name in self.rules:
            del self.rules[name]
            return True
        return False

    def enable_rule(self, name: str) -> None:
        """Enable a rule by name."""
        if name in self.rules:
            self.rules[name].enabled = True

    def disable_rule(self, name: str) -> None:
        """Disable a rule by name."""
        if name in self.rules:
            self.rules[name].enabled = False

    def on_alert(self, callback: Callable[[AlertEvent], None]) -> None:
        """Register callback for alert events.

        Args:
            callback: Function to call when alert triggers
        """
        self._callbacks.append(callback)

    def process(self, snapshot: PerformanceSnapshot) -> list[AlertEvent]:
        """Process a snapshot against all rules.

        Args:
            snapshot: Performance snapshot to check

        Returns:
            List of triggered alert events
        """
        triggered_alerts = []

        for rule in self.rules.values():
            event = rule.check(snapshot, self._previous_snapshot)
            if event:
                triggered_alerts.append(event)
                self._add_to_history(event)
                self._notify_callbacks(event)

        self._previous_snapshot = snapshot
        return triggered_alerts

    def _add_to_history(self, event: AlertEvent) -> None:
        """Add event to history with size limit."""
        self.history.append(event)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def _notify_callbacks(self, event: AlertEvent) -> None:
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def get_recent_alerts(
        self,
        severity: AlertSeverity | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[AlertEvent]:
        """Get recent alerts with optional filters.

        Args:
            severity: Filter by severity
            since: Only alerts after this time
            limit: Maximum number to return

        Returns:
            List of matching alert events
        """
        filtered = self.history

        if severity:
            filtered = [e for e in filtered if e.severity == severity]

        if since:
            filtered = [e for e in filtered if e.timestamp >= since]

        return filtered[-limit:]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get summary of alert activity."""
        last_24h = datetime.now() - timedelta(hours=24)
        recent = [e for e in self.history if e.timestamp >= last_24h]

        by_severity = {}
        for severity in AlertSeverity:
            by_severity[severity.value] = len([e for e in recent if e.severity == severity])

        by_condition = {}
        for event in recent:
            cond = event.condition.value
            by_condition[cond] = by_condition.get(cond, 0) + 1

        return {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "alerts_24h": len(recent),
            "by_severity": by_severity,
            "by_condition": by_condition,
            "last_alert": self.history[-1].to_dict() if self.history else None,
        }

    def clear_history(self) -> int:
        """Clear alert history.

        Returns:
            Number of alerts cleared
        """
        count = len(self.history)
        self.history.clear()
        return count

    def reset_all_rules(self) -> None:
        """Reset state of all rules."""
        for rule in self.rules.values():
            rule.reset()


def create_default_alerts() -> list[AlertRule]:
    """Create a set of default alert rules.

    Returns:
        List of pre-configured alert rules
    """
    return [
        AlertRule(
            name="critical_drawdown",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            severity=AlertSeverity.CRITICAL,
            threshold=0.20,
            cooldown_seconds=3600,
        ),
        AlertRule(
            name="warning_drawdown",
            condition=AlertCondition.DRAWDOWN_EXCEEDED,
            severity=AlertSeverity.WARNING,
            threshold=0.10,
            cooldown_seconds=1800,
        ),
        AlertRule(
            name="volatility_spike",
            condition=AlertCondition.VOLATILITY_SPIKE,
            severity=AlertSeverity.WARNING,
            threshold=0.50,
            cooldown_seconds=3600,
        ),
        AlertRule(
            name="regime_change",
            condition=AlertCondition.REGIME_CHANGE,
            severity=AlertSeverity.INFO,
            cooldown_seconds=0,  # Always alert on regime change
        ),
        AlertRule(
            name="loss_streak",
            condition=AlertCondition.LOSS_STREAK,
            severity=AlertSeverity.WARNING,
            streak_count=5,
            cooldown_seconds=86400,  # Once per day
        ),
        AlertRule(
            name="win_rate_low",
            condition=AlertCondition.WIN_RATE_DECLINE,
            severity=AlertSeverity.WARNING,
            threshold=0.40,
            cooldown_seconds=3600,
        ),
    ]
