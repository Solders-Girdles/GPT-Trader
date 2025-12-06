"""
Alert Manager Service for TUI.

Monitors bot state and triggers notifications for important events.
Supports configurable alert rules with cooldown to prevent spam.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from gpt_trader.utilities.logging_patterns import get_logger

if TYPE_CHECKING:
    from gpt_trader.tui.app import TraderApp
    from gpt_trader.tui.state import TuiState

logger = get_logger(__name__, component="tui")


class AlertSeverity(Enum):
    """Alert severity levels matching Textual notification severities."""

    INFORMATION = "information"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Alert:
    """Represents a triggered alert."""

    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlertRule:
    """Defines a condition that triggers an alert.

    Attributes:
        rule_id: Unique identifier for this rule.
        title: Alert title shown in notification.
        condition: Function that takes TuiState and returns (triggered, message).
        severity: Alert severity level.
        cooldown: Minimum seconds between repeated alerts for this rule.
        enabled: Whether this rule is active.
    """

    rule_id: str
    title: str
    condition: Callable[[TuiState], tuple[bool, str]]
    severity: AlertSeverity = AlertSeverity.WARNING
    cooldown: float = 60.0  # Default 1 minute cooldown
    enabled: bool = True


class AlertManager:
    """Manages alert rules and triggers notifications.

    Features:
    - Configurable alert rules with conditions
    - Cooldown tracking to prevent notification spam
    - Integration with Textual's notification system
    - Alert history for debugging
    """

    # Maximum alerts to keep in history
    MAX_HISTORY = 100

    def __init__(self, app: TraderApp) -> None:
        """Initialize AlertManager.

        Args:
            app: Reference to the TraderApp for notifications.
        """
        self.app = app
        self._rules: dict[str, AlertRule] = {}
        self._last_triggered: dict[str, float] = {}
        self._alert_history: list[Alert] = []

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register built-in alert rules for common trading scenarios."""

        # Connection lost
        self.add_rule(
            AlertRule(
                rule_id="connection_lost",
                title="Connection Lost",
                condition=lambda state: (
                    state.system_data.connection_status == "DISCONNECTED",
                    "Lost connection to exchange. Bot may not receive updates.",
                ),
                severity=AlertSeverity.ERROR,
                cooldown=30.0,
            )
        )

        # High rate limit usage
        self.add_rule(
            AlertRule(
                rule_id="rate_limit_high",
                title="Rate Limit Warning",
                condition=self._check_rate_limit,
                severity=AlertSeverity.WARNING,
                cooldown=120.0,
            )
        )

        # Reduce-only mode activated
        self.add_rule(
            AlertRule(
                rule_id="reduce_only_active",
                title="Reduce-Only Mode",
                condition=lambda state: (
                    state.risk_data.reduce_only_mode,
                    f"Reduce-only mode active: {state.risk_data.reduce_only_reason}",
                ),
                severity=AlertSeverity.WARNING,
                cooldown=300.0,  # 5 minute cooldown
            )
        )

        # Daily loss limit approaching
        self.add_rule(
            AlertRule(
                rule_id="daily_loss_warning",
                title="Daily Loss Warning",
                condition=self._check_daily_loss,
                severity=AlertSeverity.WARNING,
                cooldown=300.0,
            )
        )

        # Large unrealized loss
        self.add_rule(
            AlertRule(
                rule_id="large_unrealized_loss",
                title="Large Unrealized Loss",
                condition=self._check_unrealized_loss,
                severity=AlertSeverity.WARNING,
                cooldown=180.0,
            )
        )

        # Bot stopped unexpectedly
        self.add_rule(
            AlertRule(
                rule_id="bot_stopped",
                title="Bot Stopped",
                condition=self._check_bot_stopped,
                severity=AlertSeverity.WARNING,
                cooldown=60.0,
            )
        )

        logger.info(f"Registered {len(self._rules)} default alert rules")

    def _check_rate_limit(self, state: TuiState) -> tuple[bool, str]:
        """Check if rate limit usage is high."""
        try:
            usage_str = state.system_data.rate_limit_usage.rstrip("%")
            usage = float(usage_str)
            if usage >= 80:
                return True, f"Rate limit usage at {usage:.0f}%. Slow down requests."
        except (ValueError, AttributeError):
            pass
        return False, ""

    def _check_daily_loss(self, state: TuiState) -> tuple[bool, str]:
        """Check if daily loss is approaching limit."""
        if state.risk_data.daily_loss_limit_pct > 0:
            current = abs(state.risk_data.current_daily_loss_pct)
            limit = state.risk_data.daily_loss_limit_pct
            if current >= limit * 0.75:  # 75% of limit
                pct_of_limit = (current / limit) * 100
                return True, f"Daily loss at {pct_of_limit:.0f}% of limit."
        return False, ""

    def _check_unrealized_loss(self, state: TuiState) -> tuple[bool, str]:
        """Check for large unrealized losses."""
        total_pnl = state.position_data.total_unrealized_pnl
        if total_pnl < Decimal("-500"):  # Configurable threshold
            return True, f"Unrealized loss: ${total_pnl:,.2f}"
        return False, ""

    def _check_bot_stopped(self, state: TuiState) -> tuple[bool, str]:
        """Check if bot stopped while positions are open."""
        if not state.running and len(state.position_data.positions) > 0:
            count = len(state.position_data.positions)
            return True, f"Bot stopped with {count} open position(s)."
        return False, ""

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.

        Args:
            rule: AlertRule to add.
        """
        self._rules[rule.rule_id] = rule
        logger.debug(f"Added alert rule: {rule.rule_id}")

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule.

        Args:
            rule_id: ID of rule to remove.

        Returns:
            True if rule was removed, False if not found.
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.debug(f"Removed alert rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> None:
        """Enable an alert rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str) -> None:
        """Disable an alert rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False

    def check_alerts(self, state: TuiState) -> list[Alert]:
        """Check all rules against current state and trigger alerts.

        Args:
            state: Current TuiState to check.

        Returns:
            List of triggered alerts.
        """
        triggered: list[Alert] = []
        now = time.time()

        for rule in self._rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            last_time = self._last_triggered.get(rule.rule_id, 0)
            if now - last_time < rule.cooldown:
                continue

            # Check condition
            try:
                is_triggered, message = rule.condition(state)
                if is_triggered:
                    alert = Alert(
                        rule_id=rule.rule_id,
                        title=rule.title,
                        message=message,
                        severity=rule.severity,
                        timestamp=now,
                    )
                    triggered.append(alert)
                    self._last_triggered[rule.rule_id] = now
                    self._add_to_history(alert)

                    # Send notification
                    self._notify(alert)

            except Exception as e:
                logger.debug(f"Error checking alert rule {rule.rule_id}: {e}")

        return triggered

    def _notify(self, alert: Alert) -> None:
        """Send notification for an alert.

        Args:
            alert: Alert to notify about.
        """
        try:
            self.app.notify(
                alert.message,
                title=alert.title,
                severity=alert.severity.value,
                timeout=10,
            )
            logger.info(f"Alert triggered: {alert.title} - {alert.message}")
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history, trimming if necessary."""
        self._alert_history.append(alert)
        if len(self._alert_history) > self.MAX_HISTORY:
            self._alert_history = self._alert_history[-self.MAX_HISTORY :]

    def get_history(self, limit: int = 20) -> list[Alert]:
        """Get recent alert history.

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            List of recent alerts, newest first.
        """
        return list(reversed(self._alert_history[-limit:]))

    def clear_history(self) -> None:
        """Clear alert history."""
        self._alert_history.clear()

    def reset_cooldowns(self) -> None:
        """Reset all cooldown timers."""
        self._last_triggered.clear()
        logger.debug("Alert cooldowns reset")

    def get_rule_status(self) -> dict[str, dict]:
        """Get status of all rules.

        Returns:
            Dict mapping rule_id to status info.
        """
        now = time.time()
        status = {}
        for rule_id, rule in self._rules.items():
            last = self._last_triggered.get(rule_id, 0)
            cooldown_remaining = max(0, rule.cooldown - (now - last))
            status[rule_id] = {
                "title": rule.title,
                "enabled": rule.enabled,
                "severity": rule.severity.value,
                "cooldown": rule.cooldown,
                "cooldown_remaining": cooldown_remaining,
                "last_triggered": last if last > 0 else None,
            }
        return status
