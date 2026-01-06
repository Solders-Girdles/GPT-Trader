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

from gpt_trader.tui.thresholds import DEFAULT_ORDER_THRESHOLDS
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


class AlertCategory(Enum):
    """Alert categories aligned with system log levels.

    These categories allow filtering alerts similar to log level filtering:
    - TRADE: Trade executions, fills, order updates
    - POSITION: Position opens/closes, P&L milestones
    - STRATEGY: Strategy signals, decisions, indicator alerts
    - RISK: Risk warnings, limit breaches, guard activations
    - SYSTEM: Connection events, rate limits, circuit breakers
    - ERROR: Errors and exceptions
    """

    TRADE = "trade"
    POSITION = "position"
    STRATEGY = "strategy"
    RISK = "risk"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class Alert:
    """Represents a triggered alert."""

    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory = AlertCategory.SYSTEM
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlertRule:
    """Defines a condition that triggers an alert.

    Attributes:
        rule_id: Unique identifier for this rule.
        title: Alert title shown in notification.
        condition: Function that takes TuiState and returns (triggered, message).
        severity: Alert severity level.
        category: Alert category for filtering.
        cooldown: Minimum seconds between repeated alerts for this rule.
        enabled: Whether this rule is active.
    """

    rule_id: str
    title: str
    condition: Callable[[TuiState], tuple[bool, str]]
    severity: AlertSeverity = AlertSeverity.WARNING
    category: AlertCategory = AlertCategory.SYSTEM
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

        # Connection lost (SYSTEM category)
        self.add_rule(
            AlertRule(
                rule_id="connection_lost",
                title="Connection Lost",
                condition=lambda state: (
                    state.system_data.connection_status == "DISCONNECTED",
                    "Lost connection to exchange. Bot may not receive updates.",
                ),
                severity=AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                cooldown=30.0,
            )
        )

        # High rate limit usage (SYSTEM category)
        self.add_rule(
            AlertRule(
                rule_id="rate_limit_high",
                title="Rate Limit Warning",
                condition=self._check_rate_limit,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                cooldown=120.0,
            )
        )

        # Reduce-only mode activated (RISK category)
        self.add_rule(
            AlertRule(
                rule_id="reduce_only_active",
                title="Reduce-Only Mode",
                condition=lambda state: (
                    state.risk_data.reduce_only_mode,
                    f"Reduce-only mode active: {state.risk_data.reduce_only_reason}",
                ),
                severity=AlertSeverity.WARNING,
                category=AlertCategory.RISK,
                cooldown=300.0,  # 5 minute cooldown
            )
        )

        # Daily loss limit approaching (RISK category)
        self.add_rule(
            AlertRule(
                rule_id="daily_loss_warning",
                title="Daily Loss Warning",
                condition=self._check_daily_loss,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.RISK,
                cooldown=300.0,
            )
        )

        # Large unrealized loss (POSITION category)
        self.add_rule(
            AlertRule(
                rule_id="large_unrealized_loss",
                title="Large Unrealized Loss",
                condition=self._check_unrealized_loss,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.POSITION,
                cooldown=180.0,
            )
        )

        # Bot stopped unexpectedly (SYSTEM category)
        self.add_rule(
            AlertRule(
                rule_id="bot_stopped",
                title="Bot Stopped",
                condition=self._check_bot_stopped,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                cooldown=60.0,
            )
        )

        # Stale open orders (TRADE category)
        self.add_rule(
            AlertRule(
                rule_id="stale_open_orders",
                title="Stale Orders",
                condition=self._check_stale_orders,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.TRADE,
                cooldown=120.0,  # 2 minute cooldown
            )
        )

        # Rejected/failed orders (TRADE category) - true errors only
        self.add_rule(
            AlertRule(
                rule_id="failed_orders",
                title="Order Failed",
                condition=self._check_failed_orders,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.TRADE,
                cooldown=30.0,
            )
        )

        # Expired orders (TRADE category) - warning level, may indicate stale strategy
        self.add_rule(
            AlertRule(
                rule_id="expired_orders",
                title="Order Expired",
                condition=self._check_expired_orders,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.TRADE,
                cooldown=60.0,  # 1 minute cooldown
            )
        )

        # === Execution Health Alerts (SYSTEM category) ===

        # Circuit breaker open - execution paused
        self.add_rule(
            AlertRule(
                rule_id="circuit_breaker_open",
                title="Circuit Breaker Open",
                condition=self._check_circuit_breaker,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                cooldown=60.0,
            )
        )

        # Execution success rate critical (<80%)
        self.add_rule(
            AlertRule(
                rule_id="execution_critical",
                title="Execution Critical",
                condition=self._check_execution_critical,
                severity=AlertSeverity.ERROR,
                category=AlertCategory.SYSTEM,
                cooldown=120.0,  # 2 minute cooldown
            )
        )

        # Execution success rate warning (<95%)
        self.add_rule(
            AlertRule(
                rule_id="execution_degraded",
                title="Execution Degraded",
                condition=self._check_execution_degraded,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                cooldown=180.0,  # 3 minute cooldown
            )
        )

        # High p95 latency (>500ms)
        self.add_rule(
            AlertRule(
                rule_id="execution_p95_spike",
                title="Latency Spike",
                condition=self._check_p95_latency,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                cooldown=120.0,
            )
        )

        # High retry rate (>0.5)
        self.add_rule(
            AlertRule(
                rule_id="execution_retry_high",
                title="High Retry Rate",
                condition=self._check_retry_rate,
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SYSTEM,
                cooldown=180.0,
            )
        )

        logger.debug("Registered %s default alert rules", len(self._rules))

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

    def _check_stale_orders(self, state: TuiState) -> tuple[bool, str]:
        """Check for open orders that are older than threshold.

        Uses centralized threshold from gpt_trader.tui.thresholds.OrderThresholds.
        Alert fires at the CRITICAL threshold (same as red coloring in UI).
        """
        # Use CRITICAL threshold (age_warn) for alert - aligns with red color in UI
        stale_threshold = DEFAULT_ORDER_THRESHOLDS.age_warn

        stale_orders = []
        for order in state.order_data.orders:
            # Only check open/pending orders
            if order.status.upper() not in ("OPEN", "PENDING"):
                continue

            # Check creation time
            if order.creation_time > 0:
                age = time.time() - order.creation_time
                if age >= stale_threshold:
                    stale_orders.append((order.symbol, int(age)))

        if stale_orders:
            # Report oldest stale order
            stale_orders.sort(key=lambda x: x[1], reverse=True)
            oldest_symbol, oldest_age = stale_orders[0]
            count = len(stale_orders)
            if count == 1:
                return True, f"Order for {oldest_symbol} open for {oldest_age}s."
            else:
                return True, f"{count} orders stale. Oldest: {oldest_symbol} ({oldest_age}s)."

        return False, ""

    def _check_failed_orders(self, state: TuiState) -> tuple[bool, str]:
        """Check for rejected or failed orders.

        Only alerts on true failures (REJECTED, FAILED) - not CANCELLED
        (user-initiated) or EXPIRED (handled by separate expiration rule).
        """
        # Only alert on true failures, not user-initiated cancellations
        error_statuses = {"REJECTED", "FAILED"}

        failed_orders = [
            order for order in state.order_data.orders if order.status.upper() in error_statuses
        ]

        if failed_orders:
            order = failed_orders[0]
            return True, f"Order for {order.symbol} {order.status.lower()}."

        return False, ""

    def _check_expired_orders(self, state: TuiState) -> tuple[bool, str]:
        """Check for expired orders (time-based, not user-initiated).

        Order expiration may indicate stale limit prices or strategy issues.
        """
        expired_orders = [
            order for order in state.order_data.orders if order.status.upper() == "EXPIRED"
        ]

        if expired_orders:
            order = expired_orders[0]
            count = len(expired_orders)
            if count == 1:
                return True, f"Order for {order.symbol} expired."
            return True, f"{count} orders expired."

        return False, ""

    # === Execution Health Check Methods ===

    def _check_circuit_breaker(self, state: TuiState) -> tuple[bool, str]:
        """Check if circuit breaker is open."""
        try:
            if state.resilience_data.any_circuit_open:
                return True, "Execution paused. Broker may be experiencing issues."
        except (AttributeError, TypeError):
            pass
        return False, ""

    def _check_execution_critical(self, state: TuiState) -> tuple[bool, str]:
        """Check for critically low execution success rate (<80%)."""
        try:
            exec_data = state.execution_data
            # Only alert if we have meaningful sample size
            if exec_data.submissions_total >= 5:
                if exec_data.success_rate < 80.0:
                    return (
                        True,
                        f"Success rate at {exec_data.success_rate:.0f}%. Check broker connection.",
                    )
        except (AttributeError, TypeError):
            pass
        return False, ""

    def _check_execution_degraded(self, state: TuiState) -> tuple[bool, str]:
        """Check for degraded execution success rate (<95%, >=80%)."""
        try:
            exec_data = state.execution_data
            # Only alert if we have meaningful sample size
            if exec_data.submissions_total >= 10:
                # Only fire warning if not already critical
                if 80.0 <= exec_data.success_rate < 95.0:
                    return (
                        True,
                        f"Success rate at {exec_data.success_rate:.0f}%. Monitor closely.",
                    )
        except (AttributeError, TypeError):
            pass
        return False, ""

    def _check_p95_latency(self, state: TuiState) -> tuple[bool, str]:
        """Check for high p95 latency (>500ms)."""
        try:
            exec_data = state.execution_data
            # Only alert if we have meaningful sample size
            if exec_data.submissions_total >= 10:
                if exec_data.p95_latency_ms > 500.0:
                    return (
                        True,
                        f"p95 latency at {exec_data.p95_latency_ms:.0f}ms. Broker may be slow.",
                    )
        except (AttributeError, TypeError):
            pass
        return False, ""

    def _check_retry_rate(self, state: TuiState) -> tuple[bool, str]:
        """Check for high retry rate (>0.5)."""
        try:
            exec_data = state.execution_data
            # Only alert if we have meaningful sample size
            if exec_data.submissions_total >= 10:
                if exec_data.retry_rate > 0.5:
                    return (
                        True,
                        f"Retry rate at {exec_data.retry_rate:.1f}x. Intermittent failures.",
                    )
        except (AttributeError, TypeError):
            pass
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
                        category=rule.category,
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
        from gpt_trader.tui.notification_helpers import TIMEOUT_ALERT, get_recovery_hint

        try:
            # Add recovery hint if available for this rule
            hint = get_recovery_hint(alert.rule_id)
            message = f"{alert.message} — {hint}" if hint else alert.message

            self.app.notify(
                message,
                title=alert.title,
                severity=alert.severity.value,
                timeout=TIMEOUT_ALERT,
            )
            logger.info(f"Alert triggered: {alert.title} - {alert.message}")
        except Exception as e:
            logger.error(f"Failed to send alert notification: {e}")

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history, trimming if necessary."""
        self._alert_history.append(alert)
        if len(self._alert_history) > self.MAX_HISTORY:
            self._alert_history = self._alert_history[-self.MAX_HISTORY :]

    def get_history(
        self,
        limit: int = 20,
        categories: set[AlertCategory] | None = None,
        min_severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get recent alert history with optional filtering.

        Args:
            limit: Maximum number of alerts to return.
            categories: Filter by categories (None = all categories).
            min_severity: Minimum severity to include (None = all severities).

        Returns:
            List of recent alerts, newest first.
        """
        alerts = self._alert_history

        # Filter by category
        if categories:
            alerts = [a for a in alerts if a.category in categories]

        # Filter by severity
        if min_severity:
            severity_order = {
                AlertSeverity.INFORMATION: 0,
                AlertSeverity.WARNING: 1,
                AlertSeverity.ERROR: 2,
            }
            min_level = severity_order.get(min_severity, 0)
            alerts = [a for a in alerts if severity_order.get(a.severity, 0) >= min_level]

        return list(reversed(alerts[-limit:]))

    def get_history_by_category(self, category: AlertCategory, limit: int = 20) -> list[Alert]:
        """Get alert history filtered by a single category.

        Args:
            category: Category to filter by.
            limit: Maximum number of alerts to return.

        Returns:
            List of alerts in the specified category, newest first.
        """
        return self.get_history(limit=limit, categories={category})

    def get_category_counts(self) -> dict[AlertCategory, int]:
        """Get count of alerts per category.

        Returns:
            Dict mapping category to alert count.
        """
        counts: dict[AlertCategory, int] = {cat: 0 for cat in AlertCategory}
        for alert in self._alert_history:
            counts[alert.category] = counts.get(alert.category, 0) + 1
        return counts

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
