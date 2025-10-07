"""
Runtime Guard Alerting System

Monitors critical runtime conditions and triggers alerts when thresholds are breached.
Implements circuit breakers for daily loss limits, stale marks, and error rates.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, DecimalException, InvalidOperation
from enum import Enum
from typing import Any

from bot_v2.monitoring.alert_types import AlertSeverity


logger = logging.getLogger(__name__)


class GuardStatus(Enum):
    """Guard status states."""

    HEALTHY = "healthy"
    WARNING = "warning"
    BREACHED = "breached"
    DISABLED = "disabled"


@dataclass
class Alert:
    """Alert data structure."""

    timestamp: datetime
    guard_name: str
    severity: AlertSeverity
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "guard_name": self.guard_name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
        }


@dataclass
class GuardConfig:
    """Configuration for a runtime guard."""

    name: str
    enabled: bool = True
    threshold: float = 0.0
    window_seconds: int = 60
    severity: AlertSeverity = AlertSeverity.WARNING
    auto_shutdown: bool = False
    cooldown_seconds: int = 300  # Prevent alert spam


class RuntimeGuard:
    """Base class for runtime guards."""

    def __init__(self, config: GuardConfig) -> None:
        self.config = config
        self.status: GuardStatus = GuardStatus.HEALTHY if config.enabled else GuardStatus.DISABLED
        self.last_check: datetime = datetime.now()
        self.last_alert: datetime | None = None
        self.breach_count: int = 0
        self.alerts: list[Alert] = []

    def check(self, context: dict[str, Any]) -> Alert | None:
        """
        Check guard condition and return alert if breached.

        Args:
            context: Current runtime context

        Returns:
            Alert if condition breached, None otherwise
        """
        if not self.config.enabled:
            return None

        # Check cooldown
        if self.last_alert:
            elapsed = (datetime.now() - self.last_alert).total_seconds()
            if elapsed < self.config.cooldown_seconds:
                return None

        # Perform guard-specific check
        is_breached, message = self._evaluate(context)

        if is_breached:
            self.status = GuardStatus.BREACHED
            self.breach_count += 1
            alert = Alert(
                timestamp=datetime.now(),
                guard_name=self.config.name,
                severity=self.config.severity,
                message=message,
                context=dict(context),
            )
            self.alerts.append(alert)
            self.last_alert = datetime.now()
            return alert

        # Check if we should downgrade from breached to warning
        if self.status == GuardStatus.BREACHED:
            self.status = GuardStatus.WARNING

        self.last_check = datetime.now()
        return None

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Generic evaluation hook used when no specialised guard exists.

        The default implementation inspects the context for a numeric value,
        compares it against a threshold, and emits a human-readable message
        when the configured comparison is breached. Subclasses can override
        this method for bespoke guard behaviour while still calling into the
        base implementation for convenience.
        """

        def _to_decimal(raw: Any) -> Decimal | None:
            try:
                return Decimal(str(raw))
            except (TypeError, ValueError, DecimalException, InvalidOperation):
                return None

        metric_key = context.get("metric_key")
        candidate_keys: list[str] = []
        if isinstance(metric_key, str):
            candidate_keys.append(metric_key)
        candidate_keys.extend(
            [
                "value",
                "metric",
                self.config.name,
                f"{self.config.name}_value",
            ]
        )

        raw_value = None
        for key in candidate_keys:
            if key in context:
                raw_value = context[key]
                break

        value = _to_decimal(raw_value)
        if value is None:
            return False, ""

        threshold_sources = (
            context.get("threshold_override"),
            context.get("threshold"),
            context.get("limit"),
            self.config.threshold,
        )
        threshold_raw = next((item for item in threshold_sources if item is not None), None)
        threshold = _to_decimal(threshold_raw)
        if threshold is None:
            return False, ""

        comparison = str(context.get("comparison", context.get("operator", "gt"))).lower()
        comparisons: dict[str, tuple[str, Callable[[Decimal, Decimal], bool], bool]] = {
            "gt": ("exceeded", lambda v, t: v > t, False),
            "+gt": ("exceeded", lambda v, t: v > t, False),
            ">": ("exceeded", lambda v, t: v > t, False),
            "ge": ("reached threshold", lambda v, t: v >= t, False),
            "gte": ("reached threshold", lambda v, t: v >= t, False),
            ">=": ("reached threshold", lambda v, t: v >= t, False),
            "lt": ("dropped below", lambda v, t: v < t, False),
            "lte": ("dropped to or below", lambda v, t: v <= t, False),
            "le": ("dropped to or below", lambda v, t: v <= t, False),
            "<": ("dropped below", lambda v, t: v < t, False),
            "<=": ("dropped to or below", lambda v, t: v <= t, False),
            "eq": ("matched", lambda v, t: v == t, False),
            "==": ("matched", lambda v, t: v == t, False),
            "ne": ("deviated from", lambda v, t: v != t, False),
            "!=": ("deviated from", lambda v, t: v != t, False),
            "abs_gt": (
                "exceeded absolute limit",
                lambda v, t: v.copy_abs() > t,
                True,
            ),
            "abs_gte": (
                "reached absolute limit",
                lambda v, t: v.copy_abs() >= t,
                True,
            ),
            "abs_ge": (
                "reached absolute limit",
                lambda v, t: v.copy_abs() >= t,
                True,
            ),
        }

        descriptor, comparator, use_absolute = comparisons.get(comparison, comparisons["gt"])

        evaluated_value = value.copy_abs() if use_absolute else value
        if comparator(value, threshold):
            label = context.get("label") or self.config.name.replace("_", " ")
            units = context.get("units")

            def _fmt(num: Decimal) -> str:
                text = format(num, "f")
                if "." in text:
                    text = text.rstrip("0").rstrip(".")
                return text

            suffix = f" {units}" if units else ""
            message_templates = {
                "matched": "{label} matched expected value {threshold}{suffix}",
                "deviated from": (
                    "{label} deviated from expected value {threshold}{suffix} "
                    "(current: {value}{suffix})"
                ),
                "exceeded absolute limit": (
                    "{label} exceeded absolute limit {threshold}{suffix} "
                    "(|current|: {value}{suffix})"
                ),
                "reached absolute limit": (
                    "{label} reached absolute limit {threshold}{suffix} "
                    "(|current|: {value}{suffix})"
                ),
            }
            template = message_templates.get(
                descriptor,
                "{label} {descriptor} {threshold}{suffix} (current: {value}{suffix})",
            )
            message = template.format(
                label=label,
                descriptor=descriptor,
                threshold=_fmt(threshold),
                value=_fmt(evaluated_value),
                suffix=suffix,
            )
            return True, message

        # Allow callers to express warning thresholds without triggering an alert
        warning_sources = (
            context.get("warning_threshold"),
            context.get("warning_limit"),
        )
        warning_raw = next((item for item in warning_sources if item is not None), None)
        if warning_raw is None and context.get("warning_ratio") is not None:
            ratio = _to_decimal(context.get("warning_ratio"))
            if ratio is not None:
                warning_raw = threshold * ratio

        warning_threshold = _to_decimal(warning_raw)
        if (
            warning_threshold is not None
            and self.status == GuardStatus.HEALTHY
            and comparator(value, warning_threshold)
        ):
            self.status = GuardStatus.WARNING

        return False, ""

    def reset(self) -> None:
        """Reset guard state."""
        self.status = GuardStatus.HEALTHY if self.config.enabled else GuardStatus.DISABLED
        self.breach_count = 0
        self.last_alert = None


class DailyLossGuard(RuntimeGuard):
    """Monitor daily loss limits."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.daily_pnl: Decimal = Decimal("0")
        self.last_reset: date = datetime.now().date()

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if daily loss limit is breached."""
        # Reset daily counter if new day
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = Decimal("0")
            self.last_reset = current_date

        # Update PnL
        try:
            pnl = Decimal(str(context.get("pnl", 0)))
        except (TypeError, ValueError, DecimalException):
            return False, ""
        self.daily_pnl += pnl

        threshold = Decimal(str(abs(self.config.threshold)))
        if threshold == Decimal("0"):
            return False, ""

        # Check threshold
        if self.daily_pnl < -threshold:
            loss_amount = abs(self.daily_pnl)
            message = (
                f"Daily loss limit breached: ${loss_amount:.2f} "
                f"(limit: ${self.config.threshold:.2f})"
            )
            return True, message

        warning_threshold = threshold * Decimal("0.5")
        if self.daily_pnl <= -warning_threshold and self.status == GuardStatus.HEALTHY:
            self.status = GuardStatus.WARNING

        return False, ""


class StaleMarkGuard(RuntimeGuard):
    """Monitor for stale market data."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.last_marks: dict[str, datetime] = {}

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if marks are stale."""
        symbol = context.get("symbol")
        mark_time = context.get("mark_timestamp")

        if not isinstance(symbol, str) or mark_time is None:
            return False, ""

        # Convert mark_time to datetime if needed
        if isinstance(mark_time, str):
            try:
                mark_time = datetime.fromisoformat(mark_time)
            except ValueError:
                return False, ""
        elif isinstance(mark_time, (int, float)):
            mark_time = datetime.fromtimestamp(mark_time)
        elif not isinstance(mark_time, datetime):
            return False, ""

        self.last_marks[symbol] = mark_time

        # Check staleness
        age_seconds = (datetime.now() - mark_time).total_seconds()
        if age_seconds > self.config.threshold:
            message = (
                f"Stale marks detected for {symbol}: "
                f"{age_seconds:.1f}s old (limit: {self.config.threshold}s)"
            )
            return True, message

        return False, ""


class ErrorRateGuard(RuntimeGuard):
    """Monitor error rates."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.error_times: list[datetime] = []

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if error rate exceeds threshold."""
        if context.get("error"):
            self.error_times.append(datetime.now())

        # Clean old errors outside window
        cutoff = datetime.now() - timedelta(seconds=self.config.window_seconds)
        self.error_times = [t for t in self.error_times if t > cutoff]

        # Check threshold
        error_count = len(self.error_times)
        if error_count > self.config.threshold:
            message = (
                f"High error rate: {error_count} errors in "
                f"{self.config.window_seconds}s (limit: {int(self.config.threshold)})"
            )
            return True, message

        return False, ""


class PositionStuckGuard(RuntimeGuard):
    """Monitor for positions that aren't being managed."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.position_times: dict[str, datetime] = {}

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if any positions are stuck."""
        positions = context.get("positions", {})
        if not isinstance(positions, Mapping):
            return False, ""

        for symbol, position in positions.items():
            if not isinstance(symbol, str) or not isinstance(position, Mapping):
                continue
            # Handle quantity fields from various payloads
            size_value = position.get(
                "quantity", position.get("size", position.get("contracts", 0))
            )
            try:
                size = float(size_value)
            except (TypeError, ValueError):
                continue
            if size != 0.0:
                if symbol not in self.position_times:
                    self.position_times[symbol] = datetime.now()
            else:
                self.position_times.pop(symbol, None)

        # Check for stuck positions
        stuck_positions: list[tuple[str, float]] = []
        for symbol, open_time in self.position_times.items():
            age_seconds = (datetime.now() - open_time).total_seconds()
            if age_seconds > self.config.threshold:
                stuck_positions.append((symbol, age_seconds))

        if stuck_positions:
            details = ", ".join([f"{sym}: {age:.0f}s" for sym, age in stuck_positions])
            message = f"Stuck positions detected: {details}"
            return True, message

        return False, ""


class DrawdownGuard(RuntimeGuard):
    """Monitor maximum drawdown."""

    def __init__(self, config: GuardConfig) -> None:
        super().__init__(config)
        self.peak_equity: Decimal = Decimal("0")
        self.current_drawdown: Decimal = Decimal("0")

    def _evaluate(self, context: dict[str, Any]) -> tuple[bool, str]:
        """Check if drawdown exceeds limit."""
        try:
            equity = Decimal(str(context.get("equity", 0)))
        except (TypeError, ValueError, DecimalException):
            return False, ""

        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > Decimal("0"):
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity * Decimal(100)
            threshold = Decimal(str(self.config.threshold))

            if self.current_drawdown > threshold:
                message = (
                    f"Maximum drawdown breached: {self.current_drawdown:.2f}% "
                    f"(limit: {self.config.threshold:.2f}%)"
                )
                return True, message

        return False, ""


class RuntimeGuardManager:
    """Manages all runtime guards and alert routing."""

    def __init__(self) -> None:
        self.guards: dict[str, RuntimeGuard] = {}
        self.alert_handlers: list[Callable[[Alert], None]] = []
        self.shutdown_callback: Callable[[], None] | None = None
        self.is_running: bool = False

    def add_guard(self, guard: RuntimeGuard) -> None:
        """Add a runtime guard."""
        self.guards[guard.config.name] = guard
        logger.info(f"Added runtime guard: {guard.config.name}")

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler."""
        self.alert_handlers.append(handler)

    def set_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """Set shutdown callback for auto-shutdown guards."""
        self.shutdown_callback = callback

    def check_all(self, context: dict[str, Any]) -> list[Alert]:
        """
        Check all guards with current context.

        Args:
            context: Runtime context containing current state

        Returns:
            List of triggered alerts
        """
        alerts: list[Alert] = []

        for guard in self.guards.values():
            alert = guard.check(context)
            if alert:
                alerts.append(alert)
                self._handle_alert(alert, guard)

        return alerts

    def _handle_alert(self, alert: Alert, guard: RuntimeGuard) -> None:
        """Handle an alert."""
        # Log alert
        log_method = getattr(logger, alert.severity.value, logger.info)
        log_method(f"[{alert.guard_name}] {alert.message}")

        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")

        # Check for auto-shutdown
        if guard.config.auto_shutdown and self.shutdown_callback:
            logger.critical(f"Auto-shutdown triggered by {alert.guard_name}")
            self.shutdown_callback()

    def get_status(self) -> dict[str, Any]:
        """Get status of all guards."""
        return {
            guard_name: {
                "status": guard.status.value,
                "breach_count": guard.breach_count,
                "last_check": guard.last_check.isoformat() if guard.last_check else None,
                "last_alert": guard.last_alert.isoformat() if guard.last_alert else None,
                "enabled": guard.config.enabled,
            }
            for guard_name, guard in self.guards.items()
        }

    def reset_guard(self, guard_name: str) -> None:
        """Reset a specific guard."""
        if guard_name in self.guards:
            self.guards[guard_name].reset()
            logger.info(f"Reset guard: {guard_name}")

    def reset_all(self) -> None:
        """Reset all guards."""
        for guard in self.guards.values():
            guard.reset()
        logger.info("Reset all runtime guards")


def create_default_guards(config: Mapping[str, Any]) -> RuntimeGuardManager:
    """
    Create default runtime guards from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Configured RuntimeGuardManager
    """
    manager = RuntimeGuardManager()

    risk_cfg = config.get("risk_management", {})
    if not isinstance(risk_cfg, Mapping):
        risk_cfg = {}
    circuit_cfg = risk_cfg.get("circuit_breakers", {})
    if not isinstance(circuit_cfg, Mapping):
        circuit_cfg = {}

    def _to_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    loss_limit = _to_float(risk_cfg.get("daily_loss_limit"), 100.0)
    manager.add_guard(
        DailyLossGuard(
            GuardConfig(
                name="daily_loss",
                threshold=loss_limit,
                severity=AlertSeverity.ERROR,
                auto_shutdown=True,
            )
        )
    )

    # Stale mark guard
    stale_seconds = _to_float(circuit_cfg.get("stale_mark_seconds"), 60.0)
    manager.add_guard(
        StaleMarkGuard(
            GuardConfig(name="stale_marks", threshold=stale_seconds, severity=AlertSeverity.WARNING)
        )
    )

    # Error rate guard
    error_threshold = _to_float(circuit_cfg.get("error_threshold"), 10.0)
    manager.add_guard(
        ErrorRateGuard(
            GuardConfig(
                name="error_rate",
                threshold=error_threshold,
                window_seconds=300,
                severity=AlertSeverity.ERROR,
                auto_shutdown=True,
            )
        )
    )

    # Position stuck guard
    manager.add_guard(
        PositionStuckGuard(
            GuardConfig(
                name="position_stuck",
                threshold=1800,  # 30 minutes
                severity=AlertSeverity.WARNING,
            )
        )
    )

    # Drawdown guard
    max_drawdown = _to_float(risk_cfg.get("max_drawdown_pct"), 5.0)
    manager.add_guard(
        DrawdownGuard(
            GuardConfig(
                name="max_drawdown",
                threshold=max_drawdown,
                severity=AlertSeverity.ERROR,
                auto_shutdown=True,
            )
        )
    )

    return manager


# Example alert handlers


def log_alert_handler(alert: Alert) -> None:
    """Simple log-based alert handler."""
    logger.info(f"ALERT: {json.dumps(alert.to_dict(), indent=2)}")


def slack_alert_handler(alert: Alert, webhook_url: str) -> None:
    """Send alerts to Slack."""
    import requests

    # Color based on severity
    colors = {
        AlertSeverity.DEBUG: "#808080",
        AlertSeverity.INFO: "#0000FF",
        AlertSeverity.WARNING: "#FFA500",
        AlertSeverity.ERROR: "#FF0000",
        AlertSeverity.CRITICAL: "#8B0000",
    }

    payload = {
        "attachments": [
            {
                "color": colors.get(alert.severity, "#808080"),
                "title": f"🚨 {alert.guard_name}",
                "text": alert.message,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Time", "value": alert.timestamp.strftime("%H:%M:%S"), "short": True},
                ],
                "footer": "Trading Bot Alert System",
                "ts": int(alert.timestamp.timestamp()),
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


def email_alert_handler(alert: Alert, smtp_config: Mapping[str, Any]) -> None:
    """Send alerts via email."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        return  # Only send email for serious alerts

    msg = MIMEMultipart()
    msg["From"] = str(smtp_config["from"])
    msg["To"] = str(smtp_config["to"])
    msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.guard_name}"

    body = f"""
Trading Bot Alert

Guard: {alert.guard_name}
Severity: {alert.severity.value}
Time: {alert.timestamp}

Message:
{alert.message}

Context:
{json.dumps(alert.context, indent=2)}
"""

    msg.attach(MIMEText(body, "plain"))

    try:
        host = str(smtp_config["host"])
        port = int(smtp_config["port"])
        with smtplib.SMTP(host, port) as server:
            if smtp_config.get("use_tls"):
                server.starttls()
            if smtp_config.get("username"):
                server.login(str(smtp_config["username"]), str(smtp_config["password"]))
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Failed to send email alert: {e}")
