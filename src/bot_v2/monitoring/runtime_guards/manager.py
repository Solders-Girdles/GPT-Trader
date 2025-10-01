"""
Runtime Guard Manager and Alert Handlers.

Manages all runtime guards and alert routing:
- RuntimeGuardManager: Orchestrates multiple guards
- create_default_guards(): Factory for standard guard configuration
- Example alert handlers: log, Slack, email
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from bot_v2.monitoring.alerts import AlertSeverity
from bot_v2.monitoring.runtime_guards.base import Alert, GuardConfig, RuntimeGuard
from bot_v2.monitoring.runtime_guards.builtins import (
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)


logger = logging.getLogger(__name__)


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
