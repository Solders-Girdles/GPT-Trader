"""Guard manager orchestration and alert helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from gpt_trader.config.constants import WEBHOOK_TIMEOUT
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.metrics_collector import record_counter
from gpt_trader.utilities.logging_patterns import get_logger

from .base import Alert, GuardConfig, RuntimeGuard
from .builtins import (
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)

logger = get_logger(__name__, component="runtime_guard_manager")


class RuntimeGuardManager:
    """Manages all runtime guards and alert routing."""

    def __init__(self) -> None:
        self.guards: dict[str, RuntimeGuard] = {}
        self.alert_handlers: list[Callable[[Alert], None]] = []
        self.shutdown_callback: Callable[[], None] | None = None
        self.is_running: bool = False

    def add_guard(self, guard: RuntimeGuard) -> None:
        self.guards[guard.config.name] = guard
        logger.info(
            "Added runtime guard",
            operation="guard_manager",
            stage="add_guard",
            guard_name=guard.config.name,
        )

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        self.alert_handlers.append(handler)

    def set_shutdown_callback(self, callback: Callable[[], None]) -> None:
        self.shutdown_callback = callback

    def check_all(self, context: dict[str, Any]) -> list[Alert]:
        alerts: list[Alert] = []
        for guard in self.guards.values():
            if not guard.config.enabled:
                continue
            try:
                alert = guard.check(context)
            except Exception as exc:
                logger.error(
                    "Guard check failed",
                    operation="guard_manager",
                    stage="check_guard",
                    guard_name=guard.config.name,
                    error=str(exc),
                    exc_info=True,
                )
                record_counter(
                    "gpt_trader_guard_checks_total",
                    labels={"guard": guard.config.name, "result": "error"},
                )
                continue
            record_counter(
                "gpt_trader_guard_checks_total",
                labels={"guard": guard.config.name, "result": "success"},
            )
            if alert:
                alerts.append(alert)
                self._handle_alert(alert, guard)
        return alerts

    def _handle_alert(self, alert: Alert, guard: RuntimeGuard) -> None:
        log_method = getattr(logger, alert.severity.value, logger.info)
        log_method(
            f"Runtime guard alert: {alert.guard_name}",
            operation="guard_manager",
            stage="handle_alert",
            guard_name=alert.guard_name,
            severity=alert.severity.value,
            alert_message=alert.message,
        )

        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Alert handler error",
                    operation="guard_manager",
                    stage="handle_alert",
                    guard_name=alert.guard_name,
                    handler=repr(handler),
                    error=str(exc),
                    exc_info=True,
                )

        if guard.config.auto_shutdown and self.shutdown_callback:
            logger.critical(
                "Auto-shutdown triggered",
                operation="guard_manager",
                stage="auto_shutdown",
                guard_name=alert.guard_name,
            )
            self.shutdown_callback()

    def get_status(self) -> dict[str, Any]:
        return {
            guard_name: {
                "status": guard.status.value,
                "breach_count": guard.breach_count,
                "last_check": guard.last_check.isoformat(),
                "last_alert": guard.last_alert.isoformat() if guard.last_alert else None,
            }
            for guard_name, guard in self.guards.items()
        }

    def reset(self) -> None:
        self.guards.clear()
        self.alert_handlers.clear()
        self.shutdown_callback = None
        self.is_running = False


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def create_default_runtime_guard_manager(config: Mapping[str, Any]) -> RuntimeGuardManager:
    manager = RuntimeGuardManager()

    circuit_config = config.get("circuit_breakers", {})
    risk_config = config.get("risk_management", {})

    daily_loss_limit = _to_float(circuit_config.get("daily_loss_limit"), 500.0)
    manager.add_guard(
        DailyLossGuard(
            GuardConfig(
                name="daily_loss",
                threshold=daily_loss_limit,
                severity=AlertSeverity.CRITICAL,
                auto_shutdown=True,
            )
        )
    )

    stale_mark_seconds = _to_float(circuit_config.get("stale_mark_seconds"), 15.0)
    manager.add_guard(
        StaleMarkGuard(
            GuardConfig(
                name="stale_mark",
                threshold=stale_mark_seconds,
                severity=AlertSeverity.ERROR,
            )
        )
    )

    error_threshold = _to_float(circuit_config.get("error_threshold"), 10.0)
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

    manager.add_guard(
        PositionStuckGuard(
            GuardConfig(
                name="position_stuck",
                threshold=_to_float(circuit_config.get("position_timeout_seconds"), 1800),
                severity=AlertSeverity.WARNING,
            )
        )
    )

    max_drawdown = _to_float(risk_config.get("max_drawdown_pct"), 5.0)
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


def log_alert_handler(alert: Alert) -> None:
    logger.info(
        "Runtime guard alert dispatched",
        operation="guard_alert",
        stage="log_handler",
        guard_name=alert.guard_name,
        severity=alert.severity.value,
        payload=json.dumps(alert.to_dict(), indent=2),
    )


def slack_alert_handler(alert: Alert, webhook_url: str) -> None:  # pragma: no cover - IO
    import requests

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
                    {
                        "title": "Time",
                        "value": alert.created_at.strftime("%H:%M:%S"),
                        "short": True,
                    },
                ],
                "footer": "Trading Bot Alert System",
                "ts": int(alert.created_at.timestamp()),
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=WEBHOOK_TIMEOUT)
        response.raise_for_status()
    except Exception as exc:
        logger.error(
            "Failed to send Slack alert",
            operation="guard_alert",
            stage="slack_handler",
            guard_name=alert.guard_name,
            severity=alert.severity.value,
            error=str(exc),
            exc_info=True,
        )


def email_alert_handler(
    alert: Alert, smtp_config: Mapping[str, Any]
) -> None:  # pragma: no cover - IO
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        return

    msg = MIMEMultipart()
    msg["From"] = str(smtp_config["from"])
    msg["To"] = str(smtp_config["to"])
    msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.guard_name}"

    body = f"""
Trading Bot Alert

Guard: {alert.guard_name}
Severity: {alert.severity.value}
Time: {alert.created_at}

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
    except Exception as exc:
        logger.error(
            "Failed to send email alert",
            operation="guard_alert",
            stage="email_handler",
            guard_name=alert.guard_name,
            severity=alert.severity.value,
            error=str(exc),
            exc_info=True,
        )


__all__ = [
    "RuntimeGuardManager",
    "create_default_runtime_guard_manager",
    "log_alert_handler",
    "slack_alert_handler",
    "email_alert_handler",
]
