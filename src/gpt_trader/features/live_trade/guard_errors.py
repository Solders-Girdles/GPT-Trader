"""
Risk guard error hierarchy and alert dispatch system.

This module defines the exception hierarchy for runtime risk guards and provides
centralized error recording and alerting infrastructure.

Error Hierarchy
---------------
All guard errors inherit from ``GuardError`` which provides:

- ``category``: Classification string (limit, data, action, telemetry, etc.)
- ``recoverable``: Whether the error is transient and execution can continue

Error Types:

- **RiskLimitExceeded**: Hard risk limit breach (non-recoverable)
- **RiskGuardError**: Generic guard failure (non-recoverable)
- **RiskGuardActionError**: Failed remediation action (non-recoverable)
- **RiskGuardComputationError**: Calculation failure (non-recoverable)
- **RiskGuardDataCorrupt**: Invalid/corrupted data (non-recoverable)
- **RiskGuardDataUnavailable**: Temporary data fetch failure (recoverable)
- **RiskGuardTelemetryError**: Metrics/logging failure (recoverable)

Global Alert System
-------------------
The module maintains a global ``_alert_system`` dispatcher for centralized alerting.

**Important**: Call ``configure_guard_alert_system(dispatcher)`` at application startup
to inject a custom alert dispatcher (e.g., Slack, PagerDuty integration).

If not configured, a no-op ``GuardAlertDispatcher`` is used.

Thread Safety
-------------
The global ``_alert_system`` is not thread-safe. Configure it once at startup
before spawning threads. The alert system should be stateless for concurrent access.

Testing
-------
Use ``configure_guard_alert_system(mock_dispatcher)`` in tests to capture alerts.
Reset to ``None`` after tests to restore default behavior.

Example::

    # Production setup
    from gpt_trader.monitoring import SlackAlertDispatcher
    configure_guard_alert_system(SlackAlertDispatcher())

    # Test setup
    mock = Mock(spec=GuardAlertDispatcher)
    configure_guard_alert_system(mock)
"""

from __future__ import annotations

import logging
from typing import Any

from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger("gpt_trader.features.live_trade.guard_errors", component="guards")


class GuardError(Exception):
    category: str = "generic"
    recoverable: bool = False

    def __init__(self, guard_name: str, message: str, details: dict | None = None):
        self.guard_name = guard_name
        self.message = message
        self.details = details or {}
        super().__init__(f"[{guard_name}] {message}")


class RiskLimitExceeded(GuardError):
    category = "limit"
    recoverable = False


class RiskGuardError(GuardError):
    category = "generic"
    recoverable = False


class RiskGuardActionError(GuardError):
    category = "action"
    recoverable = False


class RiskGuardComputationError(GuardError):
    category = "computation"
    recoverable = False


class RiskGuardDataCorrupt(GuardError):
    category = "data"
    recoverable = False


class RiskGuardDataUnavailable(GuardError):
    category = "data"
    recoverable = True


class RiskGuardTelemetryError(GuardError):
    category = "telemetry"
    recoverable = True


class GuardAlertDispatcher:
    def trigger_alert(
        self, severity: AlertSeverity, category: str, message: str, metadata: dict | None = None
    ) -> None:
        pass


_alert_system: Any = None


def configure_guard_alert_system(dispatcher: Any) -> None:
    global _alert_system
    _alert_system = dispatcher


def _get_alert_system() -> Any:
    global _alert_system
    if _alert_system is None or not hasattr(_alert_system, "trigger_alert"):
        _alert_system = GuardAlertDispatcher()
    return _alert_system


def record_counter(name: str, increment: int = 1) -> None:
    # Placeholder for metrics
    pass


def record_guard_failure(error: GuardError) -> None:
    category = getattr(error, "category", "unknown")
    recoverable = getattr(error, "recoverable", False)
    guard_slug = error.guard_name.lower().replace(" ", "_")

    # Metrics
    metric_type = "recoverable_failures" if recoverable else "critical_failures"
    record_counter(f"risk.guards.{guard_slug}.{metric_type}", 1)
    record_counter(f"risk.guards.{guard_slug}.{category}", 1)

    # Logging
    level = logging.WARNING if recoverable else logging.ERROR
    logger.log(
        level,
        error.message,
        guard_failure={
            "guard": error.guard_name,
            "category": category,
            "recoverable": recoverable,
            "details": error.details,
        },
    )

    # Alerting
    severity = AlertSeverity.WARNING if recoverable else AlertSeverity.CRITICAL
    dispatcher = _get_alert_system()
    dispatcher.trigger_alert(
        severity, f"risk_guard.{error.guard_name}", error.message, error.details
    )


def record_guard_success(guard_name: str) -> None:
    guard_slug = guard_name.lower().replace(" ", "_")
    record_counter(f"risk.guards.{guard_slug}.success", 1)
