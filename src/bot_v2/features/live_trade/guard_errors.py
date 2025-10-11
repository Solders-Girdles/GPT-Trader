"""Exception types and helpers for runtime guard failures."""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from bot_v2.monitoring.alert_types import AlertSeverity
from bot_v2.monitoring.interfaces import MonitorConfig
from bot_v2.monitoring.metrics_collector import record_counter
from bot_v2.monitoring.system.alerting import AlertManager

logger = logging.getLogger(__name__)


class GuardAlertDispatcher:
    """Thin wrapper over the monitoring alert manager for guard failures."""

    def __init__(self) -> None:
        self._manager = AlertManager(MonitorConfig(enable_notifications=False))

    def trigger_alert(
        self,
        severity: AlertSeverity,
        category: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._manager.create_alert(
            severity=severity,
            component=category,
            message=message,
            details=dict(metadata or {}),
        )


_alert_system: Any | None = None


def configure_guard_alert_system(system: Any | None) -> None:
    """Override the alerting system used for guard failures (test hook)."""

    global _alert_system
    _alert_system = system  # type: ignore[assignment]


def _get_alert_system() -> GuardAlertDispatcher:
    global _alert_system
    if _alert_system is None:
        _alert_system = GuardAlertDispatcher()
    if not hasattr(_alert_system, "trigger_alert"):
        _alert_system = GuardAlertDispatcher()
    return _alert_system  # type: ignore[return-value]


@dataclass
class GuardFailureContext:
    """Structured context for a guard failure."""

    guard: str
    category: str
    recoverable: bool
    message: str
    details: MutableMapping[str, Any]

    def as_log_args(self) -> Mapping[str, Any]:
        payload: MutableMapping[str, Any] = {
            "guard": self.guard,
            "category": self.category,
            "recoverable": self.recoverable,
        }
        if self.details:
            payload["details"] = dict(self.details)
        return payload


class RiskGuardError(RuntimeError):
    """Base runtime guard error."""

    category = "unknown"
    recoverable = False

    def __init__(
        self,
        guard: str,
        message: str,
        *,
        details: MutableMapping[str, Any] | None = None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.guard = guard
        self.details: MutableMapping[str, Any] = details or {}
        if original is not None:
            self.details.setdefault("original", repr(original))
            self.details.setdefault("original_type", type(original).__name__)
        self.original = original

    @property
    def failure(self) -> GuardFailureContext:
        return GuardFailureContext(
            guard=self.guard,
            category=getattr(self, "category", "unknown"),
            recoverable=getattr(self, "recoverable", False),
            message=str(self),
            details=self.details,
        )


class RiskGuardTelemetryError(RiskGuardError):
    category = "telemetry"
    recoverable = True


class RiskGuardDataUnavailable(RiskGuardError):
    category = "data_unavailable"
    recoverable = True


class RiskGuardDataCorrupt(RiskGuardError):
    category = "data_corrupt"
    recoverable = False


class RiskGuardComputationError(RiskGuardError):
    category = "computation"
    recoverable = False


class RiskGuardActionError(RiskGuardError):
    category = "action"
    recoverable = False


class RuntimeGuardCriticalError(RiskGuardError):
    category = "critical"
    recoverable = False


def _metric_key(guard: str, suffix: str) -> str:
    guard_key = guard.lower().replace(" ", "_").replace("/", "_")
    return f"risk.guards.{guard_key}.{suffix}"


def record_guard_failure(error: RiskGuardError) -> None:
    """Emit counters/logging for a guard failure."""
    failure = error.failure

    # Increment counters for observability
    suffix = "recoverable_failures" if failure.recoverable else "critical_failures"
    record_counter(_metric_key(failure.guard, suffix))
    record_counter(_metric_key(failure.guard, failure.category))

    log_method = logger.warning if failure.recoverable else logger.error
    log_method(
        failure.message,
        extra={"guard_failure": failure.as_log_args()},
    )

    severity = AlertSeverity.WARNING if failure.recoverable else AlertSeverity.CRITICAL
    try:
        _get_alert_system().trigger_alert(
            severity,
            category=f"risk_guard.{failure.guard}",
            message=failure.message,
            metadata=dict(failure.details),
        )
    except Exception as exc:  # pragma: no cover - alerting is best-effort
        logger.debug("Guard alert dispatch failed: %s", exc, exc_info=True)


def record_guard_success(guard: str) -> None:
    """Increment success counter for a guard execution."""
    record_counter(_metric_key(guard, "success"))
