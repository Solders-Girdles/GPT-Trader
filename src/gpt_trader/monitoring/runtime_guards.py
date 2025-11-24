"""Backwards-compatible shim for runtime guard utilities."""

from __future__ import annotations

from gpt_trader.monitoring.alert_types import AlertSeverity

from .guards import (
    Alert,
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    GuardConfig,
    GuardStatus,
    PositionStuckGuard,
    RuntimeGuard,
    RuntimeGuardManager,
    StaleMarkGuard,
    create_default_runtime_guard_manager,
    email_alert_handler,
    log_alert_handler,
    slack_alert_handler,
)

__all__ = [
    "Alert",
    "GuardConfig",
    "GuardStatus",
    "RuntimeGuard",
    "RuntimeGuardManager",
    "DailyLossGuard",
    "StaleMarkGuard",
    "ErrorRateGuard",
    "PositionStuckGuard",
    "DrawdownGuard",
    "create_default_runtime_guard_manager",
    "log_alert_handler",
    "slack_alert_handler",
    "email_alert_handler",
    "AlertSeverity",
]
