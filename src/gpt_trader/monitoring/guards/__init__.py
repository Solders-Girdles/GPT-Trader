"""Runtime guard utilities for monitoring."""

from __future__ import annotations

from .base import Alert, GuardConfig, GuardStatus, RuntimeGuard
from .builtins import (
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    LauncherStarvationGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)
from .manager import (
    RuntimeGuardManager,
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
    "DailyLossGuard",
    "DrawdownGuard",
    "ErrorRateGuard",
    "LauncherStarvationGuard",
    "PositionStuckGuard",
    "StaleMarkGuard",
    "RuntimeGuardManager",
    "create_default_runtime_guard_manager",
    "log_alert_handler",
    "slack_alert_handler",
    "email_alert_handler",
]
