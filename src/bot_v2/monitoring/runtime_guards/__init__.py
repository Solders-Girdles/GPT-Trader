"""
Runtime Guard Alerting System.

Monitors critical runtime conditions and triggers alerts when thresholds are breached.
Implements circuit breakers for daily loss limits, stale marks, and error rates.

This package provides:
- Base classes for creating custom guards (base module)
- Built-in guard implementations (builtins module)
- Guard management and alert routing (manager module)
"""

from bot_v2.monitoring.alerts import AlertSeverity
from bot_v2.monitoring.runtime_guards.base import (
    Alert,
    GuardConfig,
    GuardStatus,
    RuntimeGuard,
)
from bot_v2.monitoring.runtime_guards.builtins import (
    DailyLossGuard,
    DrawdownGuard,
    ErrorRateGuard,
    PositionStuckGuard,
    StaleMarkGuard,
)
from bot_v2.monitoring.runtime_guards.manager import (
    RuntimeGuardManager,
    create_default_guards,
    email_alert_handler,
    log_alert_handler,
    slack_alert_handler,
)

__all__ = [
    # Base types and classes
    "Alert",
    "AlertSeverity",
    "GuardConfig",
    "GuardStatus",
    "RuntimeGuard",
    # Built-in guard implementations
    "DailyLossGuard",
    "DrawdownGuard",
    "ErrorRateGuard",
    "PositionStuckGuard",
    "StaleMarkGuard",
    # Manager and factory
    "RuntimeGuardManager",
    "create_default_guards",
    # Alert handlers
    "email_alert_handler",
    "log_alert_handler",
    "slack_alert_handler",
]
