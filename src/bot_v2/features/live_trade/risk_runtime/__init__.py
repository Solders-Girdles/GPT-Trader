"""Runtime guard helpers for live risk management."""

from __future__ import annotations

from bot_v2.features.live_trade.risk.runtime_monitoring import RuntimeMonitor

from .circuit_breakers import (
    CircuitBreakerAction,
    CircuitBreakerOutcome,
    CircuitBreakerRule,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
    check_volatility_circuit_breaker,
)
from .guards import check_correlation_risk, check_mark_staleness
from .metrics import append_risk_metrics
from .types import AnyLogger, LogEventFn

__all__ = [
    "CircuitBreakerAction",
    "CircuitBreakerOutcome",
    "CircuitBreakerRule",
    "CircuitBreakerSnapshot",
    "CircuitBreakerState",
    "AnyLogger",
    "LogEventFn",
    "append_risk_metrics",
    "check_correlation_risk",
    "check_mark_staleness",
    "check_volatility_circuit_breaker",
    "RuntimeMonitor",
]
