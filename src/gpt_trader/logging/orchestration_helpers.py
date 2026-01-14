"""Deprecated orchestration logging helpers.

Use runtime helpers instead.
"""

from __future__ import annotations

import importlib
import logging
import warnings
from collections.abc import Callable
from decimal import Decimal
from typing import Any, TypeVar, cast


def _runtime_helpers() -> Any:
    return importlib.import_module("gpt_trader.logging.runtime_helpers")


F = TypeVar("F", bound=Callable[..., Any])


def _warn_deprecated() -> None:
    warnings.warn(
        "orchestration_helpers is deprecated; use runtime_helpers instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def get_orchestration_logger(
    name: str, component: str | None = None, enable_json: bool = True
) -> logging.Logger:
    """Deprecated wrapper for get_runtime_logger."""
    _warn_deprecated()
    return cast(
        logging.Logger,
        _runtime_helpers().get_runtime_logger(name, component=component, enable_json=enable_json),
    )


def log_trading_operation(
    operation: str,
    symbol: str,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    _warn_deprecated()
    _runtime_helpers().log_trading_operation(
        operation=operation,
        symbol=symbol,
        level=level,
        **kwargs,
    )


def log_order_event(
    event_type: str,
    order_id: str,
    symbol: str | None = None,
    side: str | None = None,
    quantity: Decimal | None = None,
    price: Decimal | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    _warn_deprecated()
    _runtime_helpers().log_order_event(
        event_type=event_type,
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        level=level,
        **kwargs,
    )


def log_strategy_decision(
    symbol: str,
    decision: str,
    reason: str | None = None,
    confidence: float | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    _warn_deprecated()
    _runtime_helpers().log_strategy_decision(
        symbol=symbol,
        decision=decision,
        reason=reason,
        confidence=confidence,
        level=level,
        **kwargs,
    )


def log_execution_error(
    error: Exception,
    operation: str,
    symbol: str | None = None,
    order_id: str | None = None,
    level: int = logging.ERROR,
    **kwargs: Any,
) -> None:
    _warn_deprecated()
    _runtime_helpers().log_execution_error(
        error=error,
        operation=operation,
        symbol=symbol,
        order_id=order_id,
        level=level,
        **kwargs,
    )


def log_risk_event(
    event_type: str,
    symbol: str | None = None,
    trigger_value: Any = None,
    threshold: Any = None,
    action: str | None = None,
    level: int = logging.WARNING,
    **kwargs: Any,
) -> None:
    _warn_deprecated()
    _runtime_helpers().log_risk_event(
        event_type=event_type,
        symbol=symbol,
        trigger_value=trigger_value,
        threshold=threshold,
        action=action,
        level=level,
        **kwargs,
    )


def log_market_data_update(
    symbol: str,
    price: Decimal,
    volume: Decimal | None = None,
    timestamp: float | None = None,
    level: int = logging.DEBUG,
    **kwargs: Any,
) -> None:
    _warn_deprecated()
    _runtime_helpers().log_market_data_update(
        symbol=symbol,
        price=price,
        volume=volume,
        timestamp=timestamp,
        level=level,
        **kwargs,
    )


def with_trading_context(operation: str) -> Callable[[F], F]:
    _warn_deprecated()
    return cast(Callable[[F], F], _runtime_helpers().with_trading_context(operation))


def with_symbol_context(symbol: str) -> Callable[[F], F]:
    _warn_deprecated()
    return cast(Callable[[F], F], _runtime_helpers().with_symbol_context(symbol))


def with_order_context(order_id: str, symbol: str | None = None) -> Callable[[F], F]:
    _warn_deprecated()
    return cast(Callable[[F], F], _runtime_helpers().with_order_context(order_id, symbol))


__all__ = [
    "get_orchestration_logger",
    "log_trading_operation",
    "log_order_event",
    "log_strategy_decision",
    "log_execution_error",
    "log_risk_event",
    "log_market_data_update",
    "with_trading_context",
    "with_symbol_context",
    "with_order_context",
]
