"""Helper functions for orchestration components to use structured logging with correlation context."""

from __future__ import annotations

import logging
from collections.abc import Callable
from decimal import Decimal
from functools import wraps
from typing import Any, TypeVar, cast

from bot_v2.logging.correlation import (
    add_domain_field,
    correlation_context,
    get_log_context,
    order_context,
    symbol_context,
)
from bot_v2.logging.json_formatter import StructuredJSONFormatter
from bot_v2.utilities.logging_patterns import get_logger

F = TypeVar("F", bound=Callable[..., Any])


def get_orchestration_logger(
    name: str, component: str | None = None, enable_json: bool = True
) -> logging.Logger:
    """Get a logger configured for orchestration with correlation support.

    Args:
        name: Logger name
        component: Component name for context
        enable_json: Whether to enable JSON logging with correlation context

    Returns:
        Logger instance configured for orchestration
    """
    if enable_json:
        # Use the JSON logger for structured logging
        logger = logging.getLogger(f"bot_v2.json.{name}")

        # Ensure the logger has our structured formatter
        if not any(isinstance(h.formatter, StructuredJSONFormatter) for h in logger.handlers):
            # This logger should inherit handlers from the parent bot_v2.json logger
            # which is configured in setup.py with our StructuredJSONFormatter
            pass

        return logger
    else:
        # Fall back to the standard logger
        return get_logger(name, component=component).logger


def log_trading_operation(
    operation: str,
    symbol: str,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """Log a trading operation with correlation context.

    Args:
        operation: Description of the operation
        symbol: Trading symbol
        level: Log level
        **kwargs: Additional context fields
    """
    logger = get_orchestration_logger("trading_operations")

    # Add symbol to domain context
    add_domain_field("symbol", symbol)

    # Get the current log context (includes correlation ID)
    context = get_log_context()
    context.update(kwargs)

    logger.log(level, operation, extra=context)


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
    """Log an order event with correlation context.

    Args:
        event_type: Type of order event
        order_id: Order ID
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        price: Order price
        level: Log level
        **kwargs: Additional context fields
    """
    logger = get_orchestration_logger("order_events")

    # Add order context
    context = {
        "event_type": event_type,
        "order_id": order_id,
    }

    if symbol:
        context["symbol"] = symbol
    if side:
        context["side"] = side
    if quantity is not None:
        context["quantity"] = float(quantity)
    if price is not None:
        context["price"] = float(price)

    context.update(kwargs)

    # Get the current log context (includes correlation ID)
    log_context = get_log_context()
    log_context.update(context)

    logger.log(level, f"Order event: {event_type}", extra=log_context)


def log_strategy_decision(
    symbol: str,
    decision: str,
    reason: str | None = None,
    confidence: float | None = None,
    level: int = logging.INFO,
    **kwargs: Any,
) -> None:
    """Log a strategy decision with correlation context.

    Args:
        symbol: Trading symbol
        decision: Strategy decision
        reason: Decision reason
        confidence: Decision confidence (0-1)
        level: Log level
        **kwargs: Additional context fields
    """
    logger = get_orchestration_logger("strategy_decisions")

    # Add strategy context
    context = {
        "symbol": symbol,
        "decision": decision,
    }

    if reason:
        context["reason"] = reason
    if confidence is not None:
        context["confidence"] = confidence

    context.update(kwargs)

    # Get the current log context (includes correlation ID)
    log_context = get_log_context()
    log_context.update(context)

    logger.log(level, f"Strategy decision for {symbol}: {decision}", extra=log_context)


def log_execution_error(
    error: Exception,
    operation: str,
    symbol: str | None = None,
    order_id: str | None = None,
    level: int = logging.ERROR,
    **kwargs: Any,
) -> None:
    """Log an execution error with correlation context.

    Args:
        error: Exception that occurred
        operation: Operation that failed
        symbol: Trading symbol
        order_id: Order ID
        level: Log level
        **kwargs: Additional context fields
    """
    logger = get_orchestration_logger("execution_errors")

    # Add error context
    context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }

    if symbol:
        context["symbol"] = symbol
    if order_id:
        context["order_id"] = order_id

    context.update(kwargs)

    # Get the current log context (includes correlation ID)
    log_context = get_log_context()
    log_context.update(context)

    logger.log(level, f"Execution error in {operation}: {error}", extra=log_context, exc_info=True)


def log_risk_event(
    event_type: str,
    symbol: str | None = None,
    trigger_value: Any = None,
    threshold: Any = None,
    action: str | None = None,
    level: int = logging.WARNING,
    **kwargs: Any,
) -> None:
    """Log a risk management event with correlation context.

    Args:
        event_type: Type of risk event
        symbol: Trading symbol
        trigger_value: Value that triggered the event
        threshold: Threshold that was exceeded
        action: Action taken
        level: Log level
        **kwargs: Additional context fields
    """
    logger = get_orchestration_logger("risk_events")

    # Add risk context
    context = {
        "event_type": event_type,
    }

    if symbol:
        context["symbol"] = symbol
    if trigger_value is not None:
        context["trigger_value"] = str(trigger_value)
    if threshold is not None:
        context["threshold"] = str(threshold)
    if action:
        context["action"] = action

    context.update(kwargs)

    # Get the current log context (includes correlation ID)
    log_context = get_log_context()
    log_context.update(context)

    logger.log(level, f"Risk event: {event_type}", extra=log_context)


def log_market_data_update(
    symbol: str,
    price: Decimal,
    volume: Decimal | None = None,
    timestamp: float | None = None,
    level: int = logging.DEBUG,
    **kwargs: Any,
) -> None:
    """Log a market data update with correlation context.

    Args:
        symbol: Market symbol
        price: Current price
        volume: Current volume
        timestamp: Update timestamp
        level: Log level
        **kwargs: Additional context fields
    """
    logger = get_orchestration_logger("market_data")

    # Add market data context
    context = {
        "symbol": symbol,
        "price": float(price),
    }

    if volume is not None:
        context["volume"] = float(volume)
    if timestamp is not None:
        context["timestamp"] = timestamp

    context.update(kwargs)

    # Get the current log context (includes correlation ID)
    log_context = get_log_context()
    log_context.update(context)

    logger.log(level, f"Market data update: {symbol}", extra=log_context)


# Context manager decorators for common orchestration patterns


def with_trading_context(operation: str) -> Callable[[F], F]:
    """Decorator to add trading context to a function.

    Args:
        operation: Description of the operation

    Returns:
        Decorated function with correlation context
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with correlation_context(operation=operation):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def with_symbol_context(symbol: str) -> Callable[[F], F]:
    """Decorator to add symbol context to a function.

    Args:
        symbol: Trading symbol

    Returns:
        Decorated function with symbol context
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with symbol_context(symbol):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def with_order_context(order_id: str, symbol: str | None = None) -> Callable[[F], F]:
    """Decorator to add order context to a function.

    Args:
        order_id: Order ID
        symbol: Trading symbol

    Returns:
        Decorated function with order context
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with order_context(order_id, symbol):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
