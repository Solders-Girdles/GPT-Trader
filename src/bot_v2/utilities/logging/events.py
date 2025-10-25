"""Logging helpers for common trading events."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from .logger import StructuredLogger, UnifiedLogger


def _coerce_logger(
    logger: StructuredLogger | logging.Logger | None,
    default_name: str,
    *,
    component: str | None = None,
) -> StructuredLogger:
    if logger is None:
        return UnifiedLogger(default_name, component=component)
    if isinstance(logger, logging.Logger):
        return UnifiedLogger(logger.name, component=component)
    return logger


def log_trade_event(
    event_type: str,
    symbol: str,
    side: str | None = None,
    quantity: Decimal | None = None,
    price: Decimal | None = None,
    order_id: str | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
) -> None:
    logger = _coerce_logger(logger, "trading")
    context: dict[str, Any] = {
        "operation": "trade_event",
        "symbol": symbol,
        "event_type": event_type,
    }
    if side:
        context["side"] = side
    if quantity is not None:
        context["quantity"] = quantity
    if price is not None:
        context["price"] = price
    if order_id:
        context["order_id"] = order_id
    logger.info(f"Trade event: {event_type}", **context)


def log_position_update(
    symbol: str,
    position_size: Decimal,
    unrealized_pnl: Decimal | None = None,
    equity: Decimal | None = None,
    leverage: float | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
) -> None:
    logger = _coerce_logger(logger, "position")
    context: dict[str, Any] = {
        "operation": "position_update",
        "symbol": symbol,
        "position_size": position_size,
    }
    if unrealized_pnl is not None:
        context["pnl"] = unrealized_pnl
    if equity is not None:
        context["equity"] = equity
    if leverage is not None:
        context["leverage"] = leverage
    logger.info("Position update", **context)


def log_error_with_context(
    error: Exception,
    *,
    operation: str,
    component: str | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
    **context: Any,
) -> None:
    logger = _coerce_logger(logger, "errors", component=component)
    error_context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **context,
    }
    if component:
        error_context["component"] = component
    logger.error(f"Error in {operation}: {error}", **error_context)


def log_configuration_change(
    config_key: str,
    old_value: Any | None,
    new_value: Any,
    component: str | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
) -> None:
    logger = _coerce_logger(logger, "config", component=component)
    context: dict[str, Any] = {
        "operation": "config_change",
        "config_key": config_key,
        "old_value": str(old_value) if old_value is not None else "None",
        "new_value": str(new_value),
    }
    if component:
        context["component"] = component
    logger.info(f"Configuration changed: {config_key}", **context)


def log_market_data_update(
    symbol: str,
    price: Decimal,
    volume: Decimal | None = None,
    timestamp: float | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
) -> None:
    logger = _coerce_logger(logger, "market_data")
    context: dict[str, Any] = {
        "operation": "market_data_update",
        "symbol": symbol,
        "price": price,
    }
    if volume is not None:
        context["volume"] = volume
    if timestamp is not None:
        context["timestamp"] = timestamp
    logger.debug(f"Market data update: {symbol}", **context)


def log_system_health(
    status: str,
    component: str | None = None,
    metrics: dict[str, Any] | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
) -> None:
    logger = _coerce_logger(logger, "health", component=component)
    context: dict[str, Any] = {
        "operation": "health_check",
        "status": status,
    }
    if component:
        context["component"] = component
    if metrics:
        context.update(metrics)
    level = logging.INFO if status == "healthy" else logging.WARNING
    logger.log(level, f"Health status: {status}", **context)


__all__ = [
    "log_trade_event",
    "log_position_update",
    "log_error_with_context",
    "log_configuration_change",
    "log_market_data_update",
    "log_system_health",
]
