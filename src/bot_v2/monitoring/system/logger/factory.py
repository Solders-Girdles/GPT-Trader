"""Factory helpers and global convenience functions for production logger."""

from __future__ import annotations

from typing import Any

from bot_v2.orchestration.runtime_settings import RuntimeSettings

from .levels import LogLevel
from .production import ProductionLogger

_LOGGER: ProductionLogger | None = None


def get_logger(
    service_name: str = "bot_v2",
    enable_console: bool = True,
    *,
    settings: RuntimeSettings | None = None,
) -> ProductionLogger:
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = ProductionLogger(service_name, enable_console, settings=settings)
    return _LOGGER


def log_event(
    level: LogLevel,
    event_type: str,
    message: str,
    component: str | None = None,
    **kwargs: Any,
) -> None:
    get_logger().log_event(level, event_type, message, component, **kwargs)


def log_trade(
    action: str,
    symbol: str,
    quantity: float,
    price: float,
    strategy: str,
    *,
    success: bool = True,
    execution_time_ms: float | None = None,
    **kwargs: Any,
) -> None:
    get_logger().log_trade(
        action=action,
        symbol=symbol,
        quantity=quantity,
        price=price,
        strategy=strategy,
        success=success,
        execution_time_ms=execution_time_ms,
        **kwargs,
    )


def log_ml_prediction(
    model_name: str,
    prediction: Any,
    *,
    confidence: float | None = None,
    input_features: dict[str, Any] | None = None,
    inference_time_ms: float | None = None,
    **kwargs: Any,
) -> None:
    get_logger().log_ml_prediction(
        model_name=model_name,
        prediction=prediction,
        confidence=confidence,
        input_features=input_features,
        inference_time_ms=inference_time_ms,
        **kwargs,
    )


def log_performance(
    operation: str,
    duration_ms: float,
    *,
    success: bool = True,
    **kwargs: Any,
) -> None:
    get_logger().log_performance(
        operation=operation,
        duration_ms=duration_ms,
        success=success,
        **kwargs,
    )


def log_error(error: Exception, context: str | None = None, **kwargs: Any) -> None:
    get_logger().log_error(error, context, **kwargs)


def set_correlation_id(correlation_id: str | None = None) -> None:
    get_logger().set_correlation_id(correlation_id)


def get_correlation_id() -> str:
    return get_logger().get_correlation_id()


__all__ = [
    "get_logger",
    "log_event",
    "log_trade",
    "log_ml_prediction",
    "log_performance",
    "log_error",
    "set_correlation_id",
    "get_correlation_id",
]
