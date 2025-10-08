"""Standardized logging patterns for consistent log formatting and structure."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Generator

# Standardized log field names
LOG_FIELDS = {
    "operation": "operation",
    "component": "component", 
    "symbol": "symbol",
    "side": "side",
    "quantity": "quantity",
    "price": "price",
    "order_id": "order_id",
    "position_size": "position_size",
    "pnl": "pnl",
    "equity": "equity",
    "leverage": "leverage",
    "duration_ms": "duration_ms",
    "error_type": "error_type",
    "status": "status",
}


class StructuredLogger:
    """Logger with standardized structured logging patterns."""
    
    def __init__(self, name: str, component: str | None = None) -> None:
        """Initialize structured logger.
        
        Args:
            name: Logger name
            component: Component name for log context
        """
        self.logger = logging.getLogger(name)
        self.component = component
        
    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format message with structured context.
        
        Args:
            message: Log message
            **kwargs: Additional context fields
            
        Returns:
            Formatted message with context
        """
        context_parts = []
        
        # Add component if specified
        if self.component:
            context_parts.append(f"component={self.component}")
            
        # Add structured fields
        for key, value in kwargs.items():
            if key in LOG_FIELDS.values():
                # Format special types
                if isinstance(value, Decimal):
                    formatted_value = f"{value:.8f}".rstrip("0").rstrip(".")
                elif isinstance(value, (int, float)):
                    formatted_value = f"{value}"
                else:
                    formatted_value = str(value)
                    
                context_parts.append(f"{key}={formatted_value}")
                
        if context_parts:
            context_str = " | " + " ".join(context_parts)
            return f"{message}{context_str}"
        else:
            return message
            
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with structured context."""
        self.logger.info(self._format_message(message, **kwargs))
        
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with structured context."""
        self.logger.warning(self._format_message(message, **kwargs))
        
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with structured context."""
        self.logger.error(self._format_message(message, **kwargs))
        
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with structured context."""
        self.logger.debug(self._format_message(message, **kwargs))
        
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with structured context."""
        self.logger.critical(self._format_message(message, **kwargs))


@contextmanager
def log_operation(
    operation: str,
    logger: StructuredLogger | logging.Logger | None = None,
    level: int = logging.INFO,
    **context: Any,
) -> Generator[None, None, None]:
    """Context manager for logging operation start/end with timing.
    
    Args:
        operation: Description of the operation
        logger: Logger instance (creates StructuredLogger if None)
        level: Log level for operation messages
        **context: Additional context to include in logs
        
    Yields:
        None
    """
    if logger is None:
        logger = StructuredLogger("operation")
    elif isinstance(logger, logging.Logger):
        logger = StructuredLogger(logger.name)
        
    start_time = time.time()
    
    # Log operation start
    log_message = f"Started {operation}"
    if context:
        logger.log(level, log_message, operation=operation, **context)
    else:
        logger.log(level, log_message, operation=operation)
        
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        
        # Log operation completion
        log_message = f"Completed {operation}"
        logger.log(
            level,
            log_message,
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **context,
        )


def log_trade_event(
    event_type: str,
    symbol: str,
    side: str | None = None,
    quantity: Decimal | None = None,
    price: Decimal | None = None,
    order_id: str | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
) -> None:
    """Log a trading event with standardized format.
    
    Args:
        event_type: Type of trade event (e.g., "order_filled", "position_opened")
        symbol: Trading symbol
        side: Order side (buy/sell)
        quantity: Order quantity
        price: Order price
        order_id: Order ID
        logger: Logger instance
    """
    if logger is None:
        logger = StructuredLogger("trading")
    elif isinstance(logger, logging.Logger):
        logger = StructuredLogger(logger.name)
        
    context = {
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
    """Log a position update with standardized format.
    
    Args:
        symbol: Position symbol
        position_size: Current position size
        unrealized_pnl: Unrealized P&L
        equity: Account equity
        leverage: Current leverage
        logger: Logger instance
    """
    if logger is None:
        logger = StructuredLogger("position")
    elif isinstance(logger, logging.Logger):
        logger = StructuredLogger(logger.name)
        
    context = {
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
        
    logger.info(f"Position update: {symbol}", **context)


def log_error_with_context(
    error: Exception,
    operation: str,
    component: str | None = None,
    **context: Any,
) -> None:
    """Log an error with full context information.
    
    Args:
        error: Exception that occurred
        operation: Operation that failed
        component: Component where error occurred
        **context: Additional context
    """
    logger = StructuredLogger("error", component=component)
    
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
    """Log a configuration change with standardized format.
    
    Args:
        config_key: Configuration key that changed
        old_value: Previous value
        new_value: New value
        component: Component where change occurred
        logger: Logger instance
    """
    if logger is None:
        logger = StructuredLogger("config", component=component)
    elif isinstance(logger, logging.Logger):
        logger = StructuredLogger(logger.name)
        
    context = {
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
    """Log a market data update with standardized format.
    
    Args:
        symbol: Market symbol
        price: Current price
        volume: Current volume
        timestamp: Update timestamp
        logger: Logger instance
    """
    if logger is None:
        logger = StructuredLogger("market_data")
    elif isinstance(logger, logging.Logger):
        logger = StructuredLogger(logger.name)
        
    context = {
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
    """Log system health status with standardized format.
    
    Args:
        status: Health status (healthy, degraded, unhealthy)
        component: Component being monitored
        metrics: Health metrics
        logger: Logger instance
    """
    if logger is None:
        logger = StructuredLogger("health", component=component)
    elif isinstance(logger, logging.Logger):
        logger = StructuredLogger(logger.name)
        
    context = {
        "operation": "health_check",
        "status": status,
    }
    
    if component:
        context["component"] = component
    if metrics:
        context.update(metrics)
        
    level = logging.INFO if status == "healthy" else logging.WARNING
    logger.log(level, f"Health status: {status}", **context)


# Convenience functions for common logging scenarios

def get_logger(name: str, component: str | None = None) -> StructuredLogger:
    """Get a structured logger instance.
    
    Args:
        name: Logger name
        component: Component name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, component=component)


# Decorator for automatic operation logging
def log_execution(
    operation: str | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
) -> callable:
    """Decorator to automatically log function execution.
    
    Args:
        operation: Operation description (defaults to function name)
        logger: Logger instance
        level: Log level
        include_args: Whether to include function arguments in logs
        include_result: Whether to include function result in logs
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            nonlocal logger
            if logger is None:
                logger = StructuredLogger(func.__module__)
            elif isinstance(logger, logging.Logger):
                logger = StructuredLogger(logger.name)
                
            context = {}
            if include_args:
                # Add non-sensitive args to context
                for i, arg in enumerate(args):
                    if not callable(arg) and not isinstance(arg, (type, module)):
                        context[f"arg_{i}"] = str(arg)
                        
            with log_operation(op_name, logger, level, **context):
                result = func(*args, **kwargs)
                
                if include_result and result is not None:
                    logger.debug(f"Result: {result}")
                    
                return result
                
        return wrapper
    return decorator
