"""Standardized logging patterns for consistent log formatting and structure."""

from __future__ import annotations

import logging
import sys
import time
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from decimal import Decimal
from functools import wraps
from typing import Any, ParamSpec, TextIO, TypeVar

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


ExecP = ParamSpec("ExecP")
ExecR = TypeVar("ExecR")


class UnifiedLogger:
    """Unified logger with structured context and optional console mirroring."""

    LEVEL_PREFIXES: dict[int, str] = {
        logging.INFO: "ℹ️",
        logging.WARNING: "⚠️",
        logging.ERROR: "❌",
        logging.CRITICAL: "🚨",
    }

    CHANNEL_PREFIXES: dict[str, tuple[int, str]] = {
        "success": (logging.INFO, "✅"),
        "data": (logging.INFO, "📊"),
        "trading": (logging.INFO, "💰"),
        "order": (logging.INFO, "📝"),
        "position": (logging.INFO, "📈"),
        "cache": (logging.DEBUG, "📦"),
        "storage": (logging.INFO, "💾"),
        "network": (logging.INFO, "🌐"),
        "analysis": (logging.INFO, "🔍"),
        "ml": (logging.INFO, "🧠"),
    }

    def __init__(
        self,
        name: str,
        *,
        component: str | None = None,
        enable_console: bool = False,
        output_stream: TextIO | None = None,
    ) -> None:
        """Initialize unified logger.

        Args:
            name: Logger name used for structured logging.
            component: Optional logical component name.
            enable_console: Whether to mirror logs to console with prefixes.
            output_stream: Console output stream (defaults to stdout).
        """
        self.logger = logging.getLogger(name)
        self.component = component
        self.enable_console = enable_console
        self.output_stream = output_stream or sys.stdout

    @property
    def name(self) -> str:
        """Expose underlying logger name for compatibility."""
        return self.logger.name

    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format message with structured context similar to legacy logger."""
        context_parts: list[str] = []

        if self.component:
            context_parts.append(f"component={self.component}")

        for key, value in kwargs.items():
            if key in LOG_FIELDS.values():
                if isinstance(value, Decimal):
                    formatted_value = f"{value:.8f}".rstrip("0").rstrip(".")
                elif isinstance(value, (int, float)):
                    formatted_value = f"{value}"
                else:
                    formatted_value = str(value)
                context_parts.append(f"{key}={formatted_value}")

        if context_parts:
            return f"{message} | " + " ".join(context_parts)
        return message

    def _emit_console(self, message: str, prefix: str | None = None) -> None:
        """Emit console output with graceful fallback."""
        text = f"{prefix} {message}" if prefix else message
        try:
            print(text, file=self.output_stream)
        except Exception:
            print(text)

    def log(
        self,
        level: int,
        message: str,
        *args: Any,
        console_prefix: str | None = None,
        console_message: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Log message at specified level and optionally mirror to console."""
        extra = kwargs.pop("extra", None)
        exc_info = kwargs.pop("exc_info", None)
        stack_info = kwargs.pop("stack_info", None)
        stacklevel = kwargs.pop("stacklevel", None)

        if args:
            try:
                rendered_message = message % args
            except Exception:
                rendered_message = message
        else:
            rendered_message = message

        formatted_message = self._format_message(rendered_message, **kwargs)

        log_kwargs: dict[str, Any] = {}
        if exc_info is not None:
            log_kwargs["exc_info"] = exc_info
        if stack_info is not None:
            log_kwargs["stack_info"] = stack_info
        if stacklevel is not None:
            log_kwargs["stacklevel"] = stacklevel
        if extra is not None:
            log_kwargs["extra"] = extra

        self.logger.log(level, formatted_message, **log_kwargs)

        if self.enable_console:
            prefix = console_prefix
            if prefix is None:
                prefix = self.LEVEL_PREFIXES.get(level)
            self._emit_console(console_message or rendered_message, prefix=prefix)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log informational message."""
        self.log(logging.INFO, message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log warning message."""
        self.log(logging.WARNING, message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log error message."""
        self.log(logging.ERROR, message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log debug message."""
        self.log(logging.DEBUG, message, *args, console_prefix=None, **kwargs)

    def critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log critical message."""
        self.log(logging.CRITICAL, message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log an error message with exception information."""
        kwargs.setdefault("exc_info", True)
        self.error(message, *args, **kwargs)

    # Console-first helper channels -------------------------------------------------
    def success(self, message: str, **kwargs: Any) -> None:
        """Log success event."""
        level, prefix = self.CHANNEL_PREFIXES["success"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def data(self, message: str, **kwargs: Any) -> None:
        """Log data-related event."""
        level, prefix = self.CHANNEL_PREFIXES["data"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def trading(self, message: str, **kwargs: Any) -> None:
        """Log trading event."""
        level, prefix = self.CHANNEL_PREFIXES["trading"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def order(self, message: str, **kwargs: Any) -> None:
        """Log order event."""
        level, prefix = self.CHANNEL_PREFIXES["order"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def position(self, message: str, **kwargs: Any) -> None:
        """Log position event."""
        level, prefix = self.CHANNEL_PREFIXES["position"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def cache(self, message: str, **kwargs: Any) -> None:
        """Log cache event."""
        level, prefix = self.CHANNEL_PREFIXES["cache"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def storage(self, message: str, **kwargs: Any) -> None:
        """Log storage event."""
        level, prefix = self.CHANNEL_PREFIXES["storage"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def network(self, message: str, **kwargs: Any) -> None:
        """Log network event."""
        level, prefix = self.CHANNEL_PREFIXES["network"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def analysis(self, message: str, **kwargs: Any) -> None:
        """Log analysis event."""
        level, prefix = self.CHANNEL_PREFIXES["analysis"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    def ml(self, message: str, **kwargs: Any) -> None:
        """Log ML event."""
        level, prefix = self.CHANNEL_PREFIXES["ml"]
        self.log(level, message, console_prefix=prefix, **kwargs)

    # Console presentation helpers --------------------------------------------------
    def print_section(self, title: str, char: str = "=", width: int = 50) -> None:
        """Print a section separator."""
        if not self.enable_console:
            return

        section = char * width
        if title:
            padding = max(width - len(title) - 2, 0)
            left = padding // 2
            right = padding - left
            section = f"{char * left} {title} {char * right}"

        try:
            print(section, file=self.output_stream)
        except Exception:
            print(section)

    def print_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Print a formatted table."""
        if not self.enable_console or not rows:
            return

        col_widths = [len(header) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                if idx < len(col_widths):
                    col_widths[idx] = max(col_widths[idx], len(str(cell)))

        header_row = " | ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
        separator = "-" * len(header_row)

        try:
            print(header_row, file=self.output_stream)
            print(separator, file=self.output_stream)
            for row in rows:
                formatted = " | ".join(
                    f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row[: len(col_widths)])
                )
                print(formatted, file=self.output_stream)
        except Exception:
            print(header_row)
            print(separator)
            for row in rows:
                formatted = " | ".join(
                    f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row[: len(col_widths)])
                )
                print(formatted)

    def printKeyValue(self, key: str, value: Any, indent: int = 0) -> None:
        """Print a key-value pair with optional indentation."""
        if not self.enable_console:
            return

        prefix = "   " * indent
        line = f"{prefix}{key}: {value}"
        try:
            print(line, file=self.output_stream)
        except Exception:
            print(line)


# Preserve legacy name for downstream imports/patches
StructuredLogger = UnifiedLogger


@contextmanager
def log_operation(
    operation: str,
    logger: StructuredLogger | logging.Logger | None = None,
    level: int = logging.INFO,
    **context: Any,
) -> Iterator[None]:
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

    error_context: dict[str, Any] = {
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


# Convenience functions for common logging scenarios


def get_logger(
    name: str, component: str | None = None, *, enable_console: bool = False
) -> UnifiedLogger:
    """Get a unified logger instance.

    Args:
        name: Logger name
        component: Component name
        enable_console: Whether to enable console mirroring

    Returns:
        UnifiedLogger instance
    """
    return StructuredLogger(name, component=component, enable_console=enable_console)


# Decorator for automatic operation logging
def log_execution(
    operation: str | None = None,
    logger: StructuredLogger | logging.Logger | None = None,
    level: int = logging.INFO,
    include_args: bool = False,
    include_result: bool = False,
) -> Callable[[Callable[ExecP, ExecR]], Callable[ExecP, ExecR]]:
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

    def decorator(func: Callable[ExecP, ExecR]) -> Callable[ExecP, ExecR]:
        @wraps(func)
        def wrapper(*args: ExecP.args, **kwargs: ExecP.kwargs) -> ExecR:
            op_name = operation or func.__name__

            logger_obj: StructuredLogger
            if logger is None:
                logger_obj = StructuredLogger(func.__module__)
            elif isinstance(logger, logging.Logger):
                logger_obj = StructuredLogger(logger.name)
            else:
                logger_obj = logger

            context: dict[str, Any] = {}
            if include_args:
                # Add non-sensitive args to context
                for i, arg in enumerate(args):
                    if not callable(arg) and not isinstance(arg, (type, types.ModuleType)):
                        context[f"arg_{i}"] = str(arg)

            with log_operation(op_name, logger_obj, level, **context):
                result = func(*args, **kwargs)

                if include_result and result is not None:
                    logger_obj.debug(f"Result: {result}")

                return result

        return wrapper

    return decorator
