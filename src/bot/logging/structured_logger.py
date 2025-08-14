"""Structured logging system for GPT-Trader.

This module provides structured JSON logging with:
- Contextual information
- Trade event tracking
- Performance metrics
- Error correlation
- Log aggregation support
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from bot.config import get_config


class LogFormat(str, Enum):
    """Log output formats."""

    JSON = "json"
    TEXT = "text"
    COLORED = "colored"


class StructuredFormatter(logging.Formatter):
    """Formatter for structured JSON logging."""

    def __init__(
        self,
        format_type: LogFormat = LogFormat.JSON,
        include_traceback: bool = True,
        include_context: bool = True,
        context_fields: dict[str, Any] | None = None,
    ) -> None:
        """Initialize structured formatter.

        Args:
            format_type: Output format type.
            include_traceback: Whether to include stack traces.
            include_context: Whether to include context fields.
            context_fields: Additional context fields to include.
        """
        super().__init__()
        self.format_type = format_type
        self.include_traceback = include_traceback
        self.include_context = include_context
        self.context_fields = context_fields or {}

        # Text format string
        self.text_format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

        # Color codes for colored output
        self.colors = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
            "RESET": "\033[0m",  # Reset
        }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record.

        Args:
            record: Log record to format.

        Returns:
            Formatted log string.
        """
        if self.format_type == LogFormat.JSON:
            return self._format_json(record)
        elif self.format_type == LogFormat.COLORED:
            return self._format_colored(record)
        else:
            return self._format_text(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Format as JSON.

        Args:
            record: Log record to format.

        Returns:
            JSON formatted string.
        """
        log_data = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
            "process_name": record.processName,
        }

        # Add context fields
        if self.include_context:
            log_data.update(self.context_fields)

        # Add extra fields from record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info and self.include_traceback:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": (
                    traceback.format_exception(*record.exc_info) if record.exc_info else None
                ),
            }

        # Add any custom attributes
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "extra_fields",
            ]:
                log_data[key] = value

        return json.dumps(log_data, default=str)

    def _format_text(self, record: logging.LogRecord) -> str:
        """Format as plain text.

        Args:
            record: Log record to format.

        Returns:
            Text formatted string.
        """
        # Use standard text formatting
        formatter = logging.Formatter(self.text_format)
        return formatter.format(record)

    def _format_colored(self, record: logging.LogRecord) -> str:
        """Format with colors for terminal output.

        Args:
            record: Log record to format.

        Returns:
            Colored text string.
        """
        levelname = record.levelname
        color = self.colors.get(levelname, self.colors["RESET"])
        reset = self.colors["RESET"]

        # Color the level name
        record.levelname = f"{color}{levelname}{reset}"

        # Format the message
        formatted = self._format_text(record)

        # Reset level name for other handlers
        record.levelname = levelname

        return formatted


class TradeLogger:
    """Specialized logger for trade events."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize trade logger.

        Args:
            logger: Base logger to use.
        """
        self.logger = logger
        self.trade_id_counter = 0

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID.

        Returns:
            Unique trade identifier.
        """
        self.trade_id_counter += 1
        return f"TRD-{datetime.now().strftime('%Y%m%d')}-{self.trade_id_counter:06d}"

    def log_order_placed(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float | None = None,
        order_type: str = "market",
        **kwargs,
    ) -> None:
        """Log order placement.

        Args:
            symbol: Trading symbol.
            side: Buy or sell.
            quantity: Order quantity.
            price: Order price (for limit orders).
            order_type: Type of order.
            **kwargs: Additional order details.
        """
        self.logger.info(
            f"Order placed: {side} {quantity} {symbol}",
            extra={
                "extra_fields": {
                    "event_type": "order_placed",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "order_type": order_type,
                    "trade_id": self._generate_trade_id(),
                    **kwargs,
                }
            },
        )

    def log_order_filled(
        self,
        symbol: str,
        side: str,
        quantity: float,
        fill_price: float,
        commission: float = 0.0,
        **kwargs,
    ) -> None:
        """Log order fill.

        Args:
            symbol: Trading symbol.
            side: Buy or sell.
            quantity: Filled quantity.
            fill_price: Execution price.
            commission: Trading commission.
            **kwargs: Additional fill details.
        """
        self.logger.info(
            f"Order filled: {side} {quantity} {symbol} @ {fill_price}",
            extra={
                "extra_fields": {
                    "event_type": "order_filled",
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "fill_price": fill_price,
                    "commission": commission,
                    "total_value": quantity * fill_price,
                    **kwargs,
                }
            },
        )

    def log_position_opened(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        **kwargs,
    ) -> None:
        """Log position opening.

        Args:
            symbol: Trading symbol.
            quantity: Position size.
            entry_price: Entry price.
            stop_loss: Stop loss price.
            take_profit: Take profit price.
            **kwargs: Additional position details.
        """
        self.logger.info(
            f"Position opened: {symbol} {quantity} @ {entry_price}",
            extra={
                "extra_fields": {
                    "event_type": "position_opened",
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_value": quantity * entry_price,
                    **kwargs,
                }
            },
        )

    def log_position_closed(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        **kwargs,
    ) -> None:
        """Log position closing.

        Args:
            symbol: Trading symbol.
            quantity: Position size.
            entry_price: Entry price.
            exit_price: Exit price.
            pnl: Profit/loss.
            **kwargs: Additional closing details.
        """
        self.logger.info(
            f"Position closed: {symbol} P&L: {pnl:.2f}",
            extra={
                "extra_fields": {
                    "event_type": "position_closed",
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": ((exit_price - entry_price) / entry_price) * 100,
                    **kwargs,
                }
            },
        )


class PerformanceLogger:
    """Logger for performance metrics."""

    def __init__(self, logger: logging.Logger) -> None:
        """Initialize performance logger.

        Args:
            logger: Base logger to use.
        """
        self.logger = logger

    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "",
        tags: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """Log a performance metric.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            unit: Unit of measurement.
            tags: Additional tags.
            **kwargs: Additional metric details.
        """
        self.logger.info(
            f"Metric: {metric_name}={value}{unit}",
            extra={
                "extra_fields": {
                    "event_type": "metric",
                    "metric_name": metric_name,
                    "value": value,
                    "unit": unit,
                    "tags": tags or {},
                    **kwargs,
                }
            },
        )

    def log_latency(
        self, operation: str, latency_ms: float, threshold_ms: float | None = None, **kwargs
    ) -> None:
        """Log operation latency.

        Args:
            operation: Operation name.
            latency_ms: Latency in milliseconds.
            threshold_ms: Threshold for slow operations.
            **kwargs: Additional details.
        """
        level = logging.INFO
        if threshold_ms and latency_ms > threshold_ms:
            level = logging.WARNING

        self.logger.log(
            level,
            f"Latency: {operation} took {latency_ms:.2f}ms",
            extra={
                "extra_fields": {
                    "event_type": "latency",
                    "operation": operation,
                    "latency_ms": latency_ms,
                    "threshold_ms": threshold_ms,
                    "slow": threshold_ms and latency_ms > threshold_ms,
                    **kwargs,
                }
            },
        )

    def log_throughput(self, operation: str, count: int, duration_seconds: float, **kwargs) -> None:
        """Log operation throughput.

        Args:
            operation: Operation name.
            count: Number of operations.
            duration_seconds: Duration in seconds.
            **kwargs: Additional details.
        """
        throughput = count / duration_seconds if duration_seconds > 0 else 0

        self.logger.info(
            f"Throughput: {operation} {throughput:.2f} ops/sec",
            extra={
                "extra_fields": {
                    "event_type": "throughput",
                    "operation": operation,
                    "count": count,
                    "duration_seconds": duration_seconds,
                    "throughput_per_second": throughput,
                    **kwargs,
                }
            },
        )


class StructuredLogger:
    """Main structured logging class."""

    def __init__(
        self,
        name: str,
        level: str | None = None,
        format_type: LogFormat | None = None,
        log_file: Path | None = None,
        max_size_mb: int = 10,
        backup_count: int = 5,
        context_fields: dict[str, Any] | None = None,
    ) -> None:
        """Initialize structured logger.

        Args:
            name: Logger name.
            level: Logging level.
            format_type: Output format type.
            log_file: Optional log file path.
            max_size_mb: Maximum log file size in MB.
            backup_count: Number of backup files to keep.
            context_fields: Additional context fields.
        """
        # Get configuration
        config = get_config()

        # Set defaults from config
        if level is None:
            level = config.logging.level
        if format_type is None:
            format_type = LogFormat.JSON if config.logging.structured_logging else LogFormat.TEXT
        if log_file is None and config.logging.file_path:
            log_file = config.logging.file_path

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.logger.propagate = False  # Don't propagate to root logger

        # Clear existing handlers
        self.logger.handlers.clear()

        # Create formatter
        self.formatter = StructuredFormatter(
            format_type=format_type, context_fields=context_fields or {}
        )

        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)

        # Add file handler if specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                str(log_file), maxBytes=max_size_mb * 1024 * 1024, backupCount=backup_count
            )
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        # Create specialized loggers
        self.trade = TradeLogger(self.logger)
        self.performance = PerformanceLogger(self.logger)

    def add_context(self, **fields) -> None:
        """Add context fields to all future logs.

        Args:
            **fields: Context fields to add.
        """
        self.formatter.context_fields.update(fields)

    def clear_context(self) -> None:
        """Clear all context fields."""
        self.formatter.context_fields.clear()

    def with_context(self, **fields):
        """Create a context manager with temporary context fields.

        Args:
            **fields: Temporary context fields.

        Returns:
            Context manager.
        """
        return LogContext(self, fields)

    # Delegate standard logging methods
    def debug(self, msg, *args, **kwargs) -> None:
        """Log debug message."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs) -> None:
        """Log info message."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs) -> None:
        """Log warning message."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs) -> None:
        """Log error message."""
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs) -> None:
        """Log critical message."""
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(msg, *args, **kwargs)


class LogContext:
    """Context manager for temporary log context."""

    def __init__(self, logger: StructuredLogger, fields: dict[str, Any]) -> None:
        """Initialize log context.

        Args:
            logger: Structured logger instance.
            fields: Context fields to add.
        """
        self.logger = logger
        self.fields = fields
        self.original_fields = {}

    def __enter__(self):
        """Enter context."""
        # Save original fields that will be overwritten
        for key in self.fields:
            if key in self.logger.formatter.context_fields:
                self.original_fields[key] = self.logger.formatter.context_fields[key]

        # Add new fields
        self.logger.formatter.context_fields.update(self.fields)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        # Remove added fields
        for key in self.fields:
            self.logger.formatter.context_fields.pop(key, None)

        # Restore original fields
        self.logger.formatter.context_fields.update(self.original_fields)


# Factory function for creating loggers
def get_structured_logger(name: str, **kwargs) -> StructuredLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name.
        **kwargs: Additional configuration.

    Returns:
        StructuredLogger instance.
    """
    return StructuredLogger(name, **kwargs)
