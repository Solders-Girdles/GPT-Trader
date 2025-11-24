"""
Console logging utilities for consistent user-facing output.

This module provides utilities for console output that complement the
structured logging system, ensuring consistent user experience while
maintaining proper logging practices.
"""

from __future__ import annotations

from typing import Any, TextIO

from gpt_trader.utilities.logging_patterns import UnifiedLogger


class ConsoleLogger(UnifiedLogger):
    """Console-friendly unified logger with opinionated defaults."""

    def __init__(self, enable_console: bool = True, output_stream: TextIO | None = None) -> None:
        super().__init__(
            __name__,
            component="console",
            enable_console=enable_console,
            output_stream=output_stream,
        )


# Global console logger instance
_console_logger: ConsoleLogger | None = None


def get_console_logger(enable_console: bool = True) -> ConsoleLogger:
    """
    Get the global console logger instance.

    Args:
        enable_console: Whether to enable console output

    Returns:
        ConsoleLogger instance
    """
    global _console_logger
    if _console_logger is None:
        _console_logger = ConsoleLogger(enable_console=enable_console)
    elif enable_console and not _console_logger.enable_console:
        _console_logger.enable_console = True
    return _console_logger


def console_success(message: str, **kwargs: Any) -> None:
    """Log success message using global console logger."""
    get_console_logger().success(message, **kwargs)


def console_error(message: str, **kwargs: Any) -> None:
    """Log error message using global console logger."""
    get_console_logger().error(message, **kwargs)


def console_warning(message: str, **kwargs: Any) -> None:
    """Log warning message using global console logger."""
    get_console_logger().warning(message, **kwargs)


def console_info(message: str, **kwargs: Any) -> None:
    """Log info message using global console logger."""
    get_console_logger().info(message, **kwargs)


def console_data(message: str, **kwargs: Any) -> None:
    """Log data message using global console logger."""
    get_console_logger().data(message, **kwargs)


def console_trading(message: str, **kwargs: Any) -> None:
    """Log trading message using global console logger."""
    get_console_logger().trading(message, **kwargs)


def console_order(message: str, **kwargs: Any) -> None:
    """Log order message using global console logger."""
    get_console_logger().order(message, **kwargs)


def console_position(message: str, **kwargs: Any) -> None:
    """Log position message using global console logger."""
    get_console_logger().position(message, **kwargs)


def console_cache(message: str, **kwargs: Any) -> None:
    """Log cache message using global console logger."""
    get_console_logger().cache(message, **kwargs)


def console_storage(message: str, **kwargs: Any) -> None:
    """Log storage message using global console logger."""
    get_console_logger().storage(message, **kwargs)


def console_network(message: str, **kwargs: Any) -> None:
    """Log network message using global console logger."""
    get_console_logger().network(message, **kwargs)


def console_analysis(message: str, **kwargs: Any) -> None:
    """Log analysis message using global console logger."""
    get_console_logger().analysis(message, **kwargs)


def console_ml(message: str, **kwargs: Any) -> None:
    """Log ML message using global console logger."""
    get_console_logger().ml(message, **kwargs)


def console_section(title: str, char: str = "=", width: int = 50) -> None:
    """Print section separator using global console logger."""
    get_console_logger().print_section(title, char, width)


def console_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print table using global console logger."""
    get_console_logger().print_table(headers, rows)


def console_key_value(key: str, value: Any, indent: int = 0) -> None:
    """Print key-value pair using global console logger."""
    get_console_logger().printKeyValue(key, value, indent)
