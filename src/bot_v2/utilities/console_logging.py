"""
Console logging utilities for consistent user-facing output.

This module provides utilities for console output that complement the
structured logging system, ensuring consistent user experience while
maintaining proper logging practices.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__)


class ConsoleLogger:
    """
    Provides console output utilities that complement structured logging.

    This class helps maintain consistency between console output for users
    and structured logging for debugging/monitoring.
    """

    def __init__(self, enable_console: bool = True, output_stream: TextIO | None = None) -> None:
        """
        Initialize console logger.

        Args:
            enable_console: Whether to enable console output
            output_stream: Output stream (defaults to stdout)
        """
        self.enable_console = enable_console
        self.output_stream = output_stream or sys.stdout

    def success(self, message: str, **kwargs: Any) -> None:
        """
        Log success message with optional console output.

        Args:
            message: Success message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("✅", message)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log error message with optional console output.

        Args:
            message: Error message
            **kwargs: Additional context for structured logging
        """
        logger.error(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("❌", message)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning message with optional console output.

        Args:
            message: Warning message
            **kwargs: Additional context for structured logging
        """
        logger.warning(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("⚠️", message)

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info message with optional console output.

        Args:
            message: Info message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("ℹ️", message)

    def data(self, message: str, **kwargs: Any) -> None:
        """
        Log data-related message with optional console output.

        Args:
            message: Data message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("📊", message)

    def trading(self, message: str, **kwargs: Any) -> None:
        """
        Log trading-related message with optional console output.

        Args:
            message: Trading message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("💰", message)

    def order(self, message: str, **kwargs: Any) -> None:
        """
        Log order-related message with optional console output.

        Args:
            message: Order message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("📝", message)

    def position(self, message: str, **kwargs: Any) -> None:
        """
        Log position-related message with optional console output.

        Args:
            message: Position message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("📈", message)

    def cache(self, message: str, **kwargs: Any) -> None:
        """
        Log cache-related message with optional console output.

        Args:
            message: Cache message
            **kwargs: Additional context for structured logging
        """
        logger.debug(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("📦", message)

    def storage(self, message: str, **kwargs: Any) -> None:
        """
        Log storage-related message with optional console output.

        Args:
            message: Storage message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("💾", message)

    def network(self, message: str, **kwargs: Any) -> None:
        """
        Log network-related message with optional console output.

        Args:
            message: Network message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("🌐", message)

    def analysis(self, message: str, **kwargs: Any) -> None:
        """
        Log analysis-related message with optional console output.

        Args:
            message: Analysis message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("🔍", message)

    def ml(self, message: str, **kwargs: Any) -> None:
        """
        Log ML-related message with optional console output.

        Args:
            message: ML message
            **kwargs: Additional context for structured logging
        """
        logger.info(message, **kwargs)
        if self.enable_console:
            self._print_with_prefix("🧠", message)

    def print_section(self, title: str, char: str = "=", width: int = 50) -> None:
        """
        Print a section separator.

        Args:
            title: Section title
            char: Character to use for separator
            width: Width of separator
        """
        if self.enable_console:
            section = f"{char * width}"
            if title:
                section = f"{char * ((width - len(title) - 2) // 2)} {title} {char * ((width - len(title) - 2) // 2)}"
            try:
                print(section, file=self.output_stream)
            except Exception:
                print(section)

    def print_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """
        Print a formatted table.

        Args:
            headers: Table headers
            rows: Table rows
        """
        if not self.enable_console or not rows:
            return

        # Calculate column widths
        col_widths = [len(header) for header in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print headers
        header_row = " | ".join(f"{header:<{col_widths[i]}}" for i, header in enumerate(headers))
        separator = "-" * len(header_row)

        try:
            print(header_row, file=self.output_stream)
            print(separator, file=self.output_stream)

            # Print rows
            for row in rows:
                formatted_row = " | ".join(
                    f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row[: len(col_widths)])
                )
                print(formatted_row, file=self.output_stream)
        except Exception:
            print(header_row)
            print(separator)

            # Print rows
            for row in rows:
                formatted_row = " | ".join(
                    f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row[: len(col_widths)])
                )
                print(formatted_row)

    def printKeyValue(self, key: str, value: Any, indent: int = 0) -> None:
        """
        Print a key-value pair with optional indentation.

        Args:
            key: Key name
            value: Value to display
            indent: Indentation level
        """
        if self.enable_console:
            prefix = "   " * indent
            try:
                print(f"{prefix}{key}: {value}", file=self.output_stream)
            except Exception:
                print(f"{prefix}{key}: {value}")

    def _print_with_prefix(self, prefix: str, message: str) -> None:
        """
        Print message with prefix.

        Args:
            prefix: Message prefix (emoji)
            message: Message to print
        """
        try:
            print(f"{prefix} {message}", file=self.output_stream)
        except Exception:
            # Fallback if there's an issue with output stream
            print(f"{prefix} {message}")


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
