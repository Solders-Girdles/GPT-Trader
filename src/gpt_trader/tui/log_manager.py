"""Singleton log manager for TUI.

This module provides a singleton logging handler that distributes logs to all
active LogWidgets in the TUI. The handler is attached once at app startup and
shared by all LogWidget instances.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from rich.text import Text

from gpt_trader.tui.theme import THEME

if TYPE_CHECKING:
    from textual.widgets import Log

# Separate logger for TuiLogHandler errors (not attached to TuiLogHandler to avoid recursion)
_error_logger = logging.getLogger("tui.log_handler.errors")
_error_logger.propagate = False  # Prevent recursion through root logger's TUI handler
_error_logger.addHandler(logging.NullHandler())  # Suppress "no handler" warnings


class ImprovedExceptionFormatter(logging.Formatter):
    """Custom formatter that improves exception/traceback display."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with improved exception formatting."""
        # Use parent formatter for base message
        base_message = super().format(record)

        # If there's exception info, format it better
        if record.exc_info:
            # Get the formatted exception from the record
            if record.exc_text:
                exc_text = record.exc_text
            else:
                exc_text = self.formatException(record.exc_info)

            # Add indentation to exception lines for better readability
            indented_exc = "\n".join(
                f"  │ {line}" if line.strip() else "  │" for line in exc_text.split("\n")
            )

            # Combine base message with formatted exception
            return f"{base_message}\n  ╰─ Exception:\n{indented_exc}"

        return base_message


class TuiLogHandler(logging.Handler):
    """Single handler that distributes logs to all active LogWidgets."""

    def __init__(self) -> None:
        super().__init__()
        self._widgets: dict[Log, int] = {}  # widget -> min_level mapping
        self._callbacks: dict[Log, Callable[[int], None]] = {}  # widget -> callback mapping
        self._logger_filters: dict[Log, str] = {}  # widget -> logger_filter mapping
        # Use improved formatter for better exception display
        formatter = ImprovedExceptionFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        self.setFormatter(formatter)
        self.setLevel(logging.DEBUG)

    def register_widget(
        self,
        widget: Log,
        min_level: int = logging.INFO,
        on_log_callback: Callable[[int], None] | None = None,
    ) -> None:
        """
        Register a Log widget to receive logs at or above min_level.

        Args:
            widget: The Log widget to register
            min_level: Minimum log level to display
            on_log_callback: Optional callback called with log level on each log
        """
        self._widgets[widget] = min_level
        self._logger_filters[widget] = ""  # Default: show all loggers
        if on_log_callback:
            self._callbacks[widget] = on_log_callback

    def update_widget_level(self, widget: Log, min_level: int) -> None:
        """Update the minimum level for a registered widget."""
        if widget in self._widgets:
            self._widgets[widget] = min_level

    def update_widget_logger_filter(self, widget: Log, logger_filter: str) -> None:
        """Update the logger filter pattern for a registered widget."""
        if widget in self._widgets:
            self._logger_filters[widget] = logger_filter.lower()

    def unregister_widget(self, widget: Log) -> None:
        """Unregister a Log widget."""
        self._widgets.pop(widget, None)
        self._callbacks.pop(widget, None)
        self._logger_filters.pop(widget, None)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log to all registered widgets that accept this level."""
        import threading

        try:
            msg = self.format(record)

            # Create Rich Text object with style (not markup) to prevent injection
            # Using style= treats msg as literal text, preventing markup in log messages
            # from corrupting coloring or injecting formatting
            if record.levelno >= logging.ERROR:
                styled_msg = Text(msg, style=THEME.colors.error)  # Warm coral-red
            elif record.levelno >= logging.WARNING:
                styled_msg = Text(msg, style=THEME.colors.warning)  # Warm amber
            elif record.levelno >= logging.INFO:
                styled_msg = Text(msg, style=THEME.colors.success)  # Warm green
            else:
                styled_msg = Text(msg, style=THEME.colors.text_muted)  # Muted grey

            # Check if we're on the main thread
            is_main_thread = threading.current_thread() is threading.main_thread()

            # Write to widgets that accept this level and logger filter
            for widget, min_level in list(self._widgets.items()):
                if record.levelno >= min_level:
                    # Check logger filter (case-insensitive substring match)
                    logger_filter = self._logger_filters.get(widget, "")
                    if logger_filter and logger_filter not in record.name.lower():
                        continue  # Skip this log - doesn't match filter

                    # Lifecycle guard: only write to mounted widgets with active app
                    if not widget.is_mounted or not widget.app:
                        continue

                    try:
                        if is_main_thread:
                            # Already on main thread, call directly
                            self._write_to_widget(widget, styled_msg)
                            # Call counter callback if registered
                            if widget in self._callbacks:
                                self._callbacks[widget](record.levelno)
                        else:
                            # On background thread, use call_from_thread
                            widget.app.call_from_thread(self._write_to_widget, widget, styled_msg)
                            # Call counter callback if registered
                            if widget in self._callbacks:
                                widget.app.call_from_thread(self._callbacks[widget], record.levelno)
                    except Exception as e:
                        # Log errors for debugging, but don't crash the logger
                        # Use separate logger not attached to TuiLogHandler to avoid recursion
                        error_type = type(e).__name__
                        _error_logger.debug(f"Error writing to log widget: {error_type}: {e}")
                        # If widget unmounted, remove it from registry
                        if error_type == "NoActiveAppError" or not widget.is_mounted:
                            self.unregister_widget(widget)

        except Exception:
            self.handleError(record)

    @staticmethod
    def _write_to_widget(widget: Log, message: Text) -> None:
        """Write Rich Text to widget on main thread.

        IMPORTANT: We manually add \\n because the formatter doesn't include one.
        If the formatter is ever changed to add \\n, this will cause double-spacing.
        See test_formatter_does_not_add_newline() for regression detection.
        """
        # Log.write() expects strings, not Text objects
        # Use markup property to get Rich markup string, then add newline
        # The Log widget's highlight=True will parse the markup for colored output
        widget.write(message.markup + "\n")


# Global singleton instance
_tui_log_handler: TuiLogHandler | None = None


def get_tui_log_handler() -> TuiLogHandler:
    """Get or create the singleton TUI log handler."""
    global _tui_log_handler
    if _tui_log_handler is None:
        _tui_log_handler = TuiLogHandler()
    return _tui_log_handler


def attach_tui_log_handler() -> None:
    """Attach the singleton handler to root logger."""
    handler = get_tui_log_handler()
    root_logger = logging.getLogger()

    # Only attach if not already attached
    if handler not in root_logger.handlers:
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)


def detach_tui_log_handler() -> None:
    """Detach the singleton handler from root logger."""
    global _tui_log_handler
    if _tui_log_handler:
        logging.getLogger().removeHandler(_tui_log_handler)
        _tui_log_handler = None
