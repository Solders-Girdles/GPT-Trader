"""Singleton log manager for TUI.

This module provides a singleton logging handler that distributes logs to all
active LogWidgets in the TUI. The handler is attached once at app startup and
shared by all LogWidget instances.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rich.text import Text

if TYPE_CHECKING:
    from textual.widgets import Log


class TuiLogHandler(logging.Handler):
    """Single handler that distributes logs to all active LogWidgets."""

    def __init__(self) -> None:
        super().__init__()
        self._widgets: dict[Log, int] = {}  # widget -> min_level mapping
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        self.setFormatter(formatter)
        self.setLevel(logging.DEBUG)

    def register_widget(self, widget: Log, min_level: int = logging.INFO) -> None:
        """Register a Log widget to receive logs at or above min_level."""
        self._widgets[widget] = min_level

    def update_widget_level(self, widget: Log, min_level: int) -> None:
        """Update the minimum level for a registered widget."""
        if widget in self._widgets:
            self._widgets[widget] = min_level

    def unregister_widget(self, widget: Log) -> None:
        """Unregister a Log widget."""
        self._widgets.pop(widget, None)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log to all registered widgets that accept this level."""
        import sys
        import threading

        try:
            msg = self.format(record)

            # Create Rich Text object with markup for color
            if record.levelno >= logging.ERROR:
                styled_msg = Text.from_markup(f"[#bf616a]{msg}[/#bf616a]")  # Nord Red
            elif record.levelno >= logging.WARNING:
                styled_msg = Text.from_markup(f"[#ebcb8b]{msg}[/#ebcb8b]")  # Nord Yellow
            elif record.levelno >= logging.INFO:
                styled_msg = Text.from_markup(f"[#a3be8c]{msg}[/#a3be8c]")  # Nord Green
            else:
                styled_msg = Text.from_markup(f"[#4c566a]{msg}[/#4c566a]")  # Nord Grey

            # Check if we're on the main thread
            is_main_thread = threading.current_thread() is threading.main_thread()

            # Write to widgets that accept this level
            for widget, min_level in list(self._widgets.items()):
                if record.levelno >= min_level:
                    # Lifecycle guard: only write to mounted widgets with active app
                    if not widget.is_mounted or not widget.app:
                        continue

                    try:
                        if is_main_thread:
                            # Already on main thread, call directly
                            self._write_to_widget(widget, styled_msg)
                        else:
                            # On background thread, use call_from_thread
                            widget.app.call_from_thread(self._write_to_widget, widget, styled_msg)
                    except Exception as e:
                        # Log errors for debugging, but don't crash the logger
                        # Use stderr to avoid recursive logging
                        error_type = type(e).__name__
                        print(
                            f"[TuiLogHandler] Error writing to log widget: {error_type}: {e}",
                            file=sys.stderr,
                        )
                        # If widget unmounted, remove it from registry
                        if error_type == "NoActiveAppError" or not widget.is_mounted:
                            self.unregister_widget(widget)

        except Exception:
            self.handleError(record)

    @staticmethod
    def _write_to_widget(widget: Log, message: Text) -> None:
        """Write Rich Text to widget on main thread."""
        # Convert Text object to string (preserving ANSI codes) before writing
        # This fixes the TypeError: expected string or bytes-like object
        widget.write(message)


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
