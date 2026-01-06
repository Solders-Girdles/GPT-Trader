"""TUI log handler and entry classes.

Contains:
- LogEntry: Dataclass for cached log entries
- TuiLogHandler: Handler that distributes logs to RichLog widgets
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.text import Text

from gpt_trader.tui.log_constants import (
    DEFAULT_REPLAY_COUNT,
    MAX_LOG_ENTRIES,
    detect_category,
)
from gpt_trader.tui.log_formatters import (
    CompactTuiFormatter,
    ImprovedExceptionFormatter,
    StructuredTuiFormatter,
)
from gpt_trader.tui.theme import THEME

if TYPE_CHECKING:
    from textual.widgets import RichLog

# Separate logger for TuiLogHandler errors (not attached to TuiLogHandler to avoid recursion)
_error_logger = logging.getLogger("tui.log_handler.errors")
_error_logger.propagate = False  # Prevent recursion through root logger's TUI handler
_error_logger.addHandler(logging.NullHandler())  # Suppress "no handler" warnings


@dataclass
class LogEntry:
    """Cached log entry for replay to new widgets.

    Extended with metadata for enhanced debugging and AI parsing:
    - raw_message: Original unformatted message for search/export
    - is_json: Whether message contains valid JSON
    - json_data: Parsed JSON data if applicable
    - is_multiline: Whether message spans multiple lines
    - short_logger: Last component of logger name
    - level_name: Level name string (ERROR, WARNING, INFO, DEBUG)
    - category: Log category (startup, trading, risk, market, ui, system)
    - correlation_id: Request correlation ID if available
    - domain_fields: Symbol, order_id, etc. from context
    """

    level: int
    logger_name: str
    styled_message: Text
    timestamp: float = field(default_factory=time.time)
    raw_message: str = ""
    is_json: bool = False
    json_data: dict[str, Any] | None = None
    is_multiline: bool = False
    # AI-friendly structured fields
    short_logger: str = ""
    level_name: str = ""
    category: str = ""
    correlation_id: str | None = None
    domain_fields: dict[str, Any] | None = None
    # Compact formatted message (without timestamp/logger prefix)
    compact_message: str = ""


class TuiLogHandler(logging.Handler):
    """Single handler that distributes logs to all active LogWidgets.

    Features:
    - Distributes logs to all registered RichLog widgets
    - Memory-limited buffer for log history
    - Replay recent logs when new widgets register
    """

    # Maximum logs to buffer while widget is paused
    PAUSE_BUFFER_MAX = 500

    # Pattern to detect JSON objects in log messages
    _JSON_PATTERN = re.compile(r"\{[^{}]+\}")

    def __init__(
        self,
        max_entries: int = MAX_LOG_ENTRIES,
        replay_count: int = DEFAULT_REPLAY_COUNT,
    ) -> None:
        """Initialize the TUI log handler.

        Args:
            max_entries: Maximum log entries to keep in memory buffer.
            replay_count: Number of recent logs to replay to new widgets.
        """
        super().__init__()
        self._widgets: dict[RichLog, int] = {}  # widget -> min_level mapping
        self._callbacks: dict[RichLog, Callable[[int], None]] = {}  # widget -> callback mapping
        self._logger_filters: dict[RichLog, str] = {}  # widget -> logger_filter mapping
        # Per-widget display preferences
        self._show_startup: dict[RichLog, bool] = {}  # widget -> show startup INFO logs

        # Memory-limited log buffer using deque
        self._log_buffer: deque[LogEntry] = deque(maxlen=max_entries)
        self._replay_count = replay_count

        # Pause/resume state tracking
        self._paused_widgets: set[RichLog] = set()
        self._paused_buffers: dict[RichLog, deque[LogEntry]] = {}
        self._pause_lock = threading.Lock()

        # Error tracking for jump-to-error navigation
        self._error_indices: list[int] = []  # Indices of error entries in buffer
        self._buffer_index = 0  # Current buffer position for error tracking

        # Formatters for different display modes
        self._verbose_formatter = ImprovedExceptionFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S",
        )
        self._compact_formatter = CompactTuiFormatter()
        self._structured_formatter = StructuredTuiFormatter()

        # Default format mode: compact, verbose, or structured
        # "structured" is the new default for improved readability
        self._format_mode = "structured"

        self.setFormatter(self._verbose_formatter)
        self.setLevel(logging.DEBUG)

    def register_widget(
        self,
        widget: RichLog,
        min_level: int = logging.INFO,
        on_log_callback: Callable[[int], None] | None = None,
        replay_logs: bool = True,
        *,
        show_startup: bool = False,
    ) -> None:
        """
        Register a RichLog widget to receive logs at or above min_level.

        Args:
            widget: The RichLog widget to register
            min_level: Minimum log level to display
            on_log_callback: Optional callback called with log level on each log
            replay_logs: Whether to replay recent logs to this widget
        """
        self._widgets[widget] = min_level
        self._logger_filters[widget] = ""  # Default: show all loggers
        self._show_startup[widget] = show_startup
        if on_log_callback:
            self._callbacks[widget] = on_log_callback

        # Replay recent logs to new widget
        if replay_logs and self._log_buffer:
            self._replay_logs_to_widget(widget, min_level)

    def update_widget_show_startup(self, widget: RichLog, show_startup: bool) -> None:
        """Update whether a widget should display startup INFO logs."""
        if widget in self._widgets:
            self._show_startup[widget] = show_startup

    def refresh_widget(self, widget: RichLog) -> None:
        """Clear and replay logs for a widget with its current filters."""
        if widget not in self._widgets:
            return

        min_level = self._widgets.get(widget, logging.INFO)
        try:
            widget.clear()
        except Exception:
            return
        self._replay_logs_to_widget(widget, min_level)

    def _replay_logs_to_widget(
        self,
        widget: RichLog,
        min_level: int,
    ) -> None:
        """Replay recent logs to a newly registered widget.

        Args:
            widget: The RichLog widget to replay logs to.
            min_level: Minimum log level to replay.
        """
        try:
            # Ensure widget is mounted and has an app before replaying
            if not hasattr(widget, "is_mounted") or not widget.is_mounted:
                # Widget not ready yet, skip replay (will be handled by normal emit flow)
                return
            if not hasattr(widget, "app") or not widget.app:
                return

            # Get the most recent logs up to replay_count
            replay_limit = getattr(widget, "max_lines", None) or self._replay_count
            logs_to_replay = list(self._log_buffer)[-int(replay_limit) :]
            logger_filter = self._logger_filters.get(widget, "")
            show_startup = self._show_startup.get(widget, False)

            replayed = 0
            for entry in logs_to_replay:
                # Check level filter
                if entry.level < min_level:
                    continue

                # Hide startup INFO logs when requested (still show warnings/errors)
                if (
                    not show_startup
                    and entry.category == "startup"
                    and entry.level < logging.WARNING
                ):
                    continue

                # Check logger filter
                if logger_filter and logger_filter not in entry.logger_name.lower():
                    continue

                # Write to widget - use call_from_thread if not on main thread
                try:
                    if threading.current_thread() is threading.main_thread():
                        self._write_to_widget(widget, entry.styled_message)
                    else:
                        # On background thread, schedule on main thread
                        if hasattr(widget, "app") and widget.app:
                            widget.app.call_from_thread(
                                self._write_to_widget, widget, entry.styled_message
                            )
                    replayed += 1
                except Exception as e:
                    # Skip this entry if write fails, continue with others
                    _error_logger.debug(f"Error replaying log entry to widget: {e}")
                    continue

            if replayed > 0:
                # Add separator to indicate replayed logs with more detail
                total_available = len(logs_to_replay)
                separator_msg = (
                    f"─── {replayed} previous logs replayed (from {total_available} available) ───"
                )
                separator = Text(separator_msg, style=THEME.colors.text_muted)
                try:
                    if threading.current_thread() is threading.main_thread():
                        widget.write(separator)
                    else:
                        if hasattr(widget, "app") and widget.app:
                            widget.app.call_from_thread(widget.write, separator)
                except Exception:
                    pass  # Ignore separator write errors

        except Exception as e:
            _error_logger.debug(f"Error replaying logs to widget: {e}")

    def update_widget_level(self, widget: RichLog, min_level: int) -> None:
        """Update the minimum level for a registered widget."""
        if widget in self._widgets:
            self._widgets[widget] = min_level

    def update_widget_logger_filter(self, widget: RichLog, logger_filter: str) -> None:
        """Update the logger filter pattern for a registered widget."""
        if widget in self._widgets:
            self._logger_filters[widget] = logger_filter.lower()

    def unregister_widget(self, widget: RichLog) -> None:
        """Unregister a RichLog widget."""
        self._widgets.pop(widget, None)
        self._callbacks.pop(widget, None)
        self._logger_filters.pop(widget, None)
        self._show_startup.pop(widget, None)
        # Clean up pause state
        with self._pause_lock:
            self._paused_widgets.discard(widget)
            self._paused_buffers.pop(widget, None)

    def pause_widget(self, widget: RichLog) -> None:
        """Pause log streaming to widget, buffering incoming logs.

        Args:
            widget: The RichLog widget to pause.
        """
        with self._pause_lock:
            if widget not in self._paused_widgets:
                self._paused_widgets.add(widget)
                self._paused_buffers[widget] = deque(maxlen=self.PAUSE_BUFFER_MAX)

    def resume_widget(self, widget: RichLog) -> None:
        """Resume log streaming to widget, flushing buffered logs.

        Args:
            widget: The RichLog widget to resume.
        """
        with self._pause_lock:
            if widget in self._paused_widgets:
                self._paused_widgets.discard(widget)
                buffer = self._paused_buffers.pop(widget, deque())

        # Flush buffered logs outside of lock to avoid blocking emit()
        if buffer:
            for entry in buffer:
                self._write_to_widget(widget, entry.styled_message)

    def is_widget_paused(self, widget: RichLog) -> bool:
        """Check if a widget is currently paused.

        Args:
            widget: The RichLog widget to check.

        Returns:
            True if the widget is paused.
        """
        return widget in self._paused_widgets

    def get_paused_count(self, widget: RichLog) -> int:
        """Get count of buffered logs for a paused widget.

        Args:
            widget: The RichLog widget to check.

        Returns:
            Number of logs buffered while paused.
        """
        with self._pause_lock:
            return len(self._paused_buffers.get(widget, []))

    def get_error_count(self) -> int:
        """Get count of error entries in the buffer.

        Returns:
            Number of error-level log entries.
        """
        return len(self._error_indices)

    def get_error_entries(self) -> list[LogEntry]:
        """Get all error entries from buffer.

        Returns:
            List of LogEntry objects for error-level logs.
        """
        buffer_list = list(self._log_buffer)
        buffer_len = len(buffer_list)
        return [buffer_list[i] for i in self._error_indices if i < buffer_len]

    @property
    def format_mode(self) -> str:
        """Get current format mode (compact, verbose, or structured)."""
        return self._format_mode

    @format_mode.setter
    def format_mode(self, mode: str) -> None:
        """Set format mode (compact, verbose, structured, or json)."""
        if mode in ("compact", "verbose", "structured", "json"):
            self._format_mode = mode

    def emit(self, record: logging.LogRecord) -> None:
        """Emit log to all registered widgets that accept this level.

        OPTIMIZATION: Uses lazy formatting - only formats the message style
        that is actually needed for the current format mode, rather than
        formatting all 3 styles upfront.
        """
        try:
            # OPTIMIZATION: Check if ANY widget will receive this log first
            # This allows early exit before doing any formatting work
            has_recipient = False
            for widget, min_level in list(self._widgets.items()):
                if record.levelno >= min_level:
                    logger_filter = self._logger_filters.get(widget, "")
                    if not logger_filter or logger_filter in record.name.lower():
                        if widget.is_mounted and widget.app:
                            if (
                                not self._show_startup.get(widget, False)
                                and detect_category(record.name) == "startup"
                                and record.levelno < logging.WARNING
                            ):
                                continue
                            has_recipient = True
                            break

            # Early exit if no recipients and buffer is full (oldest will be dropped anyway)
            if not has_recipient and len(self._log_buffer) >= (
                self._log_buffer.maxlen or MAX_LOG_ENTRIES
            ):
                return

            # LAZY FORMATTING: Only format the style we need for display
            # This is a major optimization - previously we formatted all 3 styles
            # for every log message regardless of which was displayed
            verbose_msg: str | None = None
            compact_msg: str | None = None

            # Format based on current mode (lazy - only format what's needed)
            if self._format_mode == "compact":
                compact_msg = self._compact_formatter.format(record)
                display_msg = compact_msg
            elif self._format_mode == "structured":
                display_msg = self._structured_formatter.format(record)
            elif self._format_mode == "json":
                # JSON mode needs some fields from record directly
                short_logger = record.name.rsplit(".", 1)[-1]
                category = detect_category(record.name)
                correlation_id = None
                domain_fields = None
                try:
                    from gpt_trader.logging.correlation import get_log_context

                    context = get_log_context()
                    if context:
                        correlation_id = context.get("correlation_id")
                        domain_fields = {k: v for k, v in context.items() if k != "correlation_id"}
                except ImportError:
                    pass

                json_struct = {
                    "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                    "logger": short_logger,
                    "level": record.levelname,
                    "category": category,
                    "message": record.getMessage(),
                }
                if correlation_id:
                    json_struct["correlation_id"] = correlation_id
                if domain_fields:
                    json_struct["context"] = domain_fields
                display_msg = json.dumps(json_struct)
            else:  # verbose (default)
                verbose_msg = self._verbose_formatter.format(record)
                display_msg = verbose_msg

            # Extract structured metadata (only what we need)
            short_logger = record.name.rsplit(".", 1)[-1]
            category = detect_category(record.name)

            # Try to get correlation context if available (lazy - only if not already fetched)
            correlation_id = None
            domain_fields = None
            if self._format_mode != "json":  # Already fetched for JSON mode above
                try:
                    from gpt_trader.logging.correlation import get_log_context

                    context = get_log_context()
                    if context:
                        correlation_id = context.get("correlation_id")
                        domain_fields = {k: v for k, v in context.items() if k != "correlation_id"}
                except ImportError:
                    pass

            # Lazy: Only get verbose_msg if we need it for JSON detection or raw_message
            if verbose_msg is None:
                verbose_msg = self._verbose_formatter.format(record)

            # Detect JSON in message for pretty-printing support
            is_json = False
            json_data = None
            # Optimization: fast check before regex
            if "{" in verbose_msg:
                json_match = self._JSON_PATTERN.search(verbose_msg)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                        is_json = True
                    except json.JSONDecodeError:
                        pass

            # Detect multi-line content (stack traces, large objects)
            is_multiline = "\n" in verbose_msg or len(verbose_msg) > 200

            # Create Rich Text object with style (not markup) to prevent injection
            # Using style= treats msg as literal text, preventing markup in log messages
            # from corrupting coloring or injecting formatting
            if record.levelno >= logging.ERROR:
                styled_msg = Text(display_msg, style=THEME.colors.error)  # Warm coral-red
            elif record.levelno >= logging.WARNING:
                styled_msg = Text(display_msg, style=THEME.colors.warning)  # Warm amber
            elif record.levelno >= logging.INFO:
                styled_msg = Text(display_msg, style=THEME.colors.success)  # Warm green
            else:
                styled_msg = Text(display_msg, style=THEME.colors.text_muted)  # Muted grey

            # Lazy: Get compact_msg for LogEntry if we don't have it yet
            if compact_msg is None:
                compact_msg = self._compact_formatter.format(record)

            # Create log entry with enhanced metadata
            log_entry = LogEntry(
                level=record.levelno,
                logger_name=record.name,
                styled_message=styled_msg,
                raw_message=verbose_msg,
                is_json=is_json,
                json_data=json_data,
                is_multiline=is_multiline,
                # AI-friendly fields
                short_logger=short_logger,
                level_name=record.levelname,
                category=category,
                correlation_id=correlation_id,
                domain_fields=domain_fields,
                compact_message=compact_msg,
            )

            # Track error positions for jump-to-error navigation
            if record.levelno >= logging.ERROR:
                self._error_indices.append(self._buffer_index)

            # Store in memory-limited buffer for replay to new widgets
            # The deque will automatically remove oldest entries when maxlen is reached
            self._log_buffer.append(log_entry)
            self._buffer_index += 1

            # Check if we're on the main thread
            is_main_thread = threading.current_thread() is threading.main_thread()

            # Write to widgets that accept this level and logger filter
            for widget, min_level in list(self._widgets.items()):
                if record.levelno >= min_level:
                    # Check logger filter (case-insensitive substring match)
                    logger_filter = self._logger_filters.get(widget, "")
                    if logger_filter and logger_filter not in record.name.lower():
                        continue  # Skip this log - doesn't match filter

                    # Hide startup INFO logs when requested (still show warnings/errors).
                    if (
                        not self._show_startup.get(widget, False)
                        and category == "startup"
                        and record.levelno < logging.WARNING
                    ):
                        continue

                    # Lifecycle guard: only write to mounted widgets with active app
                    if not widget.is_mounted or not widget.app:
                        continue

                    # Check if widget is paused - buffer the log instead of writing
                    with self._pause_lock:
                        if widget in self._paused_widgets:
                            if widget in self._paused_buffers:
                                self._paused_buffers[widget].append(log_entry)
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
    def _write_to_widget(widget: RichLog, message: Text) -> None:
        """Write Rich Text to widget on main thread."""
        widget.write(message)

    def clear_buffer(self) -> int:
        """Clear the log buffer.

        Returns:
            Number of entries cleared.
        """
        count = len(self._log_buffer)
        self._log_buffer.clear()
        return count

    @property
    def buffer_size(self) -> int:
        """Current number of entries in the buffer."""
        return len(self._log_buffer)

    @property
    def buffer_max_size(self) -> int:
        """Maximum buffer capacity."""
        return self._log_buffer.maxlen or MAX_LOG_ENTRIES

    def get_buffer_stats(self) -> dict:
        """Get statistics about the log buffer.

        Returns:
            Dictionary with buffer statistics.
        """
        return {
            "current_size": self.buffer_size,
            "max_size": self.buffer_max_size,
            "replay_count": self._replay_count,
            "widget_count": len(self._widgets),
        }
