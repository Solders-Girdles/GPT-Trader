"""Singleton log manager for TUI.

This module provides a singleton logging handler that distributes logs to all
active LogWidgets in the TUI. The handler is attached once at app startup and
shared by all LogWidget instances.

Features:
- Memory-limited log buffer using deque for automatic cleanup
- Log history replay when new widgets register
- Configurable buffer size and replay behavior
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

from gpt_trader.tui.theme import THEME

if TYPE_CHECKING:
    from textual.widgets import RichLog

# Memory limits for log buffer
MAX_LOG_ENTRIES = 1000  # Maximum log entries to keep in memory
DEFAULT_REPLAY_COUNT = 100  # Default number of logs to replay to new widgets

# Separate logger for TuiLogHandler errors (not attached to TuiLogHandler to avoid recursion)
_error_logger = logging.getLogger("tui.log_handler.errors")
_error_logger.propagate = False  # Prevent recursion through root logger's TUI handler
_error_logger.addHandler(logging.NullHandler())  # Suppress "no handler" warnings


# Logger category mappings for structured metadata
LOGGER_CATEGORIES: dict[str, list[str]] = {
    "startup": [
        "app",
        "tui",
        "theme_service",
        "mode_service",
        "alert_manager",
        "responsive_manager",
    ],
    "trading": ["bot_lifecycle", "trading", "order", "execution", "strategy", "factory"],
    "risk": ["risk", "position", "portfolio"],
    "market": ["market", "price", "websocket", "coinbase"],
    "ui": ["main_screen", "ui_coordinator", "widgets"],
    "system": ["health", "config", "alert", "notifications"],
}

# Level icons for compact display
# Level icons for compact display
LEVEL_ICONS: dict[int, str] = {
    logging.CRITICAL: "✖",
    logging.ERROR: "✖",
    logging.WARNING: "⚠",
    logging.INFO: "ℹ",
    logging.DEBUG: "🐛",
}


def detect_category(logger_name: str) -> str:
    """Detect log category from logger name.

    Args:
        logger_name: Full logger name (e.g., 'gpt_trader.tui.managers.bot_lifecycle')

    Returns:
        Category string (e.g., 'trading', 'startup', 'system')
    """
    short_name = logger_name.rsplit(".", 1)[-1].lower()
    for category, keywords in LOGGER_CATEGORIES.items():
        if any(keyword in short_name for keyword in keywords):
            return category
    return "general"


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


class CompactTuiFormatter(logging.Formatter):
    """Compact formatter for TUI with icons and short logger names.

    Format: [short_logger] icon message

    Example outputs:
    - [bot_lifecycle] ✓ Mode switch completed
    - [factory] ⚠ Failed to create strategy
    - [tui] ✗ Widget initialization failed
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record in compact style with icon."""
        icon = LEVEL_ICONS.get(record.levelno, "·")
        short_name = record.name.rsplit(".", 1)[-1]
        message = record.getMessage()

        # Handle exceptions compactly
        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            if exc_type and exc_value:
                message = f"{message} ({exc_type.__name__}: {exc_value})"

        return f"[{short_name}] {icon} {message}"

    def format_with_timestamp(self, record: logging.LogRecord) -> str:
        """Format log record with timestamp prefix."""
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))
        compact = self.format(record)
        return f"{timestamp} {compact}"


# Logger name abbreviations for common long names
LOGGER_ABBREVIATIONS: dict[str, str] = {
    "strategy": "strat",
    "portfolio": "port",
    "position": "pos",
    "execution": "exec",
    "websocket": "ws",
    "coinbase": "cb",
    "bot_lifecycle": "bot",
    "ui_coordinator": "ui",
    "main_screen": "main",
    "adjustments": "adj",
    "validation": "valid",
}

# Regex patterns for domain-specific log parsing

# Debug format: "Strategy decision debug: symbol=BTC-USD ... short_ma=115.25 long_ma=113.40 ... label=neutral"
STRATEGY_DEBUG_PATTERN = re.compile(
    r"Strategy decision.*?symbol=(\S+).*?short_ma=(\d+\.?\d*).*?long_ma=(\d+\.?\d*).*?label=(\w+)"
)

# Actual decision format: "Strategy Decision for BTC-USD: BUY (momentum crossover)"
STRATEGY_DECISION_PATTERN = re.compile(
    r"Strategy Decision for (\S+):\s*(\w+)\s*\(([^)]+)\)"
)

ORDER_PATTERN = re.compile(r"(Order|order).*?(BUY|SELL|buy|sell).*?(\d+\.?\d*)\s*(\w+-\w+)")
POSITION_PATTERN = re.compile(r"(Position|position).*?(\w+-\w+).*?(\$?\d+\.?\d*)")
PRICE_PATTERN = re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)")
KEYVALUE_PATTERN = re.compile(r"(\w+)=(\S+)")


def _abbreviate_logger(name: str, max_len: int = 10) -> str:
    """Abbreviate logger name to fit in fixed width.

    Args:
        name: Full logger name (e.g., 'gpt_trader.features.strategy')
        max_len: Maximum length for abbreviated name

    Returns:
        Abbreviated name, right-padded to max_len
    """
    short = name.rsplit(".", 1)[-1]

    # Apply known abbreviations
    abbrev = LOGGER_ABBREVIATIONS.get(short, short)

    # Truncate if still too long
    if len(abbrev) > max_len:
        abbrev = abbrev[: max_len - 1] + "…"

    return abbrev.ljust(max_len)


def _format_number(value: str, decimals: int = 2) -> str:
    """Format a numeric string to specified decimal places.

    Args:
        value: String representation of number
        decimals: Number of decimal places

    Returns:
        Formatted number string
    """
    try:
        num = float(value)
        if num >= 1000:
            return f"{num:,.{decimals}f}"
        return f"{num:.{decimals}f}"
    except (ValueError, TypeError):
        return value


def _extract_key_values(message: str) -> dict[str, str]:
    """Extract key=value pairs from log message.

    Args:
        message: Log message string

    Returns:
        Dictionary of extracted key-value pairs
    """
    return dict(KEYVALUE_PATTERN.findall(message))


class StructuredTuiFormatter(logging.Formatter):
    """Structured formatter with fixed columns for easy scanning.

    Layout: TIME ICON LOGGER     | MESSAGE
            ──── ─── ────────── | ─────────────────────────────

    Features:
    - Fixed-width columns for visual alignment
    - Abbreviated logger names (max 10 chars)
    - Domain-aware message formatting (strategy, orders, positions)
    - Rounded decimals for readability
    - Key fields extracted and highlighted

    Example outputs:
        09:31 ✓ strat      BTC-USD NEUTRAL  MA 115.25/113.40
        09:31 ⚠ portfolio  Position $0.00 < share $150.00
        09:31 ✓ exec       BUY 0.05 BTC-USD @ $98,450.00
    """

    # Column widths
    TIME_WIDTH = 8  # HH:MM:SS
    ICON_WIDTH = 2  # Icon + space
    LOGGER_WIDTH = 10  # Abbreviated logger name

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured columns."""
        # Time column
        timestamp = time.strftime("%H:%M:%S", time.localtime(record.created))

        # Icon column
        icon = LEVEL_ICONS.get(record.levelno, "·")

        # Message - apply domain-specific formatting (check if system log first)
        message = self._format_message(record)
        logger_name = record.name.lower()
        
        # System/startup logs - use more descriptive logger names
        system_keywords = [
            "startup", "initializ", "mount", "ready", "connect", "disconnect",
            "error", "failed", "critical", "warning", "status", "state",
            "config", "mode", "credential", "validation", "bot", "engine",
            "app", "tui", "lifecycle", "coordinator", "service"
        ]
        is_system_log = any(keyword in logger_name or keyword in message.lower() for keyword in system_keywords)
        
        if is_system_log:
            # For system logs, use more descriptive logger name (last 2 components)
            # e.g., "gpt_trader.tui.app" -> "tui.app" instead of just "app"
            parts = record.name.rsplit(".", 2)
            if len(parts) >= 2:
                logger_abbrev = parts[-2] + "." + parts[-1]
                # Truncate if still too long, but preserve more context
                if len(logger_abbrev) > self.LOGGER_WIDTH:
                    logger_abbrev = logger_abbrev[-(self.LOGGER_WIDTH-1):]
            else:
                logger_abbrev = _abbreviate_logger(record.name, self.LOGGER_WIDTH)
        else:
            # Logger column (abbreviated, fixed width) for non-system logs
            logger_abbrev = _abbreviate_logger(record.name, self.LOGGER_WIDTH)

        # Handle exceptions
        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            if exc_type and exc_value:
                exc_name = exc_type.__name__
                # Keep exception message short
                exc_msg = str(exc_value)[:50]
                if len(str(exc_value)) > 50:
                    exc_msg += "…"
                message = f"{message} [{exc_name}: {exc_msg}]"

        return f"{timestamp} {icon} {logger_abbrev:<{self.LOGGER_WIDTH}} {message}"

    def _format_message(self, record: logging.LogRecord) -> str:
        """Apply domain-specific formatting to message.

        Args:
            record: Log record to format

        Returns:
            Formatted message string
        """
        message = record.getMessage()
        logger_name = record.name.lower()

        # System/startup logs - preserve full context for better debugging
        # These logs are critical for understanding system state
        system_keywords = [
            "startup", "initializ", "mount", "ready", "connect", "disconnect",
            "error", "failed", "critical", "warning", "status", "state",
            "config", "mode", "credential", "validation", "bot", "engine"
        ]
        is_system_log = any(keyword in logger_name or keyword in message.lower() for keyword in system_keywords)
        
        # For system logs, preserve more context - don't condense as aggressively
        if is_system_log:
            # Still apply basic formatting but preserve important details
            # Remove only truly redundant prefixes, keep the message informative
            prefixes_to_remove = [
                "Strategy decision debug: ",
                "Strategy decision: ",
            ]
            for prefix in prefixes_to_remove:
                if message.startswith(prefix):
                    message = message[len(prefix) :]
                    break
            # Return full message for system logs to preserve context
            return message

        # Strategy decision logs - condense the verbose output
        if "strategy" in logger_name:
            # Actual decision format: "Strategy Decision for BTC-USD: BUY (momentum crossover)"
            decision_match = STRATEGY_DECISION_PATTERN.search(message)
            if decision_match:
                symbol, action, reason = decision_match.groups()
                # Truncate long reasons
                if len(reason) > 25:
                    reason = reason[:22] + "..."
                return f"{symbol:<8} {action.upper():<6} {reason}"

            # Debug format with MA values
            debug_match = STRATEGY_DEBUG_PATTERN.search(message)
            if debug_match:
                symbol, short_ma, long_ma, label = debug_match.groups()
                short_ma_fmt = _format_number(short_ma)
                long_ma_fmt = _format_number(long_ma)
                return f"{symbol:<8} {label.upper():<8} MA {short_ma_fmt}/{long_ma_fmt}"

        # Order logs - extract key info
        if "order" in logger_name or "execution" in logger_name:
            match = ORDER_PATTERN.search(message)
            if match:
                _, side, quantity, symbol = match.groups()
                return f"{side.upper()} {quantity} {symbol}"

        # Position logs
        if "position" in logger_name or "portfolio" in logger_name:
            # Extract and format dollar amounts
            message = PRICE_PATTERN.sub(
                lambda m: f"${_format_number(m.group(1).replace(',', ''))}",
                message,
            )
            # Shorten common prefixes
            message = message.replace("Position value ", "Pos ")
            message = message.replace("Kelly position ", "Kelly ")
            return message

        # Generic key=value formatting - round long decimals
        # Optimization: only run if '=' is present
        if "=" in message:
            kv_pairs = _extract_key_values(message)
            for key, value in kv_pairs.items():
                # Round long decimal values
                if "." in value and len(value.split(".")[-1]) > 4:
                    formatted = _format_number(value)
                    message = message.replace(f"{key}={value}", f"{key}={formatted}")

        # Remove redundant prefixes
        prefixes_to_remove = [
            "Strategy decision debug: ",
            "Strategy decision: ",
        ]
        for prefix in prefixes_to_remove:
            if message.startswith(prefix):
                message = message[len(prefix) :]
                break

        return message


class TuiLogHandler(logging.Handler):
    """Single handler that distributes logs to all active LogWidgets.

    Features:
    - Distributes logs to all registered RichLog widgets
    - Memory-limited buffer for log history
    - Replay recent logs when new widgets register
    """

    # Maximum logs to buffer while widget is paused
    PAUSE_BUFFER_MAX = 500

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
            if not hasattr(widget, 'is_mounted') or not widget.is_mounted:
                # Widget not ready yet, skip replay (will be handled by normal emit flow)
                return
            if not hasattr(widget, 'app') or not widget.app:
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
                if not show_startup and entry.category == "startup" and entry.level < logging.WARNING:
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
                        if hasattr(widget, 'app') and widget.app:
                            widget.app.call_from_thread(self._write_to_widget, widget, entry.styled_message)
                    replayed += 1
                except Exception as e:
                    # Skip this entry if write fails, continue with others
                    _error_logger.debug(f"Error replaying log entry to widget: {e}")
                    continue

            if replayed > 0:
                # Add separator to indicate replayed logs with more detail
                from gpt_trader.tui.theme import THEME

                total_available = len(logs_to_replay)
                separator_msg = f"─── {replayed} previous logs replayed (from {total_available} available) ───"
                separator = Text(separator_msg, style=THEME.colors.text_muted)
                try:
                    if threading.current_thread() is threading.main_thread():
                        widget.write(separator)
                    else:
                        if hasattr(widget, 'app') and widget.app:
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

    # Pattern to detect JSON objects in log messages
    _JSON_PATTERN = re.compile(r"\{[^{}]+\}")

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
            if not has_recipient and len(self._log_buffer) >= (self._log_buffer.maxlen or MAX_LOG_ENTRIES):
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
