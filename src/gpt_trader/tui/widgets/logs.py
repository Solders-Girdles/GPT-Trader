import logging
import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Input, Label, RichLog, Select, Static

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class LogWidget(Static):
    """Displays application logs with enhanced debugging features.

    Features:
    - Pause/resume log streaming with buffering
    - Vim-style navigation (j/k, Ctrl+d/u, g/G)
    - Jump to next/previous error (n/N)
    - Configurable timestamp display
    - Log level and logger name filtering
    """

    # Keyboard bindings for navigation and control
    BINDINGS = [
        # Pause/resume
        Binding("space", "toggle_pause", "Pause/Resume", show=True),
        Binding("p", "toggle_pause", "Pause", show=False),
        # Format toggle
        Binding("f", "cycle_format", "Format", show=True),
        # Line navigation
        Binding("j", "scroll_down_line", "Down", show=False),
        Binding("k", "scroll_up_line", "Up", show=False),
        # Page navigation
        Binding("ctrl+d", "scroll_half_page_down", "Page Down", show=False),
        Binding("ctrl+u", "scroll_half_page_up", "Page Up", show=False),
        # Jump navigation
        Binding("g", "scroll_to_top", "Top", show=False),
        Binding("G", "scroll_to_bottom", "Bottom", show=False),
        # Error navigation
        Binding("n", "next_error", "Next Error", show=True),
        Binding("N", "previous_error", "Prev Error", show=False),
        # Timestamp toggle
        Binding("t", "cycle_timestamp_format", "Timestamps", show=True),
        # Startup section toggle
        Binding("s", "toggle_startup_section", "Startup", show=False),
    ]

    # Reactive counters for log statistics
    total_count: reactive[int] = reactive(0)
    error_count: reactive[int] = reactive(0)
    warning_count: reactive[int] = reactive(0)
    info_count: reactive[int] = reactive(0)

    # Pause state
    is_paused: reactive[bool] = reactive(False)
    paused_count: reactive[int] = reactive(0)

    # Timestamp display settings
    show_timestamps: reactive[bool] = reactive(False)
    timestamp_format: reactive[str] = reactive("relative")  # relative | absolute | both

    # Format mode: compact (default), verbose, or structured (for AI)
    format_mode: reactive[str] = reactive("compact")

    # Startup section state
    startup_collapsed: reactive[bool] = reactive(True)
    startup_count: reactive[int] = reactive(0)

    # Use percentage-based child sizing - theme styles in logs.tcss
    SCOPED_CSS = False

    DEFAULT_CSS = """
    LogWidget {
        layout: vertical;
        height: 1fr;
    }

    LogWidget .log-header {
        height: 1;
        max-height: 1;
        align-vertical: middle;
        padding: 0 1;
    }

    LogWidget .log-title {
        width: 5;
        text-style: bold;
    }

    LogWidget .log-level-select {
        width: 12;
        height: 1;
        margin: 0 1;
    }

    LogWidget .log-filter {
        width: 1fr;
        min-width: 8;
        max-width: 25;
        height: 1;
    }

    LogWidget .log-sep {
        width: 1;
        margin: 0 1;
    }

    LogWidget .stat {
        width: auto;
        min-width: 4;
        text-style: bold;
        margin-right: 1;
    }

    LogWidget .pause-indicator {
        color: #E0B366;
        text-style: bold;
        background: rgba(224, 179, 102, 0.2);
        padding: 0 1;
        display: none;
    }

    LogWidget .pause-indicator.visible {
        display: block;
    }

    LogWidget RichLog {
        height: 1fr;
        min-height: 5;
    }

    .log-expand-hint {
        height: 1;
        text-align: center;
    }
    """

    def __init__(self, compact_mode: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_mode = compact_mode
        self._min_level = logging.INFO  # Default filter level
        self._logger_filter = ""  # Default: show all loggers
        self._current_error_index = -1  # For error navigation

    def compose(self) -> ComposeResult:
        # Compact header row with inline statistics
        with Horizontal(classes="log-header"):
            yield Label("LOGS", id="log-title", classes="log-title")
            yield Label("", id="pause-indicator", classes="pause-indicator")

            select = Select(
                [
                    ("DEBUG", logging.DEBUG),
                    ("INFO", logging.INFO),
                    ("WARN", logging.WARNING),
                    ("ERROR", logging.ERROR),
                ],
                value=logging.INFO,
                allow_blank=False,
                id="log-level-select",
                classes="log-level-select",
            )
            select.can_focus = True
            yield select

            filter_input = Input(
                placeholder="filter...",
                id="logger-filter-input",
                classes="log-filter",
            )
            filter_input.can_focus = True
            yield filter_input

            # Inline statistics
            yield Static("│", classes="log-sep")
            yield Label("E:0", id="stat-errors", classes="stat stat-error")
            yield Label("W:0", id="stat-warnings", classes="stat stat-warning")
            yield Label("I:0", id="stat-info", classes="stat stat-info")

        # Use max_lines to limit display in compact mode
        max_lines = 100 if self.compact_mode else 500
        # RichLog renders Rich Text directly, so we can display colored log output safely
        log_widget = RichLog(
            id="log-stream",
            max_lines=max_lines,
            auto_scroll=True,
            markup=False,
            highlight=False,
            wrap=False,
        )
        log_widget.can_focus = True
        yield log_widget

        # Show hint only in compact mode
        if self.compact_mode:
            yield Label(
                "Press [L] to expand logs | [1] Full Logs | [2] System",
                classes="log-expand-hint",
            )

    def on_mount(self) -> None:
        """Register with global log handler."""
        try:
            from gpt_trader.tui.log_manager import get_tui_log_handler

            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", RichLog)

            logger.debug(
                f"LogWidget registering: app_available={self.app is not None}, "
                f"widget_id={id(log_display)}"
            )

            # Register with counter callback for statistics tracking
            handler.register_widget(
                log_display, self._min_level, on_log_callback=self.increment_counter
            )

            logger.debug(
                f"LogWidget registered successfully: total_widgets={len(handler._widgets)}"
            )

            # Write startup messages after widget is ready
            self.call_after_refresh(self._write_startup_messages)

        except Exception as e:
            logger.error(f"Failed to register LogWidget with handler: {e}", exc_info=True)
            # Notify user that logs may not work
            if self.app:
                self.app.notify(
                    "Log widget initialization failed. Logs may not display.",
                    severity="warning",
                    timeout=5,
                )

    def _write_startup_messages(self) -> None:
        """Write startup messages to verify log widget is working."""
        from rich.text import Text

        log = self.query_one("#log-stream", RichLog)
        log.write(Text("✓ Log system initialized", style="#a3be8c"))
        log.write(Text("Set log level using dropdown above", style="#a3be8c"))

        # Mode-aware hint message
        if self.app and hasattr(self.app, "data_source_mode"):
            mode = self.app.data_source_mode  # type: ignore[attr-defined]
            if mode == "read_only":
                log.write(Text("Observing market data (read-only mode)", style="#a3be8c"))
            else:
                log.write(Text("Waiting for bot to start... Press 'S' to begin", style="#a3be8c"))
        else:
            log.write(Text("Waiting for bot to start... Press 'S' to begin", style="#a3be8c"))

    def on_unmount(self) -> None:
        """Unregister from global log handler."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        try:
            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", RichLog)
            handler.unregister_widget(log_display)
        except Exception:
            # Widget may already be cleaned up
            pass

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle log level selection change."""
        from typing import cast

        from textual.widgets._select import NoSelection

        from gpt_trader.tui.log_manager import get_tui_log_handler

        if (
            event.select.id == "log-level-select"
            and event.value is not None
            and not isinstance(event.value, NoSelection)
        ):
            self._min_level = int(cast(int, event.value))
            # Update handler's level filter for this widget
            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", RichLog)
            handler.update_widget_level(log_display, self._min_level)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle logger filter input change."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        if event.input.id == "logger-filter-input":
            self._logger_filter = event.value.lower().strip()
            # Update handler's logger filter for this widget
            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", RichLog)
            handler.update_widget_logger_filter(log_display, self._logger_filter)

    def increment_counter(self, level: int) -> None:
        """
        Increment log counters based on level.

        Args:
            level: Logging level (logging.ERROR, logging.WARNING, etc.)
        """
        self.total_count += 1
        if level >= logging.ERROR:
            self.error_count += 1
        elif level >= logging.WARNING:
            self.warning_count += 1
        elif level >= logging.INFO:
            self.info_count += 1

    def watch_total_count(self) -> None:
        """Update statistics display when counters change."""
        self._update_statistics()

    def watch_error_count(self) -> None:
        """Update statistics display when error count changes."""
        self._update_statistics()

    def watch_warning_count(self) -> None:
        """Update statistics display when warning count changes."""
        self._update_statistics()

    def watch_info_count(self) -> None:
        """Update statistics display when info count changes."""
        self._update_statistics()

    def _update_statistics(self) -> None:
        """Update inline statistics labels with current counts."""
        try:
            self.query_one("#stat-errors", Label).update(f"E:{self.error_count}")
            self.query_one("#stat-warnings", Label).update(f"W:{self.warning_count}")
            self.query_one("#stat-info", Label).update(f"I:{self.info_count}")
        except Exception:
            # Widget might not be mounted yet
            pass

    def set_level(self, level: int) -> None:
        """Programmatically set log level (for keyboard shortcuts).

        Args:
            level: Logging level (e.g., logging.DEBUG, logging.INFO)
        """
        from gpt_trader.tui.log_manager import get_tui_log_handler

        self._min_level = level
        try:
            # Update the select widget to reflect new level
            select = self.query_one("#log-level-select", Select)
            select.value = level

            # Update the handler's level filter
            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", RichLog)
            handler.update_widget_level(log_display, level)
        except Exception:
            pass

    # ==================== Pause/Resume Actions ====================

    def action_toggle_pause(self) -> None:
        """Toggle pause state for log streaming."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        handler = get_tui_log_handler()
        log_display = self.query_one("#log-stream", RichLog)

        if self.is_paused:
            handler.resume_widget(log_display)
            self.is_paused = False
            self._update_pause_indicator()
            self.notify("Log streaming resumed", timeout=2)
        else:
            handler.pause_widget(log_display)
            self.is_paused = True
            self._update_pause_indicator()
            self.notify("Paused - press Space to resume", timeout=2)

    def watch_is_paused(self) -> None:
        """Update pause indicator when pause state changes."""
        self._update_pause_indicator()

    def _update_pause_indicator(self) -> None:
        """Update the pause indicator label in the header."""
        try:
            from gpt_trader.tui.log_manager import get_tui_log_handler

            indicator = self.query_one("#pause-indicator", Label)
            if self.is_paused:
                handler = get_tui_log_handler()
                log_display = self.query_one("#log-stream", RichLog)
                count = handler.get_paused_count(log_display)
                indicator.update(f"PAUSED ({count} new)")
                indicator.add_class("visible")
            else:
                indicator.update("")
                indicator.remove_class("visible")
        except Exception:
            pass

    # ==================== Vim-Style Navigation Actions ====================

    def action_scroll_down_line(self) -> None:
        """Scroll down one line."""
        log_display = self.query_one("#log-stream", RichLog)
        log_display.scroll_down(animate=False)

    def action_scroll_up_line(self) -> None:
        """Scroll up one line and disable auto-scroll."""
        log_display = self.query_one("#log-stream", RichLog)
        log_display.scroll_up(animate=False)
        log_display.auto_scroll = False

    def action_scroll_half_page_down(self) -> None:
        """Scroll down half a page."""
        log_display = self.query_one("#log-stream", RichLog)
        lines = max(1, log_display.size.height // 2)
        for _ in range(lines):
            log_display.scroll_down(animate=False)

    def action_scroll_half_page_up(self) -> None:
        """Scroll up half a page and disable auto-scroll."""
        log_display = self.query_one("#log-stream", RichLog)
        lines = max(1, log_display.size.height // 2)
        for _ in range(lines):
            log_display.scroll_up(animate=False)
        log_display.auto_scroll = False

    def action_scroll_to_top(self) -> None:
        """Scroll to top of logs and disable auto-scroll."""
        log_display = self.query_one("#log-stream", RichLog)
        log_display.scroll_home(animate=False)
        log_display.auto_scroll = False

    def action_scroll_to_bottom(self) -> None:
        """Scroll to bottom of logs and re-enable auto-scroll."""
        log_display = self.query_one("#log-stream", RichLog)
        log_display.scroll_end(animate=False)
        log_display.auto_scroll = True

    # ==================== Error Navigation Actions ====================

    def action_next_error(self) -> None:
        """Jump to the next error in the log."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        handler = get_tui_log_handler()
        error_count = handler.get_error_count()

        if error_count == 0:
            self.notify("No errors in log", timeout=2)
            return

        self._current_error_index = (self._current_error_index + 1) % error_count
        self.notify(f"Error {self._current_error_index + 1}/{error_count}", timeout=2)

        # Scroll to approximate position based on error index
        # Since RichLog doesn't expose line positions, scroll proportionally
        log_display = self.query_one("#log-stream", RichLog)
        log_display.auto_scroll = False

    def action_previous_error(self) -> None:
        """Jump to the previous error in the log."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        handler = get_tui_log_handler()
        error_count = handler.get_error_count()

        if error_count == 0:
            self.notify("No errors in log", timeout=2)
            return

        self._current_error_index = (self._current_error_index - 1) % error_count
        self.notify(f"Error {self._current_error_index + 1}/{error_count}", timeout=2)

        log_display = self.query_one("#log-stream", RichLog)
        log_display.auto_scroll = False

    # ==================== Timestamp Actions ====================

    def action_cycle_timestamp_format(self) -> None:
        """Cycle through timestamp display formats."""
        formats = ["off", "relative", "absolute", "both"]
        current_idx = (
            formats.index(self.timestamp_format) if self.timestamp_format in formats else 0
        )
        next_idx = (current_idx + 1) % len(formats)
        self.timestamp_format = formats[next_idx]

        if self.timestamp_format == "off":
            self.show_timestamps = False
            self.notify("Timestamps: off", timeout=2)
        else:
            self.show_timestamps = True
            self.notify(f"Timestamps: {self.timestamp_format}", timeout=2)

    def format_timestamp(self, entry_timestamp: float) -> str:
        """Format a timestamp based on current display settings.

        Args:
            entry_timestamp: Unix timestamp of the log entry.

        Returns:
            Formatted timestamp string.
        """
        now = time.time()
        if self.timestamp_format == "relative":
            delta = now - entry_timestamp
            if delta < 60:
                return f"{delta:.0f}s ago"
            elif delta < 3600:
                return f"{delta / 60:.0f}m ago"
            else:
                return f"{delta / 3600:.1f}h ago"
        elif self.timestamp_format == "absolute":
            return time.strftime("%H:%M:%S", time.localtime(entry_timestamp))
        elif self.timestamp_format == "both":
            absolute = time.strftime("%H:%M:%S", time.localtime(entry_timestamp))
            delta = now - entry_timestamp
            return f"{absolute} ({delta:.0f}s ago)"
        return ""

    # ==================== Format Mode Actions ====================

    def action_cycle_format(self) -> None:
        """Cycle through log format modes (compact, verbose, structured)."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        modes = ["compact", "verbose", "structured"]
        current_idx = modes.index(self.format_mode) if self.format_mode in modes else 0
        next_idx = (current_idx + 1) % len(modes)
        self.format_mode = modes[next_idx]

        # Update the handler's format mode
        handler = get_tui_log_handler()
        handler.format_mode = self.format_mode

        # Show description of current mode
        mode_descriptions = {
            "compact": "Compact: [logger] ✓ message",
            "verbose": "Verbose: timestamp - full.logger - LEVEL - message",
            "structured": "Structured: JSON for AI agents",
        }
        self.notify(mode_descriptions.get(self.format_mode, self.format_mode), timeout=3)

    # ==================== Startup Section Actions ====================

    def action_toggle_startup_section(self) -> None:
        """Toggle startup log section expand/collapse."""
        self.startup_collapsed = not self.startup_collapsed
        if self.startup_collapsed:
            self.notify("Startup logs collapsed", timeout=2)
        else:
            self.notify("Startup logs expanded", timeout=2)
