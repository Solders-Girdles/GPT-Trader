import logging

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Input, Label, Log, Select, Static

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class LogWidget(Static):
    """Displays application logs with optional compact mode."""

    # Reactive counters for log statistics
    total_count: reactive[int] = reactive(0)
    error_count: reactive[int] = reactive(0)
    warning_count: reactive[int] = reactive(0)
    info_count: reactive[int] = reactive(0)

    DEFAULT_CSS = """
    LogWidget {
        layout: vertical;
        height: 1fr;
    }

    LogWidget Horizontal {
        height: auto;
        max-height: 3;
        dock: top;  /* Ensure header stays at top despite being last in DOM */
    }

    LogWidget Log {
        height: 1fr;
        min-height: 5;
        border: solid $border;
        background: $surface;
    }

    .log-expand-hint {
        height: 1;
        text-align: center;
    }

    .log-statistics {
        height: 1;
        text-align: center;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, compact_mode: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_mode = compact_mode
        self._min_level = logging.INFO  # Default filter level
        self._logger_filter = ""  # Default: show all loggers

    def compose(self) -> ComposeResult:
        # Use max_lines to limit display in compact mode
        max_lines = 50 if self.compact_mode else 1000
        # Enable highlight=True to render markup for colored log output
        log_widget = Log(id="log-stream", highlight=True, max_lines=max_lines, auto_scroll=True)
        log_widget.can_focus = True
        yield log_widget

        # Statistics label (docked to bottom of log area)
        yield Label("", id="log-statistics", classes="log-statistics")

        # Header row yielded AFTER log widget to ensure it paints ON TOP (z-index fix)
        with Horizontal(classes="header-row"):
            yield Label("📋 SYSTEM LOGS", classes="header")
            select = Select(
                [
                    ("DEBUG", logging.DEBUG),
                    ("INFO", logging.INFO),
                    ("WARNING", logging.WARNING),
                    ("ERROR", logging.ERROR),
                ],
                value=logging.INFO,
                allow_blank=False,
                id="log-level-select",
            )
            select.can_focus = True
            yield select

            filter_input = Input(
                placeholder="Filter logger (e.g. 'broker', 'tui')...",
                id="logger-filter-input",
            )
            filter_input.can_focus = True
            yield filter_input

        # Show hint only in compact mode
        if self.compact_mode:
            yield Label(
                "Press [L] to expand logs | [1] Full Logs Tab | [2] System Tab",
                classes="log-expand-hint",
            )

    def on_mount(self) -> None:
        """Register with global log handler."""
        try:
            from gpt_trader.tui.log_manager import get_tui_log_handler

            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", Log)

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
        log = self.query_one("#log-stream", Log)
        log.write_line("[#a3be8c]✓ Log system initialized[/#a3be8c]")
        log.write_line("[#a3be8c]Set log level using dropdown above[/#a3be8c]")

        # Mode-aware hint message
        if self.app and hasattr(self.app, "data_source_mode"):
            mode = self.app.data_source_mode  # type: ignore[attr-defined]
            if mode == "read_only":
                log.write_line("[#a3be8c]Observing market data (read-only mode)[/#a3be8c]")
            else:
                log.write_line("[#a3be8c]Waiting for bot to start... Press 'S' to begin[/#a3be8c]")
        else:
            log.write_line("[#a3be8c]Waiting for bot to start... Press 'S' to begin[/#a3be8c]")

    def on_unmount(self) -> None:
        """Unregister from global log handler."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        try:
            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", Log)
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
            log_display = self.query_one("#log-stream", Log)
            handler.update_widget_level(log_display, self._min_level)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle logger filter input change."""
        from gpt_trader.tui.log_manager import get_tui_log_handler

        if event.input.id == "logger-filter-input":
            self._logger_filter = event.value.lower().strip()
            # Update handler's logger filter for this widget
            handler = get_tui_log_handler()
            log_display = self.query_one("#log-stream", Log)
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
        """Update the statistics label with current counts."""
        try:
            from rich.text import Text

            from gpt_trader.tui.theme import THEME

            stats_label = self.query_one("#log-statistics", Label)

            # Build colored statistics text
            stats = Text("Total: ", style=THEME.colors.text_muted)
            stats.append(str(self.total_count), style=THEME.colors.text_primary)
            stats.append(" | Errors: ", style=THEME.colors.text_muted)
            stats.append(str(self.error_count), style=THEME.colors.error)
            stats.append(" | Warnings: ", style=THEME.colors.text_muted)
            stats.append(str(self.warning_count), style=THEME.colors.warning)
            stats.append(" | Info: ", style=THEME.colors.text_muted)
            stats.append(str(self.info_count), style=THEME.colors.success)

            stats_label.update(stats)
        except Exception:
            # Widget might not be mounted yet
            pass
