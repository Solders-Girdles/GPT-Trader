import logging

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, Log, Select, Static

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="tui")


class LogWidget(Static):
    """Displays application logs with optional compact mode."""

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
    """

    def __init__(self, compact_mode: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_mode = compact_mode
        self._min_level = logging.INFO  # Default filter level

    def compose(self) -> ComposeResult:
        # Use max_lines to limit display in compact mode
        max_lines = 50 if self.compact_mode else 1000
        # Enable highlight=True to render markup for colored log output
        log_widget = Log(id="log-stream", highlight=True, max_lines=max_lines, auto_scroll=True)
        log_widget.can_focus = True
        yield log_widget

        # Header row yielded AFTER log widget to ensure it paints ON TOP (z-index fix)
        with Horizontal(classes="header-row"):
            yield Label("📋 SYSTEM LOGS", classes="header")
            yield Select(
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

            handler.register_widget(log_display, self._min_level)

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
