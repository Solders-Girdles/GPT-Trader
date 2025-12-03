import logging

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, Log, Select, Static


class TuiLogHandler(logging.Handler):
    """Custom logging handler that pushes logs to a Textual Log widget."""

    def __init__(self, widget: "LogWidget"):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)

            # Add color based on level
            if record.levelno >= logging.ERROR:
                msg = f"[#bf616a]{msg}[/#bf616a]"  # Nord Red
            elif record.levelno >= logging.WARNING:
                msg = f"[#ebcb8b]{msg}[/#ebcb8b]"  # Nord Yellow
            elif record.levelno >= logging.INFO:
                msg = f"[#a3be8c]{msg}[/#a3be8c]"  # Nord Green
            else:
                msg = f"[#4c566a]{msg}[/#4c566a]"  # Nord Grey

            self.widget.write_log(msg, record.levelno)
        except Exception:
            self.handleError(record)


class LogWidget(Static):
    """Displays application logs with optional compact mode."""

    DEFAULT_CSS = """
    LogWidget {
        layout: vertical;
        height: 1fr;
    }

    LogWidget Log {
        height: 1fr;
    }

    .log-expand-hint {
        height: 1;
        text-align: center;
    }
    """

    def __init__(self, compact_mode: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compact_mode = compact_mode
        self._handler: TuiLogHandler | None = None

    def compose(self) -> ComposeResult:
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

        # Use max_lines to limit display in compact mode
        max_lines = 10 if self.compact_mode else 1000
        yield Log(id="log-stream", highlight=True, max_lines=max_lines)

        # Show hint only in compact mode
        if self.compact_mode:
            yield Label(
                "Press [L] to expand logs | [1] Full Logs Tab | [2] System Tab",
                classes="log-expand-hint",
            )

    def on_mount(self) -> None:
        # Attach handler to root logger
        self._handler = TuiLogHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        self._handler.setFormatter(formatter)
        logging.getLogger().addHandler(self._handler)

        # Set initial filter level
        self._min_level = logging.INFO

    def on_unmount(self) -> None:
        """Remove handler when widget is unmounted."""
        if self._handler:
            try:
                logging.getLogger().removeHandler(self._handler)
                self._handler = None
            except Exception:
                # Logger may already be cleaned up
                pass

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle log level selection change."""
        from typing import cast

        from textual.widgets._select import NoSelection

        if (
            event.select.id == "log-level-select"
            and event.value is not None
            and not isinstance(event.value, NoSelection)
        ):
            self._min_level = int(cast(int, event.value))

    def write_log(self, message: str, level: int) -> None:
        # Schedule write on the main thread
        if self.app:
            self.app.call_from_thread(self._write_line, message, level)

    def _write_line(self, message: str, level: int) -> None:
        try:
            # Only write if level meets minimum requirement
            if level >= getattr(self, "_min_level", logging.INFO):
                log = self.query_one(Log)
                log.write_line(message)
        except Exception:
            pass
