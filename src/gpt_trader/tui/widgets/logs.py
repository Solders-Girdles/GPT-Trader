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
    """Displays application logs."""

    DEFAULT_CSS = """
    LogWidget {
        background: #2e3440;
        height: 100%;
    }

    LogWidget .header-row {
        height: 3;
        dock: top;
        padding: 0 1;
        background: #3b4252;
    }

    LogWidget Select {
        width: 20;
    }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-row"):
            yield Label("SYSTEM LOGS", classes="header")
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
        yield Log(id="log-stream", highlight=True)

    def on_mount(self) -> None:
        # Attach handler to root logger
        handler = TuiLogHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

        # Set initial filter level
        self._min_level = logging.INFO

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle log level selection change."""
        if event.select.id == "log-level-select" and event.value is not None:
            self._min_level = int(event.value)

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
