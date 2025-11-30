import logging

from textual.app import ComposeResult
from textual.widgets import Label, Log, Static


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

            self.widget.write_log(msg)
        except Exception:
            self.handleError(record)


class LogWidget(Static):
    """Displays application logs."""

    def compose(self) -> ComposeResult:
        yield Label("SYSTEM LOGS", classes="header")
        yield Log(id="log-stream", highlight=True)

    def on_mount(self) -> None:
        # Attach handler to root logger
        handler = TuiLogHandler(self)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    def write_log(self, message: str) -> None:
        # Schedule write on the main thread
        if self.app:
            self.app.call_from_thread(self._write_line, message)

    def _write_line(self, message: str) -> None:
        try:
            log = self.query_one(Log)
            log.write_line(message)
        except Exception:
            pass
