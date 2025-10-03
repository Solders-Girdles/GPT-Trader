"""Log emission to console and file outputs."""

import json
import logging
from typing import Any

# Level mapping (shared with logger.py)
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


class LogEmitter:
    """Handles log output to console and file with level filtering."""

    def __init__(
        self,
        service_name: str,
        enable_console: bool,
        min_level: str,
        py_logger: logging.Logger,
    ):
        """
        Initialize log emitter.

        Args:
            service_name: Service name for logging context
            enable_console: Whether to print to console
            min_level: Minimum log level to emit (debug/info/warning/error/critical)
            py_logger: Python logger instance for file output
        """
        self.service_name = service_name
        self.enable_console = enable_console
        self.min_level = min_level
        self._py_logger = py_logger

    def emit(self, entry: dict[str, Any]) -> None:
        """
        Emit log entry to configured outputs.

        Args:
            entry: Structured log entry dictionary
        """
        # 1. Level filtering
        if not self._should_emit(entry):
            return

        # 2. Console output
        if self.enable_console:
            self._emit_console(entry)

        # 3. File output
        self._emit_file(entry)

    def _should_emit(self, entry: dict[str, Any]) -> bool:
        """
        Check if entry meets minimum level threshold.

        Args:
            entry: Log entry to check

        Returns:
            True if entry should be emitted
        """
        try:
            entry_level = _LEVEL_MAP.get(entry.get("level", "info"), logging.INFO)
            min_threshold = _LEVEL_MAP.get(self.min_level, logging.INFO)
            return entry_level >= min_threshold
        except Exception:
            # On error, emit by default (fail-open for logging)
            return True

    def _emit_console(self, entry: dict[str, Any]) -> None:
        """
        Print log entry to console as JSON line.

        Args:
            entry: Log entry to print
        """
        try:
            print(json.dumps(entry, separators=(",", ":")))
        except Exception:
            # Don't let console errors break the app
            pass

    def _emit_file(self, entry: dict[str, Any]) -> None:
        """
        Write log entry to file via Python logger.

        Args:
            entry: Log entry to write
        """
        try:
            py_level = _LEVEL_MAP.get(entry.get("level", "info"), logging.INFO)
            self._py_logger.log(py_level, json.dumps(entry, separators=(",", ":")))
        except Exception:
            # Don't let logging errors break the app
            pass
