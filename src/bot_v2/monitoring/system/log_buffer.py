"""Thread-safe circular buffer for recent log entries."""

import threading
from typing import Any


class LogBuffer:
    """Thread-safe circular buffer for storing recent log entries."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize log buffer.

        Args:
            max_size: Maximum number of log entries to retain
        """
        self._buffer: list[dict[str, Any]] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def append(self, entry: dict[str, Any]) -> None:
        """
        Add log entry to buffer, evicting oldest if full.

        Args:
            entry: Log entry dictionary to append
        """
        with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) > self._max_size:
                self._buffer.pop(0)

    def get_recent(self, count: int = 100) -> list[dict[str, Any]]:
        """
        Get most recent N log entries.

        Args:
            count: Number of recent entries to retrieve

        Returns:
            List of recent log entries (newest last)
        """
        with self._lock:
            return self._buffer[-count:] if count < len(self._buffer) else self._buffer.copy()
