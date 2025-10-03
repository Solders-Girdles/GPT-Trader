"""Correlation ID context for request tracing."""

import threading
import uuid


class CorrelationContext:
    """Manages thread-local correlation IDs for distributed tracing."""

    def __init__(self) -> None:
        """Initialize correlation context with thread-local storage."""
        self._storage = threading.local()

    def set_correlation_id(self, correlation_id: str | None = None) -> None:
        """
        Set correlation ID for current thread.

        Args:
            correlation_id: Correlation ID to set, or None to auto-generate
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        self._storage.value = correlation_id

    def get_correlation_id(self) -> str:
        """
        Get current thread's correlation ID.

        Auto-generates one if not set.

        Returns:
            Correlation ID string
        """
        if not hasattr(self._storage, "value"):
            self.set_correlation_id()
        return str(self._storage.value)
