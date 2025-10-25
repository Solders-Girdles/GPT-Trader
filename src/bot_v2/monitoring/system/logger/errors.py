"""Error logging mixin for production logger."""

from __future__ import annotations

from typing import Any

from .levels import LogLevel


class ErrorLoggingMixin:
    """Provide error logging helpers."""

    def log_error(self, error: Exception, context: str | None = None, **kwargs: Any) -> None:
        entry = self._create_log_entry(
            level=LogLevel.ERROR,
            event_type="error",
            message=str(error),
            error_type=type(error).__name__,
            **kwargs,
        )
        if context:
            entry["error_context"] = context
        self._emit_log(entry)


__all__ = ["ErrorLoggingMixin"]
