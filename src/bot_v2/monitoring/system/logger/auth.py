"""Authentication event logging mixin."""

from __future__ import annotations

from typing import Any

from .levels import LogLevel


class AuthLoggingMixin:
    """Provide authentication/audit logging helpers."""

    def log_auth_event(
        self,
        action: str,
        provider: str,
        success: bool,
        error_code: str | None = None,
        **kwargs: Any,
    ) -> None:
        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"auth {action} ({provider}) {'ok' if success else 'failed'}"
        entry = self._create_log_entry(
            level=level,
            event_type="auth_event",
            message=message,
            action=action,
            provider=provider,
            success=success,
            error_code=error_code,
            **kwargs,
        )
        self._emit_log(entry)


__all__ = ["AuthLoggingMixin"]
