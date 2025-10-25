"""Factory helpers for unified logging."""

from __future__ import annotations

from .logger import UnifiedLogger


def get_logger(
    name: str,
    component: str | None = None,
    *,
    enable_console: bool = False,
) -> UnifiedLogger:
    """Get a unified logger instance."""
    return UnifiedLogger(name, component=component, enable_console=enable_console)


__all__ = ["get_logger"]
