"""Decorators for logging execution metadata."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from .logger import UnifiedLogger

ExecP = ParamSpec("ExecP")
ExecR = TypeVar("ExecR")


def log_execution(
    *,
    operation: str,
    logger: UnifiedLogger | None = None,
    level: int = 20,
) -> Callable[[Callable[ExecP, ExecR]], Callable[ExecP, ExecR]]:
    """Decorator to log function execution start/end and duration."""

    def decorator(func: Callable[ExecP, ExecR]) -> Callable[ExecP, ExecR]:
        log = logger or UnifiedLogger(func.__module__)

        @wraps(func)
        def wrapper(*args: ExecP.args, **kwargs: ExecP.kwargs) -> ExecR:
            start_time = time.time()
            log.log(level, f"Executing {operation}", operation=operation)
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                duration_ms = (time.time() - start_time) * 1000
                log.error(
                    f"{operation} failed: {exc}",
                    operation=operation,
                    duration_ms=round(duration_ms, 2),
                    exc_info=True,
                )
                raise
            duration_ms = (time.time() - start_time) * 1000
            log.log(
                level,
                f"Completed {operation}",
                operation=operation,
                duration_ms=round(duration_ms, 2),
            )
            return result

        return wrapper

    return decorator


__all__ = ["log_execution"]
