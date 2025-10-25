"""Context managers for logging operations."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .logger import StructuredLogger, UnifiedLogger


@contextmanager
def log_operation(
    operation: str,
    logger: StructuredLogger | logging.Logger | None = None,
    level: int = logging.INFO,
    **context: Any,
) -> Iterator[None]:
    """Context manager for logging operation start/end with timing."""
    if logger is None:
        logger = UnifiedLogger("operation")
    elif isinstance(logger, logging.Logger):
        logger = UnifiedLogger(logger.name)

    start_time = time.time()
    logger.log(level, f"Started {operation}", operation=operation, **context)
    try:
        yield
    finally:
        duration_ms = (time.time() - start_time) * 1000
        logger.log(
            level,
            f"Completed {operation}",
            operation=operation,
            duration_ms=round(duration_ms, 2),
            **context,
        )


__all__ = ["log_operation"]
