"""Enhanced logging system for GPT-Trader.

This module provides:
- Structured JSON logging
- Trade event tracking
- Performance metrics logging
- Log aggregation and analysis
- Context-aware logging
"""

# Import from existing logging module for backward compatibility
import logging
import os
import sys

# Import enhanced logging components
from .log_aggregator import (
    LogAggregator,
    LogEntry,
)
from .structured_logger import (
    LogContext,
    LogFormat,
    PerformanceLogger,
    StructuredFormatter,
    StructuredLogger,
    TradeLogger,
    get_structured_logger,
)

_FMT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def _ensure_root_config() -> None:
    """
    Configure a single stdout handler on the root logger exactly once.

    We do not 'force=True' by default to avoid clobbering pytest or user configs.
    If you ever see silence again due to an upstream logger stealing handlers,
    set LOG_FORCE=1 to override.
    """
    root = logging.getLogger()
    if root.handlers and os.getenv("LOG_FORCE", "0") != "1":
        return

    # Clear existing handlers if forcing.
    if root.handlers and os.getenv("LOG_FORCE", "0") == "1":
        for h in list(root.handlers):
            root.removeHandler(h)

    # Use centralized config if available; fall back to env
    try:
        from bot.config import get_config  # local import to avoid cycles

        level_name = getattr(get_config().logging, "level", "INFO")
        level = getattr(logging, str(level_name).upper(), logging.INFO)
    except Exception:
        # Fallback to environment variable for bootstrap scenarios
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT))

    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    _ensure_root_config()
    logger = logging.getLogger(name)
    logger.propagate = True

    # Ensure the logger has at least one handler
    if not logger.handlers:
        # Add a handler if none exists
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)

    return logger


__all__ = [
    # Backward compatibility
    "get_logger",
    "_ensure_root_config",
    # Structured logging
    "StructuredLogger",
    "StructuredFormatter",
    "TradeLogger",
    "PerformanceLogger",
    "LogFormat",
    "LogContext",
    "get_structured_logger",
    # Log aggregation
    "LogEntry",
    "LogAggregator",
]
