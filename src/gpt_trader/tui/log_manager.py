"""Singleton log manager for TUI.

This module provides a singleton logging handler that distributes logs to all
active LogWidgets in the TUI. The handler is attached once at app startup and
shared by all LogWidget instances.

This module re-exports from focused submodules for backwards compatibility:
- log_constants: Constants, patterns, and category detection
- log_formatters: Formatter classes and helper functions
- log_handler: LogEntry dataclass and TuiLogHandler class

Submodules:
- log_constants: MAX_LOG_ENTRIES, DEFAULT_REPLAY_COUNT, LOGGER_CATEGORIES,
                 LEVEL_ICONS, LOGGER_ABBREVIATIONS, patterns, detect_category
- log_formatters: _abbreviate_logger, _format_number, _extract_key_values,
                  ImprovedExceptionFormatter, CompactTuiFormatter, StructuredTuiFormatter
- log_handler: LogEntry, TuiLogHandler

Features:
- Memory-limited log buffer using deque for automatic cleanup
- Log history replay when new widgets register
- Configurable buffer size and replay behavior
"""

from __future__ import annotations

import logging

# Re-export from log_constants
from gpt_trader.tui.log_constants import (
    DEFAULT_REPLAY_COUNT,
    KEYVALUE_PATTERN,
    LEVEL_ICONS,
    LOGGER_ABBREVIATIONS,
    LOGGER_CATEGORIES,
    MAX_LOG_ENTRIES,
    ORDER_PATTERN,
    POSITION_PATTERN,
    PRICE_PATTERN,
    STRATEGY_DEBUG_PATTERN,
    STRATEGY_DECISION_PATTERN,
    detect_category,
)

# Re-export from log_formatters
from gpt_trader.tui.log_formatters import (
    CompactTuiFormatter,
    ImprovedExceptionFormatter,
    StructuredTuiFormatter,
    _abbreviate_logger,
    _extract_key_values,
    _format_number,
)

# Re-export from log_handler
from gpt_trader.tui.log_handler import (
    LogEntry,
    TuiLogHandler,
)

__all__ = [
    # Constants
    "MAX_LOG_ENTRIES",
    "DEFAULT_REPLAY_COUNT",
    "LOGGER_CATEGORIES",
    "LEVEL_ICONS",
    "LOGGER_ABBREVIATIONS",
    # Regex patterns
    "STRATEGY_DEBUG_PATTERN",
    "STRATEGY_DECISION_PATTERN",
    "ORDER_PATTERN",
    "POSITION_PATTERN",
    "PRICE_PATTERN",
    "KEYVALUE_PATTERN",
    # Functions
    "detect_category",
    "_abbreviate_logger",
    "_format_number",
    "_extract_key_values",
    # Formatter classes
    "ImprovedExceptionFormatter",
    "CompactTuiFormatter",
    "StructuredTuiFormatter",
    # Handler classes
    "LogEntry",
    "TuiLogHandler",
    # Singleton functions
    "get_tui_log_handler",
    "attach_tui_log_handler",
    "detach_tui_log_handler",
]

# Global singleton instance
_tui_log_handler: TuiLogHandler | None = None


def get_tui_log_handler() -> TuiLogHandler:
    """Get or create the singleton TUI log handler."""
    global _tui_log_handler
    if _tui_log_handler is None:
        _tui_log_handler = TuiLogHandler()
    return _tui_log_handler


def attach_tui_log_handler() -> None:
    """Attach the singleton handler to root logger."""
    handler = get_tui_log_handler()
    root_logger = logging.getLogger()

    # Only attach if not already attached
    if handler not in root_logger.handlers:
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.DEBUG)


def detach_tui_log_handler() -> None:
    """Detach the singleton handler from root logger."""
    global _tui_log_handler
    if _tui_log_handler:
        logging.getLogger().removeHandler(_tui_log_handler)
        _tui_log_handler = None
