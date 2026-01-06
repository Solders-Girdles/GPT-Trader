"""Constants and patterns for TUI logging.

Contains:
- Memory limits: MAX_LOG_ENTRIES, DEFAULT_REPLAY_COUNT
- Logger mappings: LOGGER_CATEGORIES, LEVEL_ICONS, LOGGER_ABBREVIATIONS
- Regex patterns for domain-specific log parsing
- detect_category() function for log classification
"""

from __future__ import annotations

import logging
import re

# Memory limits for log buffer
MAX_LOG_ENTRIES = 1000  # Maximum log entries to keep in memory
DEFAULT_REPLAY_COUNT = 100  # Default number of logs to replay to new widgets


# Logger category mappings for structured metadata
LOGGER_CATEGORIES: dict[str, list[str]] = {
    "startup": [
        "app",
        "tui",
        "theme_service",
        "mode_service",
        "alert_manager",
        "responsive_manager",
    ],
    "trading": ["bot_lifecycle", "trading", "order", "execution", "strategy", "factory"],
    "risk": ["risk", "position", "portfolio"],
    "market": ["market", "price", "websocket", "coinbase"],
    "ui": ["main_screen", "ui_coordinator", "widgets"],
    "system": ["health", "config", "alert", "notifications"],
}

# Level icons for compact display
LEVEL_ICONS: dict[int, str] = {
    logging.CRITICAL: "✖",
    logging.ERROR: "✖",
    logging.WARNING: "⚠",
    logging.INFO: "ℹ",
    logging.DEBUG: "🐛",
}

# Logger name abbreviations for common long names
LOGGER_ABBREVIATIONS: dict[str, str] = {
    "strategy": "strat",
    "portfolio": "port",
    "position": "pos",
    "execution": "exec",
    "websocket": "ws",
    "coinbase": "cb",
    "bot_lifecycle": "bot",
    "ui_coordinator": "ui",
    "main_screen": "main",
    "adjustments": "adj",
    "validation": "valid",
}

# Regex patterns for domain-specific log parsing

# Debug format: "Strategy decision debug: symbol=BTC-USD ... short_ma=115.25 long_ma=113.40 ... label=neutral"
STRATEGY_DEBUG_PATTERN = re.compile(
    r"Strategy decision.*?symbol=(\S+).*?short_ma=(\d+\.?\d*).*?long_ma=(\d+\.?\d*).*?label=(\w+)"
)

# Actual decision format: "Strategy Decision for BTC-USD: BUY (momentum crossover)"
STRATEGY_DECISION_PATTERN = re.compile(r"Strategy Decision for (\S+):\s*(\w+)\s*\(([^)]+)\)")

ORDER_PATTERN = re.compile(r"(Order|order).*?(BUY|SELL|buy|sell).*?(\d+\.?\d*)\s*(\w+-\w+)")
POSITION_PATTERN = re.compile(r"(Position|position).*?(\w+-\w+).*?(\$?\d+\.?\d*)")
PRICE_PATTERN = re.compile(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)")
KEYVALUE_PATTERN = re.compile(r"(\w+)=(\S+)")


def detect_category(logger_name: str) -> str:
    """Detect log category from logger name.

    Args:
        logger_name: Full logger name (e.g., 'gpt_trader.tui.managers.bot_lifecycle')

    Returns:
        Category string (e.g., 'trading', 'startup', 'system')
    """
    short_name = logger_name.rsplit(".", 1)[-1].lower()
    for category, keywords in LOGGER_CATEGORIES.items():
        if any(keyword in short_name for keyword in keywords):
            return category
    return "general"
