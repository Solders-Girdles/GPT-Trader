"""Shared logging utilities for the Perps bot package."""

from bot_v2.logging import get_orchestration_logger
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="coinbase_trader")
json_logger = get_orchestration_logger("coinbase_trader")

__all__ = ["logger", "json_logger"]
