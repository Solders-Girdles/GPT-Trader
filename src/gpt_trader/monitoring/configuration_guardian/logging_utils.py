"""Shared logging utilities for the configuration guardian package."""

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="configuration_guardian")

__all__ = ["logger"]
