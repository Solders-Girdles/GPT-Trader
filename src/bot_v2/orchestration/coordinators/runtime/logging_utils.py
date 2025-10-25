"""Logging utilities for runtime coordination."""

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="runtime_coordinator")

__all__ = ["logger"]
