"""Logging helper for daily report generation."""

from bot_v2.utilities.logging import get_logger

logger = get_logger(__name__, component="daily_report")

__all__ = ["logger"]
