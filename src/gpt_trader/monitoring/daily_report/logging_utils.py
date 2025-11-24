"""Logging helper for daily report generation."""

from gpt_trader.utilities import get_logger

logger = get_logger(__name__, component="daily_report")

__all__ = ["logger"]
