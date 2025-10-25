"""Logging utilities for the LiveRiskManager package."""

from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="live_trade_risk")

__all__ = ["logger"]
