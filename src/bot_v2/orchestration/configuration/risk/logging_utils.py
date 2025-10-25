"""Logging helper for risk configuration."""

from bot_v2.utilities.logging import get_logger

logger = get_logger(__name__, component="risk_config_core")

__all__ = ["logger"]
