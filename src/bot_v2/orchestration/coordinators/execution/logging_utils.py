"""Logging helpers shared across execution coordinator modules."""

from bot_v2.logging import get_orchestration_logger
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="execution_coordinator")
json_logger = get_orchestration_logger("execution_coordinator")

__all__ = ["logger", "json_logger"]
