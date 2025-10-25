"""Logging helpers for strategy orchestrator."""

from bot_v2.logging import get_orchestration_logger
from bot_v2.utilities.logging import get_logger

logger = get_logger(__name__, component="strategy_orchestrator")
json_logger = get_orchestration_logger("strategy_orchestrator")

__all__ = ["logger", "json_logger"]
