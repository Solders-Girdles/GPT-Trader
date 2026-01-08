"""Logging helpers for strategy orchestrator."""

from gpt_trader.logging import get_orchestration_logger
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="strategy_orchestrator")
json_logger = get_orchestration_logger("strategy_orchestrator")

__all__ = ["logger", "json_logger"]
