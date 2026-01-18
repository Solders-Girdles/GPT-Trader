"""Logging helpers for strategy orchestrator."""

from gpt_trader.utilities.logging import get_logger, get_runtime_logger

logger = get_logger(__name__, component="strategy_orchestrator")
json_logger = get_runtime_logger("strategy_orchestrator")

__all__ = ["logger", "json_logger"]
