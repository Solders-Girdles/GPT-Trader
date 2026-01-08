"""
DEPRECATED: orchestrator has moved to gpt_trader.features.live_trade.orchestrator.orchestrator

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.orchestrator.orchestrator import StrategyOrchestrator
"""

import warnings

from gpt_trader.features.live_trade.orchestrator.orchestrator import (
    StrategyOrchestrator,
    SymbolProcessingContext,
)

warnings.warn(
    "gpt_trader.orchestration.strategy_orchestrator.orchestrator is deprecated. "
    "Import from gpt_trader.features.live_trade.orchestrator.orchestrator instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["StrategyOrchestrator", "SymbolProcessingContext"]
