"""
DEPRECATED: initialization has moved to gpt_trader.features.live_trade.orchestrator.initialization

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.orchestrator.initialization import StrategyInitializationMixin
"""

import warnings

from gpt_trader.features.live_trade.orchestrator.initialization import (
    StrategyInitializationMixin,
)

warnings.warn(
    "gpt_trader.orchestration.strategy_orchestrator.initialization is deprecated. "
    "Import from gpt_trader.features.live_trade.orchestrator.initialization instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["StrategyInitializationMixin"]
