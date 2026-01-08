"""
DEPRECATED: spot_filters has moved to gpt_trader.features.live_trade.orchestrator.spot_filters

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.orchestrator.spot_filters import SpotFiltersMixin
"""

import warnings

from gpt_trader.features.live_trade.orchestrator.spot_filters import SpotFiltersMixin

warnings.warn(
    "gpt_trader.orchestration.strategy_orchestrator.spot_filters is deprecated. "
    "Import from gpt_trader.features.live_trade.orchestrator.spot_filters instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SpotFiltersMixin"]
