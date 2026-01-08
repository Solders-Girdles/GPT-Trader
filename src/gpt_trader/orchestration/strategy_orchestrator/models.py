"""
DEPRECATED: models has moved to gpt_trader.features.live_trade.orchestrator.models

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.orchestrator.models import SymbolProcessingContext
"""

import warnings

from gpt_trader.features.live_trade.orchestrator.models import SymbolProcessingContext

warnings.warn(
    "gpt_trader.orchestration.strategy_orchestrator.models is deprecated. "
    "Import from gpt_trader.features.live_trade.orchestrator.models instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SymbolProcessingContext"]
