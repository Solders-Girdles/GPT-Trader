"""
DEPRECATED: context has moved to gpt_trader.features.live_trade.orchestrator.context

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.orchestrator.context import ContextBuilderMixin
"""

import warnings

from gpt_trader.features.live_trade.orchestrator.context import ContextBuilderMixin

warnings.warn(
    "gpt_trader.orchestration.strategy_orchestrator.context is deprecated. "
    "Import from gpt_trader.features.live_trade.orchestrator.context instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ContextBuilderMixin"]
