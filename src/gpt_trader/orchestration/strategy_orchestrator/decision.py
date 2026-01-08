"""
DEPRECATED: decision has moved to gpt_trader.features.live_trade.orchestrator.decision

This shim exists for backward compatibility. Update imports to:
    from gpt_trader.features.live_trade.orchestrator.decision import DecisionEngineMixin
"""

import warnings

from gpt_trader.features.live_trade.orchestrator.decision import DecisionEngineMixin

warnings.warn(
    "gpt_trader.orchestration.strategy_orchestrator.decision is deprecated. "
    "Import from gpt_trader.features.live_trade.orchestrator.decision instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DecisionEngineMixin"]
