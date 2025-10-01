"""
Live trading features - Risk, Execution, and Strategy components.

DEPRECATED: The legacy facade (connect_broker, place_order, etc.) has been removed.

For production trading, use the orchestration layer:
    from bot_v2.orchestration import build_bot, PerpsBot

Example:
    bot, registry = build_bot(config)
    await bot.run()

Active modules:
- risk/ - LiveRiskManager, position sizing, runtime monitoring
- strategies/ - PerpsBaselineStrategy and decision logic
- advanced_execution.py - AdvancedExecutionEngine with dynamic sizing
- indicators.py - Technical analysis functions
- guard_errors.py - Runtime guard exceptions
- risk_runtime.py - Circuit breaker and runtime state types

Legacy facade functions (connect_broker, place_order, etc.) have been
removed. Migrate to bot_v2.orchestration for modern entrypoints.
"""

# Re-export core types for convenience
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

# Export active types from this module
from bot_v2.features.live_trade.types import AccountInfo, BrokerConnection

__all__ = [
    # Core broker types
    "Order",
    "Position",
    "OrderStatus",
    "OrderSide",
    "OrderType",
    # Local types
    "AccountInfo",
    "BrokerConnection",
]
