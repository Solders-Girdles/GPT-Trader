"""Helpers for resolving broker responses during order placement."""

from __future__ import annotations

from typing import Any

from bot_v2.features.brokerages.core.interfaces import Order
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine


async def resolve_order_from_result(
    mixin: "OrderPlacementMixin",
    exec_engine: Any,
    broker: Any,
    result: Any,
) -> Order | None:
    """Resolve broker responses into Order objects."""
    if isinstance(exec_engine, AdvancedExecutionEngine):
        return result if isinstance(result, Order) else None

    resolved = await mixin._await_if_needed(result)
    if isinstance(resolved, Order):
        return resolved
    if resolved and broker is not None:
        order_lookup = broker.get_order(resolved)
        order_lookup = await mixin._await_if_needed(order_lookup)
        return order_lookup if isinstance(order_lookup, Order) else None
    return None


__all__ = ["resolve_order_from_result"]
