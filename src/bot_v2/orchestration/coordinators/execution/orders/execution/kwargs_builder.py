"""Order keyword construction helpers for decision execution."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def build_order_kwargs(
    exec_engine: Any,
    symbol: str,
    side: Any,
    quantity: Decimal,
    order_type: Any,
    reduce_only: bool,
    limit_price: Decimal | None,
    stop_price: Decimal | None,
    tif: Any,
    leverage: Any,
    product: Any,
) -> dict[str, Any]:
    from bot_v2.features.brokerages.core.interfaces import TimeInForce
    from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine

    kwargs: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "order_type": order_type,
        "reduce_only": reduce_only,
        "leverage": leverage,
    }

    if isinstance(exec_engine, AdvancedExecutionEngine):
        kwargs.update(
            {
                "limit_price": limit_price,
                "stop_price": stop_price,
                "time_in_force": tif or TimeInForce.GTC,
            }
        )
    else:
        kwargs.update(
            {
                "product": product,
                "price": limit_price,
                "stop_price": stop_price,
                "tif": tif or None,
            }
        )
    return kwargs


__all__ = ["build_order_kwargs"]
