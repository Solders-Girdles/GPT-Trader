"""
Legacy live-trade facade retained for backwards compatibility.

Production trading flows live in ``bot_v2.orchestration`` (Coinbase
perpetuals).  The symbols exposed here now proxy to the simulated broker
so historical demos and thin wrappers keep functioning.
"""

from typing import Any

# Import core types
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)
from bot_v2.features.live_trade.live_trade import (
    connect_broker,
    disconnect,
    get_account,
    get_account_snapshot,
    get_positions,
    get_positions_trading,
    place_order,
)

# Keep local types that don't exist in core
from bot_v2.features.live_trade.types import AccountInfo, BrokerConnection

# Module-level broker connection (singleton pattern)
_broker_connection: BrokerConnection | None = None


def execute_live_trade(
    symbol: str, action: str, quantity: int, strategy_info: dict[str, Any]
) -> dict[str, Any]:
    """
    Facade function for orchestrator compatibility.

    Args:
        symbol: Trading symbol to trade
        action: 'buy' or 'sell'
        quantity: Number of shares to trade
        strategy_info: Strategy metadata (strategy name, confidence, etc.)

    Returns:
        Dict with trade execution result
    """
    global _broker_connection

    try:
        # Connect to broker if not already connected
        if _broker_connection is None:
            _broker_connection = connect_broker()

        # Note: This needs proper implementation with actual broker methods
        # For now, return a placeholder result

        # Place order through actual broker methods
        order_side = OrderSide.BUY if action == "buy" else OrderSide.SELL
        order = place_order(
            symbol=symbol,
            side=order_side,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )

        if not order:
            raise Exception("Order placement failed")

        # Use core Order fields
        result = {
            "filled": order.status == OrderStatus.FILLED,
            "order_id": order.id,
            "filled_price": float(order.avg_fill_price) if order.avg_fill_price else 0.0,
        }

        return {
            "status": "executed" if result.get("filled", False) else "pending",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "order_id": result.get("order_id", "unknown"),
            "filled_price": result.get("filled_price", 0.0),
            "strategy": strategy_info.get("strategy", "unknown"),
            "confidence": strategy_info.get("confidence", 0.5),
            "message": (
                f"Live trade {'executed' if result.get('filled', False) else 'submitted'}: "
                f"{action} {quantity} units of {symbol}"
            ),
        }

    except Exception as e:
        return {
            "status": "error",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "error": str(e),
            "message": f"Failed to execute live trade: {e}",
        }


__all__ = [
    "connect_broker",
    "place_order",
    "get_positions",
    "get_positions_trading",
    "get_account",
    "get_account_snapshot",
    "disconnect",
    "execute_live_trade",  # Added facade function
    "BrokerConnection",
    "Order",
    "Position",
    "OrderStatus",
    "AccountInfo",
]
