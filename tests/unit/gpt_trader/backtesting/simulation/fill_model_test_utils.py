from __future__ import annotations

from decimal import Decimal

from gpt_trader.core import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)


def make_order(
    symbol: str = "BTC-USD",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.MARKET,
    quantity: Decimal = Decimal("1"),
    price: Decimal | None = None,
    stop_price: Decimal | None = None,
) -> Order:
    """Helper to create test orders."""
    return Order(
        id="test-order-001",
        symbol=symbol,
        side=side,
        type=order_type,
        quantity=quantity,
        status=OrderStatus.SUBMITTED,
        price=price,
        stop_price=stop_price,
        tif=TimeInForce.GTC,
    )
