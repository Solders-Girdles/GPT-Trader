"""Test helpers for trade statistics calculation module."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from gpt_trader.core import Order, OrderSide, OrderStatus, OrderType


def _create_order(
    symbol: str = "BTC-USD",
    side: str = "BUY",
    order_type: str = "MARKET",
    quantity: Decimal = Decimal("1"),
    avg_fill_price: Decimal | None = Decimal("50000"),
    filled_quantity: Decimal = Decimal("1"),
    submitted_at: datetime | None = None,
) -> Order:
    """Create a real Order instance for testing.

    Uses proper core types instead of MagicMock for better type safety.
    """
    return Order(
        id=f"test-order-{symbol}-{side}",
        symbol=symbol,
        side=OrderSide(side),
        type=OrderType(order_type),
        quantity=quantity,
        status=OrderStatus.FILLED,
        filled_quantity=filled_quantity,
        avg_fill_price=avg_fill_price,
        submitted_at=submitted_at or datetime.now(),
    )


# Keep alias for backward compatibility during migration
_create_mock_order = _create_order
