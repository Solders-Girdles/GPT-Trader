"""Helpers for TradeMatcher unit tests."""

from gpt_trader.tui.types import Trade


def create_trade(
    trade_id: str,
    symbol: str,
    side: str,
    quantity: str,
    price: str,
    fee: str = "0.00",
) -> Trade:
    """Helper to create a Trade object for testing."""
    return Trade(
        trade_id=trade_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_id=f"order_{trade_id}",
        time="2024-01-15T10:00:00.000Z",
        fee=fee,
    )
