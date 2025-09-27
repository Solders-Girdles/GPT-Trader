"""Unit tests for MockBroker basic behaviors.

These are deprecated for regular CI and are marked to be skipped by default.
"""

from decimal import Decimal
import pytest

from bot_v2.orchestration.mock_broker import MockBroker
from bot_v2.features.brokerages.core.interfaces import OrderType, OrderSide, OrderStatus

# Mark entire module as relying on MockBroker (deprecated path)
pytestmark = pytest.mark.uses_mock_broker


def test_get_quote_has_bid_ask_last():
    b = MockBroker()
    b.connect()
    q = b.get_quote("BTC-PERP")
    assert q.bid < q.ask
    assert q.last > 0


def test_place_market_order_updates_positions():
    b = MockBroker()
    b.connect()
    pre = b.list_positions()
    assert len(pre) == 0

    order = b.place_order(
        symbol="BTC-PERP",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        qty=Decimal("0.01"),
    )
    assert order.status in (OrderStatus.FILLED, OrderStatus.SUBMITTED)
    pos = b.list_positions()
    assert len(pos) == 1
    assert pos[0].symbol == "BTC-PERP"
    assert pos[0].qty > 0


def test_cancel_submitted_limit_order():
    b = MockBroker()
    b.connect()
    order = b.place_order(
        symbol="ETH-PERP",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        qty=Decimal("0.02"),
        price=Decimal("2000"),
    )
    assert order.status == OrderStatus.SUBMITTED
    ok = b.cancel_order(order.id)
    assert ok is True
