"""Tests for HybridPaperBroker executor compatibility."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import OrderSide, OrderStatus, OrderType
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerExecutorCompatibility:
    """Test HybridPaperBroker compatibility with BrokerExecutor kwarg calling convention."""

    @pytest.fixture
    def broker(self, broker_factory) -> HybridPaperBroker:
        return broker_factory(slippage_bps=0, commission_bps=Decimal("0"))

    def test_place_order_accepts_symbol_and_price_kwargs(self, broker: HybridPaperBroker) -> None:
        broker._last_prices["BTC-USD"] = Decimal("100")

        order = broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            price=Decimal("99"),
            client_id="test_client_id",
        )

        assert order.symbol == "BTC-USD"
        assert order.client_id == "test_client_id"
        assert order.avg_fill_price == Decimal("99")

    def test_reduce_only_sell_clamps_to_available_base(self, broker: HybridPaperBroker) -> None:
        broker._last_prices["BTC-USD"] = Decimal("100")
        broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1"),
        )

        order = broker.place_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2"),
            reduce_only=True,
        )

        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("1")
