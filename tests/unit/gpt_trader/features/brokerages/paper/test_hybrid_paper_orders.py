"""Tests for HybridPaperBroker order execution."""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.core import OrderSide, OrderStatus, OrderType
from gpt_trader.features.brokerages.paper.hybrid import HybridPaperBroker


class TestHybridPaperBrokerOrderExecution:
    """Test HybridPaperBroker order execution."""

    @pytest.fixture
    def broker(self, broker_factory) -> HybridPaperBroker:
        """Create broker fixture with mocked client."""
        return broker_factory(
            initial_equity=Decimal("10000"),
            slippage_bps=10,
            commission_bps=Decimal("10"),
        )

    def test_place_order_buy_market(self, broker: HybridPaperBroker) -> None:
        """Test placing a buy market order."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.symbol == "BTC-USD"
        assert order.side == OrderSide.BUY
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == Decimal("0.1")
        # Price should include slippage (10 bps = 0.1%)
        assert order.avg_fill_price > Decimal("50000")

    def test_place_order_sell_market(self, broker: HybridPaperBroker) -> None:
        """Test placing a sell market order."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.side == OrderSide.SELL
        assert order.status == OrderStatus.FILLED
        # Price should include negative slippage for sells
        assert order.avg_fill_price < Decimal("50000")

    def test_place_order_updates_position(self, broker: HybridPaperBroker) -> None:
        """Test place_order updates position state."""
        broker._last_prices["BTC-USD"] = Decimal("50000")

        broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert "BTC-USD" in broker._positions
        assert broker._positions["BTC-USD"].quantity == Decimal("0.1")

    def test_place_order_updates_balance(self, broker: HybridPaperBroker) -> None:
        """Test place_order updates balance."""
        broker._last_prices["BTC-USD"] = Decimal("50000")
        initial_balance = broker._balances["USD"].total

        broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        # Balance should decrease (buy reduces cash)
        assert broker._balances["USD"].total < initial_balance

    def test_place_order_with_dict_payload(self, broker: HybridPaperBroker) -> None:
        """Test place_order handles dict payload format."""
        broker._last_prices["ETH-USD"] = Decimal("3000")

        order = broker.place_order(
            symbol_or_payload={
                "product_id": "ETH-USD",
                "side": "BUY",
                "order_configuration": {"market_market_ioc": {"base_size": "1.0"}},
            }
        )

        assert order.symbol == "ETH-USD"
        assert order.filled_quantity == Decimal("1.0")

    def test_place_order_rejected_when_no_price(self, broker: HybridPaperBroker) -> None:
        """Test order rejected when price unavailable."""
        broker._client.get_market_product_ticker.side_effect = Exception("API error")

        order = broker.place_order(
            symbol_or_payload="UNKNOWN-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        assert order.status == OrderStatus.REJECTED

    def test_cancel_order_success(self, broker: HybridPaperBroker) -> None:
        """Test successful order cancellation."""
        broker._last_prices["BTC-USD"] = Decimal("50000")
        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        result = broker.cancel_order(order.id)

        assert result is True
        assert broker._orders[order.id].status == OrderStatus.CANCELLED

    def test_cancel_order_not_found(self, broker: HybridPaperBroker) -> None:
        """Test cancellation of non-existent order."""
        result = broker.cancel_order("nonexistent_order")

        assert result is False

    def test_get_order_returns_order(self, broker: HybridPaperBroker) -> None:
        """Test getting an order by ID."""
        broker._last_prices["BTC-USD"] = Decimal("50000")
        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
        )

        result = broker.get_order(order.id)

        assert result == order

    def test_get_order_not_found(self, broker: HybridPaperBroker) -> None:
        """Test getting non-existent order."""
        result = broker.get_order("nonexistent")

        assert result is None


class TestHybridPaperBrokerLimitOrders:
    """Test HybridPaperBroker limit order and quote size edge cases."""

    @pytest.fixture
    def broker(self, broker_factory) -> HybridPaperBroker:
        """Create broker fixture with mocked client."""
        return broker_factory(slippage_bps=10)

    def test_limit_buy_uses_limit_below_slippage(self, broker: HybridPaperBroker) -> None:
        """Test limit buy uses limit price when below slippage-adjusted price."""
        broker._last_prices["BTC-USD"] = Decimal("100")

        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            limit_price=Decimal("99"),
        )

        assert order.avg_fill_price == Decimal("99")

    def test_limit_sell_uses_limit_above_slippage(self, broker: HybridPaperBroker) -> None:
        """Test limit sell uses limit price when above slippage-adjusted price."""
        broker._last_prices["BTC-USD"] = Decimal("100")

        order = broker.place_order(
            symbol_or_payload="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1"),
            limit_price=Decimal("101"),
        )

        assert order.avg_fill_price == Decimal("101")

    def test_quote_size_payload_computes_quantity(self, broker: HybridPaperBroker) -> None:
        """Test quote_size payload computes quantity correctly."""
        broker._last_prices["ETH-USD"] = Decimal("200")

        order = broker.place_order(
            symbol_or_payload={
                "product_id": "ETH-USD",
                "side": "BUY",
                "order_configuration": {"market_market_ioc": {"quote_size": "100"}},
            }
        )

        assert order.filled_quantity == Decimal("0.5")
