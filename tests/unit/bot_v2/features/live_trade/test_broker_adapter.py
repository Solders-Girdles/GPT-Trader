"""
Unit tests for BrokerAdapter.

Tests broker-specific parameter mapping, TimeInForce conversion,
and order submission logic.
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock
from typing import Any

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.features.live_trade.broker_adapter import BrokerAdapter


class TestBasicOrderSubmission:
    """Test basic order submission flow."""

    def test_market_order_submission(self):
        """Should submit market order with basic parameters."""
        broker = Mock()
        order = Order(
            id="order-1",
            client_id="test-1",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=None,
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=datetime.now(),
            updated_at=datetime.now(),
        )
        broker.place_order = Mock(return_value=order)

        adapter = BrokerAdapter(broker)

        result = adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            client_id="test-1",
            reduce_only=False,
            leverage=None,
        )

        assert result == order
        broker.place_order.assert_called_once()
        call_kwargs = broker.place_order.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-PERP"
        assert call_kwargs["side"] == OrderSide.BUY
        assert call_kwargs["quantity"] == Decimal("0.1")

    def test_limit_order_with_price(self):
        """Should submit limit order with limit_price."""
        broker = Mock()
        order = Order(
            id="order-2",
            client_id="test-2",
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            price=Decimal("50000"),
            stop_price=None,
            tif=TimeInForce.GTC,
            status=OrderStatus.SUBMITTED,
            filled_quantity=Decimal("0"),
            avg_fill_price=None,
            submitted_at=datetime.now(),
            updated_at=datetime.now(),
        )
        broker.place_order = Mock(return_value=order)

        adapter = BrokerAdapter(broker)

        result = adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.5"),
            client_id="test-2",
            reduce_only=False,
            leverage=None,
            limit_price=Decimal("50000"),
        )

        assert result == order
        call_kwargs = broker.place_order.call_args.kwargs
        # Should use "limit_price" parameter if broker supports it
        assert call_kwargs.get("limit_price") == Decimal("50000") or call_kwargs.get(
            "price"
        ) == Decimal("50000")

    def test_reduce_only_flag(self):
        """Should pass reduce_only flag to broker."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            client_id="test-3",
            reduce_only=True,
            leverage=None,
        )

        call_kwargs = broker.place_order.call_args.kwargs
        assert call_kwargs["reduce_only"] is True


class TestParameterMapping:
    """Test parameter name mapping for different brokers."""

    def test_limit_price_parameter_name(self):
        """Should use 'limit_price' if broker supports it."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            limit_price=Decimal("50000"),
        )

        call_kwargs = broker.place_order.call_args.kwargs
        # If broker signature has 'limit_price', it should be used
        assert "limit_price" in call_kwargs or "price" in call_kwargs

    def test_price_parameter_fallback(self):
        """Should fallback to 'price' if broker uses that instead of 'limit_price'."""

        # Create broker with 'price' parameter instead of 'limit_price'
        def place_order_with_price(
            symbol: str,
            side: OrderSide,
            order_type: OrderType,
            quantity: Decimal,
            client_id: str,
            reduce_only: bool,
            leverage: int | None,
            price: Decimal | None = None,  # Uses 'price' not 'limit_price'
        ) -> Order:
            return Mock()

        broker = Mock()
        broker.place_order = place_order_with_price

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            limit_price=Decimal("50000"),
        )

        # Should have been called (no exception)
        # The 'price' parameter should have received the limit_price value

    def test_stop_price_parameter(self):
        """Should include stop_price for stop orders."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            stop_price=Decimal("48000"),
        )

        call_kwargs = broker.place_order.call_args.kwargs
        assert "stop_price" in call_kwargs


class TestTimeInForceConversion:
    """Test TimeInForce enum/string conversion."""

    def test_time_in_force_enum_input(self):
        """Should handle TimeInForce enum input."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            limit_price=Decimal("50000"),
            time_in_force=TimeInForce.IOC,
        )

        # Should convert successfully (no exception)

    def test_time_in_force_string_input(self):
        """Should handle TimeInForce string input."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        # Call _convert_time_in_force directly
        tif_enum, tif_str = adapter._convert_time_in_force("GTC")

        assert tif_enum == TimeInForce.GTC
        assert tif_str == "gtc"

    def test_invalid_time_in_force_defaults_to_gtc(self):
        """Should default to GTC for invalid TimeInForce."""
        broker = Mock()
        adapter = BrokerAdapter(broker)

        tif_enum, tif_str = adapter._convert_time_in_force("INVALID")

        assert tif_enum == TimeInForce.GTC
        assert tif_str == "gtc"

    def test_time_in_force_parameter_name(self):
        """Should use 'time_in_force' parameter if broker supports it."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            limit_price=Decimal("50000"),
            time_in_force=TimeInForce.GTC,
        )

        call_kwargs = broker.place_order.call_args.kwargs
        # Should have either 'time_in_force' or 'tif'
        assert "time_in_force" in call_kwargs or "tif" in call_kwargs

    def test_tif_parameter_fallback(self):
        """Should use 'tif' parameter if broker uses that instead of 'time_in_force'."""

        # Create broker with 'tif' parameter
        def place_order_with_tif(
            symbol: str,
            side: OrderSide,
            order_type: OrderType,
            quantity: Decimal,
            client_id: str,
            reduce_only: bool,
            leverage: int | None,
            tif: TimeInForce = TimeInForce.GTC,  # Uses 'tif' not 'time_in_force'
        ) -> Order:
            return Mock()

        broker = Mock()
        broker.place_order = place_order_with_tif

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            limit_price=Decimal("50000"),
            time_in_force=TimeInForce.IOC,
        )

        # Should have been called (no exception)


class TestLeverageParameter:
    """Test leverage parameter handling."""

    def test_leverage_passed_to_broker(self):
        """Should pass leverage parameter to broker."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=5,
        )

        call_kwargs = broker.place_order.call_args.kwargs
        assert call_kwargs["leverage"] == 5

    def test_leverage_none_passed_to_broker(self):
        """Should pass None leverage to broker."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
        )

        call_kwargs = broker.place_order.call_args.kwargs
        assert call_kwargs["leverage"] is None


class TestErrorHandling:
    """Test error handling for broker failures."""

    def test_broker_exception_propagated(self):
        """Should propagate broker exceptions."""
        broker = Mock()
        broker.place_order = Mock(side_effect=RuntimeError("Broker error"))

        adapter = BrokerAdapter(broker)

        with pytest.raises(RuntimeError, match="Broker error"):
            adapter.submit_order(
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("0.1"),
                client_id="test",
                reduce_only=False,
                leverage=None,
            )

    def test_broker_returns_none(self):
        """Should handle broker returning None gracefully."""
        broker = Mock()
        broker.place_order = Mock(return_value=None)

        adapter = BrokerAdapter(broker)

        result = adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            client_id="test",
            reduce_only=False,
            leverage=None,
        )

        assert result is None


class TestStopOrders:
    """Test stop order specific scenarios."""

    def test_stop_order_with_stop_price(self):
        """Should submit stop order with stop_price."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=Decimal("0.2"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            stop_price=Decimal("48000"),
        )

        call_kwargs = broker.place_order.call_args.kwargs
        assert call_kwargs.get("stop_price") == Decimal("48000")

    def test_stop_limit_order_with_both_prices(self):
        """Should submit stop-limit order with both stop_price and limit_price."""
        broker = Mock()
        broker.place_order = Mock(return_value=Mock())

        adapter = BrokerAdapter(broker)

        adapter.submit_order(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.STOP_LIMIT,
            quantity=Decimal("0.3"),
            client_id="test",
            reduce_only=False,
            leverage=None,
            stop_price=Decimal("52000"),
            limit_price=Decimal("52100"),
        )

        call_kwargs = broker.place_order.call_args.kwargs
        assert call_kwargs.get("stop_price") == Decimal("52000")
        # Should have limit_price or price
        assert call_kwargs.get("limit_price") == Decimal("52100") or call_kwargs.get(
            "price"
        ) == Decimal("52100")
