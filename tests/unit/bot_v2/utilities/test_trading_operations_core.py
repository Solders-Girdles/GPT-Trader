"""
Tests for trading operations utilities - Core functionality.

This module tests the core TradingOperations and PositionManager classes.
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from tests.shared.mock_brokers import create_mock_broker_with_async_methods

from bot_v2.errors import ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    TimeInForce,
)
from bot_v2.utilities.trading_operations import (
    PositionManager,
    TradingOperations,
    create_position_manager,
    create_trading_operations,
)


class TestTradingOperations:
    """Test cases for TradingOperations class."""

    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker client."""
        broker = create_mock_broker_with_async_methods()
        broker.place_order.return_value = None
        broker.cancel_order.return_value = False
        broker.get_positions.return_value = []
        broker.get_account.return_value = None
        return broker

    @pytest.fixture
    def mock_risk_manager(self):
        """Create a mock risk manager."""
        return Mock()

    @pytest.fixture
    def trading_ops(self, mock_broker, mock_risk_manager):
        """Create TradingOperations instance with mocked dependencies."""
        with patch("bot_v2.utilities.trading_operations.get_error_handler") as mock_handler:
            mock_handler.return_value.with_retry.side_effect = lambda func, **kwargs: func()
            return TradingOperations(mock_broker, mock_risk_manager)

    def test_init(self, mock_broker, mock_risk_manager):
        """Test TradingOperations initialization."""
        with patch("bot_v2.utilities.trading_operations.get_error_handler"):
            ops = TradingOperations(mock_broker, mock_risk_manager)

            assert ops.broker == mock_broker
            assert ops.risk_manager == mock_risk_manager
            assert ops.error_handler is not None

    def test_place_order_success(self, trading_ops, mock_broker):
        """Test successful order placement."""
        # Setup mock order
        mock_order = Mock()
        mock_order.id = "order_123"
        mock_broker.place_order.return_value = mock_order

        # Place order
        result = trading_ops.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )

        # Verify result
        assert result == mock_order
        mock_broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
            limit_price=None,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
        )

    def test_place_order_validation_error(self, trading_ops):
        """Test order placement with validation error."""
        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="",  # Invalid empty symbol
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.MARKET,
            )

    def test_place_order_limit_order_requires_price(self, trading_ops):
        """Test that limit orders require a limit price."""
        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.LIMIT,
                limit_price=None,
            )

    def test_place_order_stop_order_requires_price(self, trading_ops):
        """Test that stop orders require a stop price."""
        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.STOP,
                stop_price=None,
            )

    def test_place_order_stop_limit_requires_both_prices(self, trading_ops):
        """Test that stop-limit orders require both limit and stop prices."""
        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.STOP_LIMIT,
                limit_price=Decimal("50000"),
                stop_price=None,
            )

    def test_place_order_broker_returns_none(self, trading_ops, mock_broker):
        """Test handling when broker returns None for order placement."""
        mock_broker.place_order.return_value = None

        result = trading_ops.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )

        assert result is None

    def test_cancel_order_success(self, trading_ops, mock_broker):
        """Test successful order cancellation."""
        mock_broker.cancel_order.return_value = True

        result = trading_ops.cancel_order("order_123")

        assert result is True
        mock_broker.cancel_order.assert_called_once_with("order_123")

    def test_cancel_order_invalid_id(self, trading_ops):
        """Test order cancellation with invalid ID."""
        result = trading_ops.cancel_order("")  # Empty string

        assert result is False

    def test_cancel_order_non_string_id(self, trading_ops):
        """Test order cancellation with non-string ID."""
        result = trading_ops.cancel_order(123)  # Not a string

        assert result is False

    def test_get_positions_success(self, trading_ops, mock_broker):
        """Test successful position retrieval."""
        mock_positions = [Mock(), Mock()]
        mock_broker.get_positions.return_value = mock_positions

        result = trading_ops.get_positions()

        assert result == mock_positions
        mock_broker.get_positions.assert_called_once()

    def test_get_positions_failure(self, trading_ops, mock_broker):
        """Test position retrieval when broker fails."""
        mock_broker.get_positions.side_effect = Exception("Network error")

        result = trading_ops.get_positions()

        assert result == []

    def test_get_account_success(self, trading_ops, mock_broker):
        """Test successful account retrieval."""
        mock_account = Mock()
        mock_account.account_id = "acc_123"
        mock_broker.get_account.return_value = mock_account

        result = trading_ops.get_account()

        assert result == mock_account
        mock_broker.get_account.assert_called_once()

    def test_get_account_failure(self, trading_ops, mock_broker):
        """Test account retrieval when broker fails."""
        mock_broker.get_account.side_effect = Exception("Network error")

        result = trading_ops.get_account()

        assert result is None

    def test_validate_order_inputs_valid(self, trading_ops):
        """Test validation with valid inputs."""
        # Should not raise any exception
        trading_ops._validate_order_inputs(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
            limit_price=None,
            stop_price=None,
            time_in_force=TimeInForce.GTC,
        )

    def test_validate_order_inputs_negative_quantity(self, trading_ops):
        """Test validation with negative quantity."""
        with pytest.raises(ValidationError):
            trading_ops._validate_order_inputs(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("-1.0"),
                order_type=OrderType.MARKET,
                limit_price=None,
                stop_price=None,
                time_in_force=TimeInForce.GTC,
            )

    def test_validate_order_inputs_zero_quantity(self, trading_ops):
        """Test validation with zero quantity."""
        with pytest.raises(ValidationError):
            trading_ops._validate_order_inputs(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("0"),
                order_type=OrderType.MARKET,
                limit_price=None,
                stop_price=None,
                time_in_force=TimeInForce.GTC,
            )


class TestPositionManager:
    """Test cases for PositionManager class."""

    @pytest.fixture
    def mock_trading_ops(self):
        """Create a mock trading operations instance."""
        trading_ops = Mock()
        trading_ops.get_positions.return_value = []
        trading_ops.place_order.return_value = Mock()
        return trading_ops

    @pytest.fixture
    def position_manager(self, mock_trading_ops):
        """Create PositionManager instance."""
        return PositionManager(mock_trading_ops)

    def test_close_all_positions_no_positions(self, position_manager, mock_trading_ops):
        """Test closing positions when there are none."""
        mock_trading_ops.get_positions.return_value = []

        result = position_manager.close_all_positions()

        assert result is True
        mock_trading_ops.get_positions.assert_called_once()

    def test_close_all_positions_success(self, position_manager, mock_trading_ops):
        """Test successfully closing positions."""
        # Setup mock positions
        mock_position1 = Mock()
        mock_position1.symbol = "BTC-USD"
        mock_position1.side = "long"
        mock_position1.quantity = Decimal("1.0")

        mock_position2 = Mock()
        mock_position2.symbol = "ETH-USD"
        mock_position2.side = "short"
        mock_position2.quantity = Decimal("10.0")

        mock_trading_ops.get_positions.return_value = [mock_position1, mock_position2]
        mock_trading_ops.place_order.return_value = Mock()

        result = position_manager.close_all_positions()

        assert result is True

        # Verify orders were placed
        assert mock_trading_ops.place_order.call_count == 2

        # Verify closing sides
        calls = mock_trading_ops.place_order.call_args_list
        assert calls[0][1]["side"] == OrderSide.SELL  # Close long position
        assert calls[0][1]["symbol"] == "BTC-USD"
        assert calls[1][1]["side"] == OrderSide.BUY  # Close short position
        assert calls[1][1]["symbol"] == "ETH-USD"

    def test_close_all_positions_partial_failure(self, position_manager, mock_trading_ops):
        """Test closing positions with some failures."""
        # Setup mock positions
        mock_position1 = Mock()
        mock_position1.symbol = "BTC-USD"
        mock_position1.side = "long"
        mock_position1.quantity = Decimal("1.0")

        mock_position2 = Mock()
        mock_position2.symbol = "ETH-USD"
        mock_position2.side = "short"
        mock_position2.quantity = Decimal("10.0")

        mock_trading_ops.get_positions.return_value = [mock_position1, mock_position2]

        # First order succeeds, second fails
        mock_trading_ops.place_order.side_effect = [Mock(), None]

        result = position_manager.close_all_positions()

        assert result is False  # Overall failure due to partial failure


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_trading_operations(self):
        """Test create_trading_operations factory function."""
        mock_broker = Mock()
        mock_risk_manager = Mock()

        with patch("bot_v2.utilities.trading_operations.get_error_handler"):
            result = create_trading_operations(mock_broker, mock_risk_manager)

            assert isinstance(result, TradingOperations)
            assert result.broker == mock_broker
            assert result.risk_manager == mock_risk_manager

    def test_create_position_manager(self):
        """Test create_position_manager factory function."""
        mock_trading_ops = Mock()

        result = create_position_manager(mock_trading_ops)

        assert isinstance(result, PositionManager)
        assert result.trading_ops == mock_trading_ops
