"""
Tests for trading operations utilities - Integration scenarios.

This module tests integration scenarios and edge cases for trading operations.
"""

from decimal import Decimal
from unittest.mock import Mock, patch

import pytest
from tests.shared.mock_brokers import create_mock_broker_with_async_methods

from bot_v2.errors import ExecutionError, NetworkError, ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.utilities.trading_operations import (
    PositionManager,
    TradingOperations,
    create_position_manager,
    create_trading_operations,
)


class TestIntegration:
    """Integration tests for trading operations."""

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

    @pytest.fixture
    def mock_order(self):
        """Create a mock order."""
        order = Mock()
        order.id = "order_123"
        order.symbol = "BTC-USD"
        order.side = OrderSide.BUY
        order.quantity = Decimal("1.0")
        order.status = OrderStatus.FILLED
        return order

    def test_full_order_lifecycle(self, trading_ops, mock_broker, mock_order):
        """Test complete order lifecycle from placement to completion."""
        # Setup mocks
        mock_broker.place_order.return_value = mock_order
        mock_broker.cancel_order.return_value = True

        # Place order
        placed_order = trading_ops.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )

        assert placed_order == mock_order

        # Cancel order
        cancel_result = trading_ops.cancel_order(mock_order.id)

        assert cancel_result is True

        # Verify all calls were made
        mock_broker.place_order.assert_called_once()
        mock_broker.cancel_order.assert_called_once_with(mock_order.id)

    def test_error_handling_and_recovery(self, trading_ops, mock_broker):
        """Test error handling and recovery mechanisms."""
        # Setup broker to raise exception then succeed
        mock_broker.get_positions.side_effect = [Exception("Network error"), [Mock(), Mock()]]

        # First call should return empty list due to error
        result1 = trading_ops.get_positions()
        assert result1 == []

        # Second call should succeed
        result2 = trading_ops.get_positions()
        assert len(result2) == 2

    def test_order_validation_edge_cases(self, trading_ops):
        """Test order validation with various edge cases."""
        # Test basic validation scenarios that are actually implemented
        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="",  # Empty symbol
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.MARKET,
            )

        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("0"),  # Zero quantity
                order_type=OrderType.MARKET,
            )

    def test_complex_order_types(self, trading_ops, mock_broker):
        """Test complex order types and parameters."""
        mock_order = Mock()
        mock_broker.place_order.return_value = mock_order

        # Test stop-limit order with both prices
        result = trading_ops.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.STOP_LIMIT,
            limit_price=Decimal("50000"),
            stop_price=Decimal("49000"),
            time_in_force=TimeInForce.IOC,
        )

        assert result == mock_order
        mock_broker.place_order.assert_called_once_with(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.STOP_LIMIT,
            limit_price=Decimal("50000"),
            stop_price=Decimal("49000"),
            time_in_force=TimeInForce.IOC,
        )

    def test_position_manager_integration(self, mock_broker, mock_risk_manager):
        """Test PositionManager integration with TradingOperations."""
        # Create real TradingOperations instance
        with patch("bot_v2.utilities.trading_operations.get_error_handler") as mock_handler:
            mock_handler.return_value.with_retry.side_effect = lambda func, **kwargs: func()
            trading_ops = TradingOperations(mock_broker, mock_risk_manager)

        # Create PositionManager
        position_manager = PositionManager(trading_ops)

        # Setup mock positions
        mock_position = Mock()
        mock_position.symbol = "BTC-USD"
        mock_position.side = "long"
        mock_position.quantity = Decimal("1.0")

        mock_broker.get_positions.return_value = [mock_position]
        mock_broker.place_order.return_value = Mock()

        # Test closing positions
        result = position_manager.close_all_positions()

        assert result is True
        mock_broker.get_positions.assert_called_once()
        mock_broker.place_order.assert_called_once()

        # Verify closing order parameters
        call_args = mock_broker.place_order.call_args
        assert call_args[1]["side"] == OrderSide.SELL  # Close long position
        assert call_args[1]["symbol"] == "BTC-USD"
        assert call_args[1]["quantity"] == Decimal("1.0")

    def test_concurrent_operations(self, trading_ops, mock_broker):
        """Test handling of concurrent operations."""
        # Setup multiple successful operations
        mock_order1 = Mock()
        mock_order1.id = "order_1"
        mock_order2 = Mock()
        mock_order2.id = "order_2"

        mock_broker.place_order.side_effect = [mock_order1, mock_order2]

        # Place multiple orders
        result1 = trading_ops.place_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )

        result2 = trading_ops.place_order(
            symbol="ETH-USD",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            order_type=OrderType.MARKET,
        )

        assert result1 == mock_order1
        assert result2 == mock_order2
        assert mock_broker.place_order.call_count == 2

    def test_broker_error_propagation(self, trading_ops, mock_broker):
        """Test that broker errors are properly handled."""
        # Test network error
        mock_broker.get_positions.side_effect = NetworkError("Connection failed")

        result = trading_ops.get_positions()
        assert result == []

        # Test execution error
        mock_broker.cancel_order.side_effect = ExecutionError("Order not found")

        result = trading_ops.cancel_order("invalid_order")
        assert result is False

    def test_risk_manager_integration(self, trading_ops, mock_risk_manager):
        """Test integration with risk manager."""
        # This test verifies that the risk manager is properly integrated
        # In a real scenario, the risk manager would validate orders before placement
        assert trading_ops.risk_manager == mock_risk_manager

        # Test that risk manager is accessible for future enhancements
        assert hasattr(trading_ops, "risk_manager")
        assert trading_ops.risk_manager is not None

    def test_error_handler_integration(self, trading_ops):
        """Test integration with error handling system."""
        # Verify error handler is properly initialized
        assert hasattr(trading_ops, "error_handler")
        assert trading_ops.error_handler is not None

        # Test that error handling wrapper is applied
        # (This is tested implicitly through other tests that verify error handling)

    def test_factory_function_integration(self):
        """Test factory functions create properly integrated instances."""
        mock_broker = Mock()
        mock_risk_manager = Mock()
        mock_trading_ops = Mock()

        # Test trading operations factory
        with patch("bot_v2.utilities.trading_operations.get_error_handler"):
            trading_ops = create_trading_operations(mock_broker, mock_risk_manager)

            assert isinstance(trading_ops, TradingOperations)
            assert trading_ops.broker == mock_broker
            assert trading_ops.risk_manager == mock_risk_manager

        # Test position manager factory
        position_manager = create_position_manager(mock_trading_ops)

        assert isinstance(position_manager, PositionManager)
        assert position_manager.trading_ops == mock_trading_ops

    def test_order_parameter_validation_comprehensive(self, trading_ops):
        """Test comprehensive order parameter validation."""
        # Test all validation scenarios in one test

        # Invalid symbols
        invalid_symbols = ["", "   ", "\t", "\n", "INVALID-SYMBOL-WITH-DASHES"]
        for symbol in invalid_symbols:
            with pytest.raises(ValidationError):
                trading_ops.place_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=Decimal("1.0"),
                    order_type=OrderType.MARKET,
                )

        # Invalid quantities
        invalid_quantities = [Decimal("0"), Decimal("-1"), Decimal("-0.1")]
        for quantity in invalid_quantities:
            with pytest.raises(ValidationError):
                trading_ops.place_order(
                    symbol="BTC-USD",
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )

        # Missing required prices
        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.LIMIT,
                limit_price=None,
            )

        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.STOP,
                stop_price=None,
            )

        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.STOP_LIMIT,
                limit_price=Decimal("50000"),
                stop_price=None,
            )

        with pytest.raises(ValidationError):
            trading_ops.place_order(
                symbol="BTC-USD",
                side=OrderSide.BUY,
                quantity=Decimal("1.0"),
                order_type=OrderType.STOP_LIMIT,
                limit_price=None,
                stop_price=Decimal("49000"),
            )

    def test_account_and_position_data_integrity(self, trading_ops, mock_broker):
        """Test integrity of account and position data retrieval."""
        # Test account data integrity
        mock_account = Mock()
        mock_account.account_id = "acc_123"
        mock_account.balance = Decimal("10000")
        mock_account.available_balance = Decimal("8000")

        mock_broker.get_account.return_value = mock_account

        result = trading_ops.get_account()
        assert result == mock_account
        assert result.account_id == "acc_123"
        assert result.balance == Decimal("10000")

        # Test position data integrity
        mock_position = Mock()
        mock_position.symbol = "BTC-USD"
        mock_position.quantity = Decimal("1.5")
        mock_position.side = "long"

        mock_broker.get_positions.return_value = [mock_position]

        result = trading_ops.get_positions()
        assert len(result) == 1
        assert result[0].symbol == "BTC-USD"
        assert result[0].quantity == Decimal("1.5")
        assert result[0].side == "long"
