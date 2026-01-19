"""Tests for StateValidator order and trade validation."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.tui.state_management.validators import StateValidator


class TestStateValidatorOrders:
    """Test StateValidator order validation."""

    def test_validate_orders_valid(self):
        """Test order validation passes for valid orders."""
        validator = StateValidator()
        mock_order = MagicMock()
        mock_order.order_id = "order-123"
        mock_order.symbol = "BTC-USD"
        mock_order.side = "BUY"
        mock_order.quantity = "1.5"
        mock_order.price = "50000.00"

        result = validator._validate_orders([mock_order])

        assert result.valid

    def test_validate_orders_missing_order_id(self):
        """Test order validation warns on missing order_id."""
        validator = StateValidator()
        mock_order = MagicMock()
        mock_order.order_id = None
        mock_order.symbol = "BTC-USD"
        mock_order.side = "BUY"

        result = validator._validate_orders([mock_order])

        assert len(result.warnings) > 0


class TestStateValidatorTrades:
    """Test StateValidator trade validation."""

    def test_validate_trades_valid(self):
        """Test trade validation passes for valid trades."""
        validator = StateValidator()
        mock_trade = MagicMock()
        mock_trade.trade_id = "trade-123"
        mock_trade.symbol = "BTC-USD"
        mock_trade.side = "BUY"
        mock_trade.quantity = "1.0"
        mock_trade.price = "50000.00"
        mock_trade.fee = "10.00"

        result = validator._validate_trades([mock_trade])

        assert result.valid

    def test_validate_trades_negative_quantity_error(self):
        """Test trade validation catches negative quantity."""
        validator = StateValidator()
        mock_trade = MagicMock()
        mock_trade.trade_id = "trade-123"
        mock_trade.symbol = "BTC-USD"
        mock_trade.quantity = "-1.0"
        mock_trade.price = "50000.00"
        mock_trade.fee = "0"

        result = validator._validate_trades([mock_trade])

        assert not result.valid


class TestStateValidatorQuantityHelper:
    """Test StateValidator quantity helper."""

    def test_validate_quantity_helper(self):
        """Test _validate_quantity helper function."""
        validator = StateValidator()

        # Valid positive quantity
        result = validator._validate_quantity("100", "test.quantity")
        assert result.valid

        # Valid negative (for shorts)
        result = validator._validate_quantity("-100", "test.quantity", allow_negative=True)
        assert result.valid

        # Invalid negative when not allowed
        result = validator._validate_quantity("-100", "test.quantity", allow_negative=False)
        assert not result.valid
