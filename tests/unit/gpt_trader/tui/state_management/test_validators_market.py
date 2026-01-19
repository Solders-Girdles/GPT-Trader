"""Tests for StateValidator market validation."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.tui.state_management.validators import StateValidator


class TestStateValidatorMarket:
    """Test StateValidator market validation."""

    def test_validate_market_valid_prices(self):
        """Test market validation passes for valid prices."""
        validator = StateValidator()
        mock_market = MagicMock()
        mock_market.last_prices = {"BTC-USD": "50000.00", "ETH-USD": "3000.00"}
        mock_market.last_price_update = 1234567890.0

        result = validator._validate_market(mock_market)

        assert result.valid

    def test_validate_market_negative_price_error(self):
        """Test market validation catches negative prices."""
        validator = StateValidator()
        mock_market = MagicMock()
        mock_market.last_prices = {"BTC-USD": "-100.00"}
        mock_market.last_price_update = 1234567890.0

        result = validator._validate_market(mock_market)

        assert not result.valid
        assert any("Negative price" in e.message for e in result.errors)

    def test_validate_market_invalid_price_format(self):
        """Test market validation catches invalid price format."""
        validator = StateValidator()
        mock_market = MagicMock()
        mock_market.last_prices = {"BTC-USD": "not_a_number"}
        mock_market.last_price_update = 1234567890.0

        result = validator._validate_market(mock_market)

        assert not result.valid
        assert any("Invalid price format" in e.message for e in result.errors)

    def test_validate_price_helper(self):
        """Test _validate_price helper function."""
        validator = StateValidator()

        # Valid price
        result = validator._validate_price("50000.00", "test.price")
        assert result.valid

        # Negative price
        result = validator._validate_price("-100", "test.price")
        assert not result.valid

        # Invalid format
        result = validator._validate_price("abc", "test.price")
        assert not result.valid
