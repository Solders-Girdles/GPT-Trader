"""Happy-path tests for order validation in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestOrderValidationHappyPath:
    """Test valid order request scenarios."""

    def test_valid_limit_order(
        self, security_validator: Any, sample_order_requests: dict[str, dict[str, Any]]
    ) -> None:
        """Test valid limit order passes validation."""
        order = sample_order_requests["valid_limit_order"]
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert result.is_valid
        assert result.sanitized_value == order

    def test_valid_market_order(
        self, security_validator: Any, sample_order_requests: dict[str, dict[str, Any]]
    ) -> None:
        """Test valid market order passes validation."""
        order = sample_order_requests["valid_market_order"]
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert result.is_valid
        assert result.sanitized_value == order

    def test_market_order_price_ignored(self, security_validator: Any) -> None:
        """Test market order ignores price field."""
        # Market order with price (should be ignored)
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "market",
            "price": 50000.0,  # Should be ignored
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert result.is_valid

    def test_order_validation_with_numeric_strings(self, security_validator: Any) -> None:
        """Test order validation with numeric string values."""
        order = {
            "symbol": "BTC-USD",
            "quantity": "0.001",  # String
            "order_type": "limit",
            "price": "50000.0",  # String
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should convert strings to numbers and validate
        assert result.is_valid
