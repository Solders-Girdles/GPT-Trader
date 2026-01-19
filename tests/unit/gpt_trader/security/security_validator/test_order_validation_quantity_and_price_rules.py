"""Quantity/price rule tests for order validation in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestOrderValidationQuantityAndPriceRules:
    """Test quantity and price validation behavior."""

    def test_limit_order_price_validation(self, security_validator: Any) -> None:
        """Test limit order price validation."""
        # Limit order without price
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "limit",
            # Missing price
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        assert any("Value must be at least" in error for error in result.errors)

    def test_order_quantity_validation(self, security_validator: Any) -> None:
        """Test order quantity validation."""
        # Order with invalid quantity
        order = {
            "symbol": "BTC-USD",
            "quantity": "invalid",
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        assert any("Invalid numeric value" in error for error in result.errors)

    def test_order_with_very_small_quantity(self, security_validator: Any) -> None:
        """Test order with very small quantity."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.000001,  # Very small
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should fail minimum quantity check
        assert not result.is_valid

    def test_order_with_very_large_quantity(self, security_validator: Any) -> None:
        """Test order with very large quantity."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 1000000,  # Very large
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should fail maximum quantity check
        assert not result.is_valid

    def test_order_with_negative_quantity(self, security_validator: Any) -> None:
        """Test order with negative quantity."""
        order = {
            "symbol": "BTC-USD",
            "quantity": -0.001,  # Negative
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should fail minimum quantity check
        assert not result.is_valid

    def test_order_with_negative_price(self, security_validator: Any) -> None:
        """Test order with negative price."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "limit",
            "price": -50000.0,  # Negative
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should fail minimum price check
        assert not result.is_valid

    def test_order_validation_with_missing_price_for_market_order(
        self, security_validator: Any
    ) -> None:
        """Test market order validation with missing price."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "market",
            # No price field
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
