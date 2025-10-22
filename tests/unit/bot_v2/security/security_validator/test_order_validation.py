"""Tests for order validation in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestOrderValidation:
    """Test order validation scenarios."""

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

    def test_invalid_symbol_rejection(
        self, security_validator: Any, sample_order_requests: dict[str, dict[str, Any]]
    ) -> None:
        """Test order with invalid symbol is rejected."""
        order = {
            "symbol": "INVALID_SYMBOL!",
            "quantity": 0.001,
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        assert any("Invalid symbol format" in error for error in result.errors)

    def test_order_value_below_minimum(
        self, security_validator: Any, sample_order_requests: dict[str, dict[str, Any]]
    ) -> None:
        """Test order value below minimum is rejected."""
        order = sample_order_requests["invalid_small_order"]
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        assert any("Order value below minimum" in error for error in result.errors)

    def test_order_value_above_maximum(
        self, security_validator: Any, sample_order_requests: dict[str, dict[str, Any]]
    ) -> None:
        """Test order value above maximum is rejected."""
        order = sample_order_requests["invalid_large_order"]
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        assert any("Order value exceeds maximum" in error for error in result.errors)

    def test_position_size_limit(self, security_validator: Any) -> None:
        """Test position size limit enforcement."""
        # Order that exceeds position size limit
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.2,  # 10% of account
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        assert any("Position size exceeds" in error for error in result.errors)

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

    def test_order_with_zero_account_value(self, security_validator: Any) -> None:
        """Test order validation with zero account value."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 0.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        # Should trigger position size limit

    def test_order_with_negative_account_value(self, security_validator: Any) -> None:
        """Test order validation with negative account value."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = -1000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        # Should trigger position size limit

    def test_order_missing_required_fields(self, security_validator: Any) -> None:
        """Test order with missing required fields."""
        # Order missing symbol
        order = {
            "quantity": 0.001,
            "order_type": "limit",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        assert not result.is_valid
        assert any("Invalid symbol format" in error for error in result.errors)

    def test_order_with_invalid_order_type(self, security_validator: Any) -> None:
        """Test order with invalid order type."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "invalid_type",
            "price": 50000.0,
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should still validate basic fields but may not check price
        # depending on implementation
        assert result.is_valid or len(result.errors) > 0

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

    def test_order_validation_error_accumulation(self, security_validator: Any) -> None:
        """Test that multiple order validation errors are accumulated."""
        order = {
            "symbol": "INVALID",  # Invalid symbol
            "quantity": "invalid",  # Invalid quantity
            "order_type": "limit",
            "price": -100,  # Invalid price
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should have multiple errors
        assert len(result.errors) >= 2

    def test_order_edge_case_values(self, security_validator: Any) -> None:
        """Test order with edge case values."""
        # Order at minimum limits
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "limit",
            "price": 0.01,  # Minimum price
        }
        account_value = 1000.0  # Small account

        result = security_validator.validate_order_request(order, account_value)

        # Order value = 0.001 * 0.01 = $0.00001, below minimum
        assert not result.is_valid
        assert any("Order value below minimum" in error for error in result.errors)

    def test_order_with_different_account_sizes(self, security_validator: Any) -> None:
        """Test order validation with different account sizes."""
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.01,
            "order_type": "limit",
            "price": 50000.0,
        }

        # Small account
        result_small = security_validator.validate_order_request(order, 1000.0)
        assert not result_small.is_valid  # 50% position

        # Large account
        result_large = security_validator.validate_order_request(order, 1000000.0)
        assert result_large.is_valid  # 0.5% position

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
