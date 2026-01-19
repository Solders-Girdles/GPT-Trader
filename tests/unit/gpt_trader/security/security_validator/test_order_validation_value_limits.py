"""Value/limit tests for order validation in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestOrderValidationValueLimits:
    """Test order value and position size limits."""

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

    def test_non_limit_order_with_non_numeric_price(self, security_validator: Any) -> None:
        """Test non-limit order with non-numeric price falls back to default.

        This covers line 68 in order_validator.py where price_value is set to 100.0
        when order is not a limit order and price is non-numeric.
        """
        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "market",  # Not limit
            "price": "not-a-number",  # Non-numeric price
        }
        account_value = 100000.0

        result = security_validator.validate_order_request(order, account_value)

        # Should use default price of 100.0 for order value calculation
        # Order value = 0.001 * 100.0 = 0.10 < $1 minimum
        assert not result.is_valid
        assert any("Order value below minimum" in error for error in result.errors)
