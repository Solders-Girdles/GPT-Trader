"""Account-size sensitivity tests for order validation in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestOrderValidationAccountValueEffects:
    """Test account value effects on order validation."""

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
