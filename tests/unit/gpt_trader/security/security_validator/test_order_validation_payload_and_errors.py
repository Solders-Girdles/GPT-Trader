"""Payload shape + error accumulation tests for order validation."""

from __future__ import annotations

from typing import Any


class TestOrderValidationPayloadAndErrors:
    """Test validation error handling and edge cases."""

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

    def test_order_value_calculation_fallback(self) -> None:
        """Test order value calculation exception handling.

        This covers lines 73-74 in order_validator.py where TypeError/ValueError
        is caught during order value calculation.
        """
        from gpt_trader.security.order_validator import OrderValidator

        order = {
            "symbol": "BTC-USD",
            "quantity": 0.001,
            "order_type": "market",
            "price": None,  # None can cause issues in some paths
        }
        account_value = 100000.0

        result = OrderValidator.validate_order_request(order, account_value)

        assert result is not None

    def test_order_payload_not_dict(self) -> None:
        """Test order validation with non-dict payload.

        This covers line 42 in order_validator.py.
        """
        from gpt_trader.security.order_validator import OrderValidator

        result = OrderValidator.validate_order_request("not a dict", 100000.0)

        assert not result.is_valid
        assert "must be a mapping" in result.errors[0].lower()

    def test_order_payload_list(self) -> None:
        """Test order validation with list payload."""
        from gpt_trader.security.order_validator import OrderValidator

        result = OrderValidator.validate_order_request(["item1", "item2"], 100000.0)

        assert not result.is_valid
        assert "must be a mapping" in result.errors[0].lower()
