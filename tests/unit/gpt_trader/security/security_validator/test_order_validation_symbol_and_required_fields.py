"""Symbol + required-field tests for order validation in SecurityValidator."""

from __future__ import annotations

from typing import Any


class TestOrderValidationSymbolAndRequiredFields:
    """Test symbol and required field validation."""

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
