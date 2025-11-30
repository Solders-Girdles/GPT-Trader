"""Order validation for trading operations."""

from decimal import Decimal
from typing import Any

from .input_sanitizer import ValidationResult
from .numeric_validator import NumericValidator
from .symbol_validator import SymbolValidator


class OrderValidator:
    """Validate trading orders."""

    # Trading limits
    TRADING_LIMITS = {
        "max_position_size": 0.05,  # 5% of portfolio
        "max_daily_loss": 0.02,  # 2% daily loss limit
        "max_leverage": 2.0,  # 2:1 leverage
        "max_concentration": 0.20,  # 20% in single symbol
        "max_orders_per_minute": 5,
        "min_order_value": 1.0,  # $1 minimum
        "max_order_value": 100000.0,  # $100k maximum
    }

    @classmethod
    def validate_order_request(
        cls, order: dict[str, Any], account_value: float, limits: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        Validate trading order request.

        Args:
            order: Order details
            account_value: Current account value
            limits: Optional dictionary of trading limits to override defaults
        """
        limits = limits or cls.TRADING_LIMITS
        errors = []

        if not isinstance(order, dict):
            return ValidationResult(False, ["Order payload must be a mapping"])

        # Validate symbol
        symbol_result = SymbolValidator.validate_symbol(order.get("symbol", ""))
        if not symbol_result.is_valid:
            errors.extend(symbol_result.errors)

        # Validate quantity
        quantity = order.get("quantity", 0)
        quantity_result = NumericValidator.validate_numeric(
            quantity, min_val=0.001, max_val=1000000
        )
        if not quantity_result.is_valid:
            errors.extend(quantity_result.errors)
        quantity_value = quantity_result.sanitized_value if quantity_result.is_valid else 0.0

        # Validate price if limit order
        price_value = order.get("price", 100)
        if order.get("order_type") == "limit":
            price = order.get("price", 0)
            price_result = NumericValidator.validate_numeric(price, min_val=0.01, max_val=1000000)
            if not price_result.is_valid:
                errors.extend(price_result.errors)
            else:
                price_value = price_result.sanitized_value
        elif not isinstance(price_value, (int, float, Decimal)):
            price_value = 100.0

        # Check position size limits
        try:
            order_value = float(quantity_value) * float(price_value)
        except (TypeError, ValueError):
            order_value = 0.0

        if order_value < limits.get("min_order_value", 1.0):
            errors.append(f"Order value below minimum: ${limits.get('min_order_value', 1.0)}")

        if order_value > limits.get("max_order_value", 100000.0):
            errors.append(
                f"Order value exceeds maximum: ${limits.get('max_order_value', 100000.0)}"
            )

        # Check position concentration
        position_pct = order_value / account_value if account_value and account_value > 0 else 1.0
        if position_pct > limits.get("max_position_size", 0.05):
            errors.append(
                f"Position size exceeds {limits.get('max_position_size', 0.05) * 100}% limit"
            )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, sanitized_value=order if not errors else None
        )
