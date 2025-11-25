"""Numeric validation for financial values."""

from decimal import Decimal, InvalidOperation
from typing import Any

from gpt_trader.validation import DecimalRule, RuleError

from .input_sanitizer import ValidationResult


class NumericValidator:
    """Validate numeric inputs."""

    _NUMERIC_RULE = DecimalRule()

    @classmethod
    def validate_numeric(
        cls, value: Any, min_val: float | None = None, max_val: float | None = None
    ) -> ValidationResult:
        """Validate numeric input"""
        errors = []

        try:
            # Convert to Decimal for precise financial calculations
            num_value = cls._NUMERIC_RULE(value, "value")
            if not isinstance(num_value, Decimal):
                raise TypeError(f"Expected Decimal, got {type(num_value).__name__}")

            if not num_value.is_finite():
                raise InvalidOperation("Non-finite value")

            if min_val is not None and num_value < Decimal(str(min_val)):
                errors.append(f"Value must be at least {min_val}")

            if max_val is not None and num_value > Decimal(str(max_val)):
                errors.append(f"Value must not exceed {max_val}")

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                sanitized_value=float(num_value) if not errors else None,
            )

        except (InvalidOperation, ValueError, RuleError, TypeError):
            return ValidationResult(False, ["Invalid numeric value"])
