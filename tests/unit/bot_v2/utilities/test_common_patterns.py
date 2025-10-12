"""Tests for common utility patterns and helpers."""

from __future__ import annotations

from decimal import Decimal

import pytest

from bot_v2.errors import ValidationError
from bot_v2.utilities.common_patterns import (
    format_decimal,
    safe_decimal_division,
    validate_decimal_positive,
    validate_decimal_range,
)


class TestValidateDecimalPositive:
    """Test cases for validate_decimal_positive function."""

    def test_valid_positive_decimal(self):
        """Test valid positive decimal."""
        result = validate_decimal_positive("10.5", "test_value")
        assert result == Decimal("10.5")

    def test_valid_zero_with_allow(self):
        """Test zero value when allowed."""
        result = validate_decimal_positive("0", "test_value", allow_zero=True)
        assert result == Decimal("0")

    def test_invalid_zero_without_allow(self):
        """Test zero value when not allowed."""
        with pytest.raises(ValidationError, match="test_value must be > 0"):
            validate_decimal_positive("0", "test_value", allow_zero=False)

    def test_invalid_negative(self):
        """Test negative value."""
        with pytest.raises(ValidationError, match="test_value must be > 0"):
            validate_decimal_positive("-5", "test_value")

    def test_invalid_string(self):
        """Test invalid string value."""
        with pytest.raises(ValidationError, match="test_value must be a valid number"):
            validate_decimal_positive("invalid", "test_value")

    def test_various_input_types(self):
        """Test various input types."""
        assert validate_decimal_positive(10, "test") == Decimal("10")
        assert validate_decimal_positive(10.5, "test") == Decimal("10.5")
        assert validate_decimal_positive(Decimal("5.5"), "test") == Decimal("5.5")


class TestValidateDecimalRange:
    """Test cases for validate_decimal_range function."""

    def test_valid_in_range(self):
        """Test value within valid range."""
        result = validate_decimal_range(
            "5.5", "test_value", min_value=Decimal("1"), max_value=Decimal("10")
        )
        assert result == Decimal("5.5")

    def test_below_minimum(self):
        """Test value below minimum."""
        with pytest.raises(ValidationError, match="test_value must be >= 1"):
            validate_decimal_range(
                "0.5", "test_value", min_value=Decimal("1"), max_value=Decimal("10")
            )

    def test_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ValidationError, match="test_value must be <= 10"):
            validate_decimal_range(
                "15", "test_value", min_value=Decimal("1"), max_value=Decimal("10")
            )

    def test_exclusive_bounds(self):
        """Test exclusive bounds."""
        # Should fail at exact minimum
        with pytest.raises(ValidationError, match="test_value must be > 1"):
            validate_decimal_range(
                "1",
                "test_value",
                min_value=Decimal("1"),
                max_value=Decimal("10"),
                inclusive_min=False,
            )

        # Should fail at exact maximum
        with pytest.raises(ValidationError, match="test_value must be < 10"):
            validate_decimal_range(
                "10",
                "test_value",
                min_value=Decimal("1"),
                max_value=Decimal("10"),
                inclusive_max=False,
            )

    def test_no_bounds(self):
        """Test with no bounds specified."""
        result = validate_decimal_range("5.5", "test_value")
        assert result == Decimal("5.5")


class TestSafeDecimalDivision:
    """Test cases for safe_decimal_division function."""

    def test_successful_division(self):
        """Test successful division."""
        result = safe_decimal_division("10", "2")
        assert result == Decimal("5")

    def test_division_by_zero_with_default(self):
        """Test division by zero with default value."""
        result = safe_decimal_division("10", "0", default=Decimal("-1"))
        assert result == Decimal("-1")

    def test_division_by_zero_without_default(self):
        """Test division by zero without default value."""
        with pytest.raises(ValidationError, match="result: Division by zero"):
            safe_decimal_division("10", "0")

    def test_invalid_inputs_with_default(self):
        """Test invalid inputs with default value."""
        result = safe_decimal_division("invalid", "2", default=Decimal("-1"))
        assert result == Decimal("-1")

    def test_invalid_inputs_without_default(self):
        """Test invalid inputs without default value."""
        with pytest.raises(ValidationError, match="result: Invalid division operation"):
            safe_decimal_division("invalid", "2")

    def test_various_input_types(self):
        """Test various input types."""
        assert safe_decimal_division(10, 2) == Decimal("5")
        assert safe_decimal_division(10.0, 2.0) == Decimal("5")
        assert safe_decimal_division(Decimal("10"), Decimal("2")) == Decimal("5")


class TestFormatDecimal:
    """Test cases for format_decimal function."""

    def test_basic_formatting(self):
        """Test basic decimal formatting."""
        result = format_decimal(Decimal("10.5000"))
        assert result == "10.5"

    def test_with_decimal_places(self):
        """Test formatting with specific decimal places."""
        result = format_decimal(Decimal("10.5"), decimal_places=4)
        assert result == "10.5000"

    def test_strip_trailing_zeros_false(self):
        """Test formatting without stripping trailing zeros."""
        result = format_decimal(Decimal("10.5000"), strip_trailing_zeros=False)
        assert result == "10.5000"

    def test_integer_value(self):
        """Test formatting integer value."""
        result = format_decimal(Decimal("10"))
        assert result == "10"

    def test_various_input_types(self):
        """Test various input types."""
        assert format_decimal("10.5000") == "10.5"
        assert format_decimal(10.5000) == "10.5"
        assert format_decimal(10) == "10"

    def test_very_small_decimal(self):
        """Test formatting very small decimal."""
        result = format_decimal(Decimal("0.0001000"))
        assert result == "0.0001"

    def test_rounding_with_decimal_places(self):
        """Test rounding when specifying decimal places."""
        result = format_decimal(Decimal("10.5678"), decimal_places=2)
        assert result == "10.57"
