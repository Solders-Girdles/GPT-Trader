"""Numeric validation edge behaviour tests for SecurityValidator."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest


class TestNumericEdgeBehaviour:
    """Cover invalid values and boundary conditions."""

    @pytest.mark.parametrize(
        "value",
        [
            "not_a_number",
            "abc123",
            "12.34.56",
            "infinity",
            "NaN",
            None,
            [],
            {},
            "123abc",
            "$123.45",
            "1,234.56",
        ],
    )
    def test_invalid_numeric_values(self, security_validator: Any, value: Any) -> None:
        result = security_validator.validate_numeric(value)

        assert not result.is_valid
        assert any("Invalid numeric value" in error for error in result.errors)

    def test_numeric_with_leading_trailing_whitespace(self, security_validator: Any) -> None:
        for value, expected in [(" 123.45", 123.45), ("123.45 ", 123.45), ("  123.45  ", 123.45)]:
            result = security_validator.validate_numeric(value)
            assert result.is_valid
            assert result.sanitized_value == expected

    def test_numeric_with_plus_sign(self, security_validator: Any) -> None:
        result = security_validator.validate_numeric("+123.45")
        assert result.is_valid
        assert result.sanitized_value == 123.45

    def test_numeric_boundary_values(self, security_validator: Any) -> None:
        assert security_validator.validate_numeric("5", min_val=5, max_val=10).is_valid
        assert security_validator.validate_numeric("10", min_val=5, max_val=10).is_valid

    def test_numeric_type_conversion(self, security_validator: Any) -> None:
        assert isinstance(security_validator.validate_numeric("123").sanitized_value, float)
        assert isinstance(security_validator.validate_numeric(123).sanitized_value, float)
        assert isinstance(security_validator.validate_numeric(123.45).sanitized_value, float)

    def test_numeric_error_messages(self, security_validator: Any) -> None:
        result = security_validator.validate_numeric("3", min_val=5)
        assert not result.is_valid
        assert "at least 5" in result.errors[0]

        result = security_validator.validate_numeric("15", max_val=10)
        assert not result.is_valid
        assert "not exceed 10" in result.errors[0]

        result = security_validator.validate_numeric("invalid")
        assert not result.is_valid
        assert "Invalid numeric value" in result.errors[0]

    def test_numeric_validation_with_decimal_input(self, security_validator: Any) -> None:
        result = security_validator.validate_numeric(Decimal("123.456789"))
        assert result.is_valid
        assert result.sanitized_value == 123.456789

    def test_numeric_validation_consistency(self, security_validator: Any) -> None:
        value = "123.45"
        first = security_validator.validate_numeric(value)
        second = security_validator.validate_numeric(value)

        assert first.is_valid == second.is_valid
        assert first.sanitized_value == second.sanitized_value
        assert first.errors == second.errors

    def test_numeric_validation_performance(self, security_validator: Any) -> None:
        for value in ["123.45", "678.90", "1000"] * 100:
            assert security_validator.validate_numeric(value).is_valid

    def test_numeric_validation_with_extreme_ranges(self, security_validator: Any) -> None:
        assert security_validator.validate_numeric("1e10", min_val=0, max_val=1e20).is_valid
        assert security_validator.validate_numeric("1e-10", min_val=0, max_val=1e-5).is_valid

    def test_numeric_validation_with_floating_point_edge_cases(
        self, security_validator: Any
    ) -> None:
        for value, expected in [
            ("0.1", 0.1),
            ("0.2", 0.2),
            ("0.3", 0.3),
            ("1.0", 1.0),
            ("1.1", 1.1),
        ]:
            result = security_validator.validate_numeric(value)
            assert result.is_valid
            assert abs(result.sanitized_value - expected) < 0.0001

    @pytest.mark.parametrize("value", ["inf", "-inf", float("inf"), float("-inf"), float("nan")])
    def test_numeric_validation_rejects_non_finite(
        self, security_validator: Any, value: Any
    ) -> None:
        result = security_validator.validate_numeric(value)
        assert not result.is_valid
        assert any("Invalid numeric value" in error for error in result.errors)


class TestNumericValidationExceptionHandling:
    """Test exception handling in numeric validation."""

    def test_numeric_validation_non_decimal_return(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that TypeError is raised if DecimalRule returns non-Decimal.

        This covers the type assertion (line 27) in numeric_validator.py.
        The DecimalRule should always return a Decimal, but if it doesn't,
        the validator catches the TypeError and returns an error.
        """
        from gpt_trader.security.numeric_validator import NumericValidator

        # Mock the DecimalRule to return a float instead of Decimal
        rule_mock = MagicMock(return_value=123.45)
        monkeypatch.setattr(NumericValidator, "_NUMERIC_RULE", rule_mock)

        result = NumericValidator.validate_numeric("123.45")

        assert not result.is_valid
        assert "Invalid numeric value" in result.errors
