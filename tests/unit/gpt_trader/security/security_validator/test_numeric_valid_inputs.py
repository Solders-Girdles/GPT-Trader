"""Numeric validation happy-path tests for SecurityValidator."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest


class TestNumericValidInputs:
    """Validate acceptable numeric inputs and range checks."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("123.45", 123.45),
            ("100", 100.0),
            ("0.001", 0.001),
            ("1e6", 1_000_000.0),
            (123, 123.0),
            (123.45, 123.45),
            (Decimal("123.45"), 123.45),
        ],
    )
    def test_valid_numeric_values(
        self, security_validator: Any, value: Any, expected: float
    ) -> None:
        result = security_validator.validate_numeric(value)

        assert result.is_valid
        assert result.sanitized_value == expected

    def test_numeric_range_validation_min(self, security_validator: Any) -> None:
        assert security_validator.validate_numeric("10", min_val=5).is_valid

        result = security_validator.validate_numeric("3", min_val=5)
        assert not result.is_valid
        assert any("at least 5" in error for error in result.errors)

    def test_numeric_range_validation_max(self, security_validator: Any) -> None:
        assert security_validator.validate_numeric("10", max_val=20).is_valid

        result = security_validator.validate_numeric("25", max_val=20)
        assert not result.is_valid
        assert any("not exceed 20" in error for error in result.errors)

    def test_numeric_range_validation_both(self, security_validator: Any) -> None:
        assert security_validator.validate_numeric("10", min_val=5, max_val=20).is_valid
        assert not security_validator.validate_numeric("3", min_val=5, max_val=20).is_valid
        assert not security_validator.validate_numeric("25", min_val=5, max_val=20).is_valid

    def test_numeric_precision_handling(self, security_validator: Any) -> None:
        result = security_validator.validate_numeric("123.456789012345")

        assert result.is_valid
        assert abs(result.sanitized_value - 123.456789012345) < 0.0001

    def test_numeric_negative_values(self, security_validator: Any) -> None:
        assert security_validator.validate_numeric("-10").is_valid
        assert security_validator.validate_numeric("-5", min_val=-10, max_val=0).is_valid

    def test_numeric_zero_handling(self, security_validator: Any) -> None:
        assert security_validator.validate_numeric("0").is_valid
        assert not security_validator.validate_numeric("0", min_val=1).is_valid
        assert not security_validator.validate_numeric("0", max_val=-1).is_valid

    def test_numeric_scientific_notation(self, security_validator: Any) -> None:
        cases = [("1e3", 1000.0), ("1.5e-2", 0.015), ("2E6", 2_000_000.0), ("-1.5e2", -150.0)]

        for value, expected in cases:
            result = security_validator.validate_numeric(value)
            assert result.is_valid
            assert result.sanitized_value == expected

    def test_numeric_edge_cases(self, security_validator: Any) -> None:
        edge_cases = [
            ("0.0000001", 0.0000001),
            ("999999999", 999999999.0),
            ("1.23456789012345", 1.23456789012345),
        ]

        for value, expected in edge_cases:
            result = security_validator.validate_numeric(value)
            assert result.is_valid
            assert abs(result.sanitized_value - expected) < 0.0001
