import decimal

import pytest

from src.bot_v2.utilities.common_patterns import (
    format_decimal,
    safe_decimal_division,
    validate_decimal_positive,
    validate_decimal_range,
)
from src.bot_v2.errors import ValidationError


class TestValidateDecimalPositive:
    def test_accepts_positive_value(self):
        result = validate_decimal_positive("3.14", field_name="pi")
        assert result == decimal.Decimal("3.14")

    def test_allows_zero_when_configured(self):
        result = validate_decimal_positive("0", allow_zero=True)
        assert result == decimal.Decimal("0")

    @pytest.mark.parametrize(
        "value,allow_zero,message",
        [
            ("0", False, "value must be > 0"),
            ("-1.2", True, "value must be >= 0"),
            ("abc", False, "value must be a valid number"),
        ],
    )
    def test_rejects_invalid_values(self, value, allow_zero, message):
        with pytest.raises(ValidationError) as exc:
            validate_decimal_positive(value, allow_zero=allow_zero)
        assert message in str(exc.value)


class TestValidateDecimalRange:
    def test_handles_inclusive_bounds(self):
        result = validate_decimal_range(
            "5.0",
            field_name="score",
            min_value=decimal.Decimal("5.0"),
            max_value=decimal.Decimal("10"),
            inclusive_min=True,
            inclusive_max=True,
        )
        assert result == decimal.Decimal("5.0")

    def test_handles_exclusive_bounds(self):
        with pytest.raises(ValidationError, match="score must be > 1"):
            validate_decimal_range(
                1,
                field_name="score",
                min_value=decimal.Decimal("1"),
                inclusive_min=False,
            )

        with pytest.raises(ValidationError, match="score must be < 10"):
            validate_decimal_range(
                10,
                field_name="score",
                max_value=decimal.Decimal("10"),
                inclusive_max=False,
            )

    def test_invalid_number_raises(self):
        with pytest.raises(decimal.InvalidOperation):
            validate_decimal_range("not-a-number", field_name="rate")


class TestSafeDecimalDivision:
    def test_divides_values(self):
        result = safe_decimal_division("10", "4")
        assert result == decimal.Decimal("2.5")

    def test_returns_default_on_zero_division(self):
        result = safe_decimal_division("10", 0, default=decimal.Decimal("0"))
        assert result == decimal.Decimal("0")

    def test_raises_when_no_default(self):
        with pytest.raises(ValidationError, match="Division by zero"):
            safe_decimal_division("10", 0, field_name="ratio")

    def test_invalid_inputs_return_default(self):
        result = safe_decimal_division("not-a-number", "5", default=decimal.Decimal("0"))
        assert result == decimal.Decimal("0")

        with pytest.raises(ValidationError, match="Invalid division operation"):
            safe_decimal_division("bad", "5")


class TestFormatDecimal:
    @pytest.mark.parametrize(
        "value,decimal_places,expected",
        [
            ("123.456", None, "123.456"),
            ("123.4000", None, "123.4"),
            ("123.4", 2, "123.40"),
            ("123.456", 2, "123.46"),
            ("123.5", 0, "124"),
        ],
    )
    def test_formats_decimal_values(self, value, decimal_places, expected):
        assert format_decimal(value, decimal_places=decimal_places) == expected

    def test_preserves_trailing_zeros_when_requested(self):
        assert format_decimal("1.2300", strip_trailing_zeros=False) == "1.2300"

    def test_returns_input_string_when_not_convertible(self):
        assert format_decimal("invalid") == "invalid"
