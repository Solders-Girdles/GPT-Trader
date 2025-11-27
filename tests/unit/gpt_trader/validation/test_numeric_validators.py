"""Tests for numeric validators."""

from __future__ import annotations

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.numeric_validators import (
    PercentageValidator,
    PositiveNumberValidator,
)


class TestPositiveNumberValidator:
    """Tests for PositiveNumberValidator."""

    def test_validates_positive_integer(self) -> None:
        validator = PositiveNumberValidator()
        assert validator.validate(42, "quantity") == 42.0

    def test_validates_positive_float(self) -> None:
        validator = PositiveNumberValidator()
        assert validator.validate(3.14, "price") == 3.14

    def test_validates_string_number(self) -> None:
        validator = PositiveNumberValidator()
        assert validator.validate("100.5", "amount") == 100.5

    def test_rejects_zero_by_default(self) -> None:
        validator = PositiveNumberValidator()
        with pytest.raises(ValidationError, match="must be > 0"):
            validator.validate(0, "quantity")

    def test_allows_zero_when_configured(self) -> None:
        validator = PositiveNumberValidator(allow_zero=True)
        assert validator.validate(0, "quantity") == 0.0

    def test_rejects_negative_number(self) -> None:
        validator = PositiveNumberValidator()
        with pytest.raises(ValidationError, match="must be > 0"):
            validator.validate(-5, "quantity")

    def test_rejects_negative_with_allow_zero(self) -> None:
        validator = PositiveNumberValidator(allow_zero=True)
        with pytest.raises(ValidationError, match="must be >= 0"):
            validator.validate(-1, "amount")

    def test_rejects_non_numeric_string(self) -> None:
        validator = PositiveNumberValidator()
        with pytest.raises(ValidationError, match="must be a number"):
            validator.validate("abc", "price")

    def test_rejects_none(self) -> None:
        validator = PositiveNumberValidator()
        with pytest.raises(ValidationError, match="must be a number"):
            validator.validate(None, "value")

    def test_field_name_in_error_message(self) -> None:
        validator = PositiveNumberValidator()
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(-1, "position_size")
        assert exc_info.value.context["field"] == "position_size"


class TestPercentageValidator:
    """Tests for PercentageValidator."""

    def test_validates_decimal_percentage(self) -> None:
        validator = PercentageValidator(as_decimal=True)
        assert validator.validate(0.5, "risk") == 0.5

    def test_validates_boundary_zero(self) -> None:
        validator = PercentageValidator(as_decimal=True)
        assert validator.validate(0, "risk") == 0.0

    def test_validates_boundary_one(self) -> None:
        validator = PercentageValidator(as_decimal=True)
        assert validator.validate(1, "risk") == 1.0

    def test_rejects_above_one_decimal(self) -> None:
        validator = PercentageValidator(as_decimal=True)
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            validator.validate(1.5, "risk")

    def test_rejects_negative_decimal(self) -> None:
        validator = PercentageValidator(as_decimal=True)
        with pytest.raises(ValidationError, match="must be between 0 and 1"):
            validator.validate(-0.1, "risk")

    def test_validates_whole_percentage(self) -> None:
        validator = PercentageValidator(as_decimal=False)
        # 50% should be converted to 0.5
        assert validator.validate(50, "allocation") == 0.5

    def test_validates_whole_percentage_boundaries(self) -> None:
        validator = PercentageValidator(as_decimal=False)
        assert validator.validate(0, "min") == 0.0
        assert validator.validate(100, "max") == 1.0

    def test_rejects_above_100_whole(self) -> None:
        validator = PercentageValidator(as_decimal=False)
        with pytest.raises(ValidationError, match="must be between 0 and 100"):
            validator.validate(150, "allocation")

    def test_rejects_negative_whole(self) -> None:
        validator = PercentageValidator(as_decimal=False)
        with pytest.raises(ValidationError, match="must be between 0 and 100"):
            validator.validate(-10, "allocation")

    def test_validates_string_percentage(self) -> None:
        validator = PercentageValidator(as_decimal=True)
        assert validator.validate("0.25", "fee") == 0.25

    def test_rejects_non_numeric(self) -> None:
        validator = PercentageValidator()
        with pytest.raises(ValidationError, match="must be a number"):
            validator.validate("fifty", "percentage")

    def test_field_name_in_error(self) -> None:
        validator = PercentageValidator(as_decimal=True)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(2.0, "stop_loss_pct")
        assert exc_info.value.context["field"] == "stop_loss_pct"
