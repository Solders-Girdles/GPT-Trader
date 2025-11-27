"""Tests for type-based validators."""

from __future__ import annotations

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.type_validators import (
    ChoiceValidator,
    RangeValidator,
    RegexValidator,
    TypeValidator,
)


class TestTypeValidator:
    """Tests for TypeValidator."""

    def test_validates_correct_type_string(self) -> None:
        validator = TypeValidator(expected_type=str)
        assert validator.validate("hello", "name") == "hello"

    def test_validates_correct_type_int(self) -> None:
        validator = TypeValidator(expected_type=int)
        assert validator.validate(42, "count") == 42

    def test_validates_correct_type_list(self) -> None:
        validator = TypeValidator(expected_type=list)
        result = validator.validate([1, 2, 3], "items")
        assert result == [1, 2, 3]

    def test_rejects_wrong_type(self) -> None:
        validator = TypeValidator(expected_type=str)
        with pytest.raises(ValidationError, match="must be of type str"):
            validator.validate(123, "name")

    def test_uses_custom_error_message(self) -> None:
        validator = TypeValidator(expected_type=int, error_message="Expected an integer")
        with pytest.raises(ValidationError, match="Expected an integer"):
            validator.validate("not an int", "value")

    def test_field_name_in_error(self) -> None:
        validator = TypeValidator(expected_type=float)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("string", "price")
        assert exc_info.value.context["field"] == "price"


class TestRangeValidator:
    """Tests for RangeValidator."""

    def test_validates_within_range_inclusive(self) -> None:
        validator = RangeValidator(min_value=0, max_value=100)
        assert validator.validate(50, "percentage") == 50

    def test_validates_at_min_boundary_inclusive(self) -> None:
        validator = RangeValidator(min_value=0, max_value=100, inclusive=True)
        assert validator.validate(0, "value") == 0

    def test_validates_at_max_boundary_inclusive(self) -> None:
        validator = RangeValidator(min_value=0, max_value=100, inclusive=True)
        assert validator.validate(100, "value") == 100

    def test_rejects_below_min_inclusive(self) -> None:
        validator = RangeValidator(min_value=0, max_value=100)
        with pytest.raises(ValidationError, match="must be >= 0"):
            validator.validate(-1, "value")

    def test_rejects_above_max_inclusive(self) -> None:
        validator = RangeValidator(min_value=0, max_value=100)
        with pytest.raises(ValidationError, match="must be <= 100"):
            validator.validate(101, "value")

    def test_rejects_at_min_boundary_exclusive(self) -> None:
        validator = RangeValidator(min_value=0, max_value=100, inclusive=False)
        with pytest.raises(ValidationError, match="must be > 0"):
            validator.validate(0, "value")

    def test_rejects_at_max_boundary_exclusive(self) -> None:
        validator = RangeValidator(min_value=0, max_value=100, inclusive=False)
        with pytest.raises(ValidationError, match="must be < 100"):
            validator.validate(100, "value")

    def test_validates_with_only_min(self) -> None:
        validator = RangeValidator(min_value=10)
        assert validator.validate(100, "value") == 100

    def test_validates_with_only_max(self) -> None:
        validator = RangeValidator(max_value=50)
        assert validator.validate(-100, "value") == -100

    def test_uses_custom_error_message(self) -> None:
        validator = RangeValidator(min_value=0, error_message="Value must be positive")
        with pytest.raises(ValidationError, match="Value must be positive"):
            validator.validate(-5, "amount")

    def test_field_name_in_error(self) -> None:
        validator = RangeValidator(min_value=0)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(-1, "leverage")
        assert exc_info.value.context["field"] == "leverage"


class TestChoiceValidator:
    """Tests for ChoiceValidator."""

    def test_validates_valid_choice(self) -> None:
        validator = ChoiceValidator(choices=["buy", "sell", "hold"])
        assert validator.validate("buy", "action") == "buy"

    def test_validates_numeric_choice(self) -> None:
        validator = ChoiceValidator(choices=[1, 2, 3])
        assert validator.validate(2, "option") == 2

    def test_rejects_invalid_choice(self) -> None:
        validator = ChoiceValidator(choices=["buy", "sell"])
        with pytest.raises(ValidationError, match="must be one of"):
            validator.validate("hold", "action")

    def test_case_sensitive(self) -> None:
        validator = ChoiceValidator(choices=["BUY", "SELL"])
        with pytest.raises(ValidationError, match="must be one of"):
            validator.validate("buy", "action")

    def test_uses_custom_error_message(self) -> None:
        validator = ChoiceValidator(choices=["a", "b"], error_message="Invalid option")
        with pytest.raises(ValidationError, match="Invalid option"):
            validator.validate("c", "choice")

    def test_field_name_in_error(self) -> None:
        validator = ChoiceValidator(choices=["long", "short"])
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("neutral", "position_type")
        assert exc_info.value.context["field"] == "position_type"


class TestRegexValidator:
    """Tests for RegexValidator."""

    def test_validates_matching_pattern(self) -> None:
        validator = RegexValidator(pattern=r"^\d{3}-\d{4}$")
        assert validator.validate("123-4567", "phone") == "123-4567"

    def test_validates_email_pattern(self) -> None:
        validator = RegexValidator(pattern=r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
        assert validator.validate("test@example.com", "email") == "test@example.com"

    def test_rejects_non_matching_pattern(self) -> None:
        validator = RegexValidator(pattern=r"^\d{3}-\d{4}$")
        with pytest.raises(ValidationError, match="does not match required pattern"):
            validator.validate("12-34567", "phone")

    def test_rejects_non_string_input(self) -> None:
        validator = RegexValidator(pattern=r"^\d+$")
        with pytest.raises(ValidationError, match="must be a string"):
            validator.validate(12345, "value")

    def test_uses_custom_error_message(self) -> None:
        validator = RegexValidator(pattern=r"^\d+$", error_message="Numbers only please")
        with pytest.raises(ValidationError, match="Numbers only please"):
            validator.validate("abc", "code")

    def test_validates_start_anchor(self) -> None:
        validator = RegexValidator(pattern=r"^[A-Z]")
        assert validator.validate("Apple", "name") == "Apple"

    def test_field_name_in_error(self) -> None:
        validator = RegexValidator(pattern=r"^[a-z]+$")
        with pytest.raises(ValidationError) as exc_info:
            validator.validate("ABC", "identifier")
        assert exc_info.value.context["field"] == "identifier"
