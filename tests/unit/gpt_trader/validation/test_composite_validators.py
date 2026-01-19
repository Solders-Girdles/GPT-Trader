"""Tests for composite validators."""

from __future__ import annotations

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.base_validators import Validator
from gpt_trader.validation.composite_validators import CompositeValidator

LT_100 = "Must be less than 100"
BETWEEN_1_100 = "Must be between 1 and 100"


class TestCompositeValidatorBasics:
    """Test basic CompositeValidator functionality."""

    def test_single_validator(self) -> None:
        """Test composite with single validator."""
        positive = Validator(predicate=lambda x: x > 0)
        composite = CompositeValidator(positive)

        result = composite.validate(5, "value")
        assert result == 5

    def test_multiple_validators_all_pass(self) -> None:
        """Test composite with multiple validators that all pass."""
        positive = Validator(predicate=lambda x: x > 0)
        less_than_100 = Validator(predicate=lambda x: x < 100)

        composite = CompositeValidator(positive, less_than_100)

        result = composite.validate(50, "value")
        assert result == 50

    def test_first_validator_fails(self) -> None:
        """Test that first failing validator raises error."""
        positive = Validator(predicate=lambda x: x > 0, error_message="Must be positive")
        less_than_100 = Validator(predicate=lambda x: x < 100)

        composite = CompositeValidator(positive, less_than_100)

        with pytest.raises(ValidationError, match="Must be positive"):
            composite.validate(-5, "value")

    def test_second_validator_fails(self) -> None:
        """Test that second failing validator raises error."""
        positive = Validator(predicate=lambda x: x > 0)
        less_than_100 = Validator(predicate=lambda x: x < 100, error_message=LT_100)

        composite = CompositeValidator(positive, less_than_100)

        with pytest.raises(ValidationError, match=LT_100):
            composite.validate(150, "value")

    def test_empty_composite_passes_through(self) -> None:
        """Test composite with no validators passes value through."""
        composite = CompositeValidator()

        result = composite.validate(42, "value")
        assert result == 42


class TestCompositeValidatorChaining:
    """Test validator chaining behavior."""

    def test_validators_run_in_order(self) -> None:
        """Test that validators run in the order they were provided."""
        call_order: list[str] = []

        def make_tracking_validator(name: str) -> Validator:
            def track(value: int) -> bool:
                call_order.append(name)
                return True

            return Validator(predicate=track)

        composite = CompositeValidator(
            make_tracking_validator("first"),
            make_tracking_validator("second"),
            make_tracking_validator("third"),
        )

        composite.validate(1, "value")

        assert call_order == ["first", "second", "third"]

    def test_transformation_chain(self) -> None:
        """Test that transformations are chained correctly."""
        double = Validator(predicate=lambda x: (True, x * 2))
        add_ten = Validator(predicate=lambda x: (True, x + 10))

        composite = CompositeValidator(double, add_ten)

        result = composite.validate(5, "value")
        assert result == 20

    def test_mixed_transformation_and_validation(self) -> None:
        """Test mixing transformations and pure validation."""
        # Transform string to int
        to_int = Validator(predicate=lambda x: (True, int(x)))
        # Validate is positive
        positive = Validator(predicate=lambda x: x > 0, error_message="Must be positive")

        composite = CompositeValidator(to_int, positive)

        result = composite.validate("42", "value")
        assert result == 42

        with pytest.raises(ValidationError, match="Must be positive"):
            composite.validate("-5", "value")


class TestCompositeValidatorCallable:
    """Test CompositeValidator callable interface."""

    def test_callable_interface(self) -> None:
        """Test that CompositeValidator can be called directly."""
        positive = Validator(predicate=lambda x: x > 0)
        composite = CompositeValidator(positive)

        # Call directly like a function
        result = composite(5, "value")
        assert result == 5

    def test_callable_with_default_field_name(self) -> None:
        """Test callable with default field name."""
        positive = Validator(predicate=lambda x: x > 0)
        composite = CompositeValidator(positive)

        result = composite(5)
        assert result == 5


class TestCompositeValidatorWithDifferentTypes:
    """Test CompositeValidator with different value types."""

    def test_string_validators(self) -> None:
        """Test composite with string validators."""
        non_empty = Validator(predicate=lambda x: len(x) > 0, error_message="Cannot be empty")
        max_length = Validator(predicate=lambda x: len(x) <= 10, error_message="Too long")

        composite = CompositeValidator(non_empty, max_length)

        assert composite.validate("hello", "name") == "hello"

        with pytest.raises(ValidationError, match="Cannot be empty"):
            composite.validate("", "name")

        with pytest.raises(ValidationError, match="Too long"):
            composite.validate("this is way too long", "name")

    def test_list_validators(self) -> None:
        """Test composite with list validators."""
        is_list = Validator(predicate=lambda x: isinstance(x, list), error_message="Must be list")
        non_empty = Validator(predicate=lambda x: len(x) > 0, error_message="Cannot be empty")

        composite = CompositeValidator(is_list, non_empty)

        assert composite.validate([1, 2, 3], "items") == [1, 2, 3]

        with pytest.raises(ValidationError, match="Cannot be empty"):
            composite.validate([], "items")

    def test_numeric_range_validators(self) -> None:
        """Test composite for numeric range validation."""
        is_int = Validator(predicate=lambda x: isinstance(x, int), error_message="Must be integer")
        in_range = Validator(predicate=lambda x: 1 <= x <= 100, error_message=BETWEEN_1_100)

        composite = CompositeValidator(is_int, in_range)

        assert composite.validate(50, "percent") == 50

        with pytest.raises(ValidationError, match="Must be integer"):
            composite.validate(50.5, "percent")

        with pytest.raises(ValidationError, match=BETWEEN_1_100):
            composite.validate(150, "percent")


class TestCompositeValidatorEdgeCases:
    """Test edge cases for CompositeValidator."""

    def test_validator_error_propagates_field_name(self) -> None:
        """Test that field name is propagated to error."""
        failing = Validator(predicate=lambda x: False, error_message="Always fails")
        composite = CompositeValidator(failing)

        with pytest.raises(ValidationError) as exc_info:
            composite.validate(1, "my_field")

        assert exc_info.value.context["field"] == "my_field"

    def test_none_value_handling(self) -> None:
        """Test handling of None values."""
        accepts_none = Validator(predicate=lambda x: x is None)
        composite = CompositeValidator(accepts_none)

        result = composite.validate(None, "optional")
        assert result is None

    def test_deeply_nested_composites(self) -> None:
        """Test nesting CompositeValidators."""
        inner1 = CompositeValidator(Validator(predicate=lambda x: (True, x + 1)))  # Add 1
        inner2 = CompositeValidator(Validator(predicate=lambda x: (True, x * 2)))  # Double

        # Compose composites
        outer = CompositeValidator(inner1, inner2)

        # 5 -> 6 -> 12
        result = outer.validate(5, "value")
        assert result == 12

    def test_many_validators_in_chain(self) -> None:
        """Test chain with many validators."""
        validators = [Validator(predicate=lambda x: (True, x + 1)) for _ in range(10)]
        composite = CompositeValidator(*validators)

        # Each validator adds 1, so 0 + 10 = 10
        result = composite.validate(0, "value")
        assert result == 10

    def test_early_failure_skips_remaining_validators(self) -> None:
        """Test that early failure skips remaining validators."""
        call_count = {"count": 0}

        def counting_validator(value: int) -> bool:
            call_count["count"] += 1
            return True

        failing = Validator(predicate=lambda x: False, error_message="Failed")
        counting = Validator(predicate=counting_validator)

        composite = CompositeValidator(failing, counting)

        with pytest.raises(ValidationError):
            composite.validate(1, "value")

        # Second validator should not have been called
        assert call_count["count"] == 0
