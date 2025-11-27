"""Tests for validation decorators."""

from __future__ import annotations

from typing import Any

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.base_validators import Validator
from gpt_trader.validation.decorators import validate_inputs


class TestValidateInputsDecorator:
    """Test validate_inputs decorator."""

    def test_decorator_with_validator_instance(self) -> None:
        """Test decorator with Validator instance."""
        # Validator that requires positive numbers
        positive_validator = Validator(
            error_message="Must be positive",
            predicate=lambda x: x > 0,
        )

        @validate_inputs(value=positive_validator)
        def process(value: int) -> int:
            return value * 2

        result = process(5)
        assert result == 10

    def test_decorator_rejects_invalid_input(self) -> None:
        """Test that decorator rejects invalid input."""
        positive_validator = Validator(
            error_message="Must be positive",
            predicate=lambda x: x > 0,
        )

        @validate_inputs(value=positive_validator)
        def process(value: int) -> int:
            return value * 2

        with pytest.raises(ValidationError, match="Must be positive"):
            process(-5)

    def test_decorator_with_callable(self) -> None:
        """Test decorator with plain callable instead of Validator."""

        def uppercase(value: Any, field_name: str) -> str:
            return str(value).upper()

        @validate_inputs(name=uppercase)
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = greet("world")
        assert result == "Hello, WORLD!"

    def test_decorator_with_multiple_validators(self) -> None:
        """Test decorator with multiple validators."""
        positive = Validator(predicate=lambda x: x > 0)
        non_empty = Validator(predicate=lambda x: len(x) > 0)

        @validate_inputs(count=positive, name=non_empty)
        def repeat(name: str, count: int) -> str:
            return name * count

        result = repeat("hi", 3)
        assert result == "hihihi"

    def test_decorator_validates_only_specified_params(self) -> None:
        """Test that decorator only validates specified parameters."""
        positive = Validator(predicate=lambda x: x > 0)

        @validate_inputs(a=positive)
        def add(a: int, b: int) -> int:
            return a + b

        # a is validated, b is not
        result = add(5, -3)  # b can be negative
        assert result == 2

        with pytest.raises(ValidationError):
            add(-5, 3)  # a cannot be negative

    def test_decorator_with_kwargs(self) -> None:
        """Test decorator with keyword arguments."""
        positive = Validator(predicate=lambda x: x > 0)

        @validate_inputs(value=positive)
        def double(value: int) -> int:
            return value * 2

        # Call with keyword argument
        result = double(value=5)
        assert result == 10

    def test_decorator_with_default_values(self) -> None:
        """Test decorator with default parameter values."""
        positive = Validator(predicate=lambda x: x > 0)

        @validate_inputs(multiplier=positive)
        def scale(value: int, multiplier: int = 2) -> int:
            return value * multiplier

        # Use default value
        result1 = scale(5)
        assert result1 == 10

        # Override default
        result2 = scale(5, multiplier=3)
        assert result2 == 15

    def test_decorator_transforms_input(self) -> None:
        """Test that decorator can transform input values."""
        # Validator that returns (True, transformed_value)
        to_int = Validator(
            predicate=lambda x: (True, int(x)),
        )

        @validate_inputs(value=to_int)
        def double(value: int) -> int:
            return value * 2

        result = double("5")  # Pass string, validator converts to int
        assert result == 10
        assert isinstance(result, int)

    def test_decorator_preserves_function_behavior(self) -> None:
        """Test that decorator preserves normal function behavior."""

        @validate_inputs()  # No validators
        def add(a: int, b: int) -> int:
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_decorator_with_mixed_positional_and_keyword(self) -> None:
        """Test decorator with mixed positional and keyword arguments."""
        positive = Validator(predicate=lambda x: x > 0)

        @validate_inputs(a=positive, b=positive)
        def multiply(a: int, b: int) -> int:
            return a * b

        # Mixed call
        result = multiply(3, b=4)
        assert result == 12


class TestValidateInputsEdgeCases:
    """Test edge cases for validate_inputs decorator."""

    def test_validator_for_nonexistent_param_is_ignored(self) -> None:
        """Test that validators for non-existent params are ignored."""
        positive = Validator(predicate=lambda x: x > 0)

        @validate_inputs(nonexistent=positive)
        def process(value: int) -> int:
            return value

        # Should work without error
        result = process(5)
        assert result == 5

    def test_callable_validator_receives_field_name(self) -> None:
        """Test that callable validators receive the field name."""
        received_field_name = []

        def capture_field(value: Any, field_name: str) -> Any:
            received_field_name.append(field_name)
            return value

        @validate_inputs(my_param=capture_field)
        def process(my_param: str) -> str:
            return my_param

        process("test")
        assert received_field_name == ["my_param"]

    def test_validation_error_contains_field_info(self) -> None:
        """Test that ValidationError contains field information."""
        validator = Validator(
            error_message="Invalid value",
            predicate=lambda x: False,  # Always fail
        )

        @validate_inputs(important_field=validator)
        def process(important_field: int) -> int:
            return important_field

        with pytest.raises(ValidationError) as exc_info:
            process(42)

        assert exc_info.value.context["field"] == "important_field"

    def test_decorator_with_none_values(self) -> None:
        """Test decorator handles None values."""
        # Validator that accepts None
        accepts_none = Validator(predicate=lambda x: x is None or x > 0)

        @validate_inputs(value=accepts_none)
        def process(value: int | None) -> int | None:
            return value

        result = process(None)
        assert result is None

        result2 = process(5)
        assert result2 == 5

    def test_decorator_with_complex_types(self) -> None:
        """Test decorator with complex types like lists."""
        non_empty_list = Validator(
            predicate=lambda x: isinstance(x, list) and len(x) > 0,
            error_message="Must be non-empty list",
        )

        @validate_inputs(items=non_empty_list)
        def first_item(items: list) -> Any:
            return items[0]

        result = first_item([1, 2, 3])
        assert result == 1

        with pytest.raises(ValidationError, match="non-empty list"):
            first_item([])
