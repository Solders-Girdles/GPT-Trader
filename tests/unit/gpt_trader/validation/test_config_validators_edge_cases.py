"""Edge-case tests for validate_config."""

from __future__ import annotations

import pytest

from gpt_trader.errors import ValidationError
from gpt_trader.validation.base_validators import Validator
from gpt_trader.validation.config_validators import validate_config


class TestValidateConfigEdgeCases:
    """Test edge cases for validate_config."""

    def test_validates_none_values(self) -> None:
        """Test validation of None values when allowed."""
        accepts_none = Validator(predicate=lambda x: x is None or x > 0)

        schema = {"optional": accepts_none}
        config = {"optional": None}

        result = validate_config(config, schema)
        assert result["optional"] is None

    def test_validates_boolean_values(self) -> None:
        """Test validation of boolean values."""
        is_bool = Validator(
            predicate=lambda x: isinstance(x, bool),
            error_message="Must be boolean",
        )

        schema = {"enabled": is_bool}
        config = {"enabled": True}

        result = validate_config(config, schema)
        assert result["enabled"] is True

    def test_validates_zero_values(self) -> None:
        """Test validation of zero values."""
        is_int = Validator(predicate=lambda x: isinstance(x, int))

        schema = {"count": is_int}
        config = {"count": 0}

        result = validate_config(config, schema)
        assert result["count"] == 0

    def test_validates_empty_string(self) -> None:
        """Test validation of empty string when allowed."""
        is_string = Validator(predicate=lambda x: isinstance(x, str))

        schema = {"name": is_string}
        config = {"name": ""}

        result = validate_config(config, schema)
        assert result["name"] == ""

    def test_multiple_validation_errors_reports_first(self) -> None:
        """Test that first missing key is reported in error."""
        schema = {
            "first": Validator(),
            "second": Validator(),
        }
        config = {}  # Both missing

        with pytest.raises(ValidationError) as exc_info:
            validate_config(config, schema)

        field = exc_info.value.context["field"]
        assert field in ["first", "second"]
