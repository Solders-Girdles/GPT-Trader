"""Tests for validate_config with common validator types."""

from __future__ import annotations

from gpt_trader.validation.base_validators import Validator
from gpt_trader.validation.config_validators import validate_config


class TestValidateConfigWithValidators:
    """Test validate_config with various validator types."""

    def test_with_string_validator(self) -> None:
        """Test config validation with string validators."""
        non_empty_string = Validator(
            predicate=lambda x: isinstance(x, str) and len(x) > 0,
            error_message="Must be non-empty string",
        )

        schema = {"username": non_empty_string}
        config = {"username": "john_doe"}

        result = validate_config(config, schema)
        assert result["username"] == "john_doe"

    def test_with_numeric_validator(self) -> None:
        """Test config validation with numeric validators."""
        positive_int = Validator(
            predicate=lambda x: isinstance(x, int) and x > 0,
            error_message="Must be positive integer",
        )

        schema = {"timeout": positive_int}
        config = {"timeout": 30}

        result = validate_config(config, schema)
        assert result["timeout"] == 30

    def test_with_list_validator(self) -> None:
        """Test config validation with list validators."""
        non_empty_list = Validator(
            predicate=lambda x: isinstance(x, list) and len(x) > 0,
            error_message="Must be non-empty list",
        )

        schema = {"symbols": non_empty_list}
        config = {"symbols": ["BTC", "ETH"]}

        result = validate_config(config, schema)
        assert result["symbols"] == ["BTC", "ETH"]

    def test_with_dict_validator(self) -> None:
        """Test config validation with dict validators."""
        is_dict = Validator(
            predicate=lambda x: isinstance(x, dict),
            error_message="Must be a dictionary",
        )

        schema = {"options": is_dict}
        config = {"options": {"a": 1, "b": 2}}

        result = validate_config(config, schema)
        assert result["options"] == {"a": 1, "b": 2}
