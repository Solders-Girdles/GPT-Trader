"""Tests for configuration validators."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import gpt_trader.validation.config_validators as config_validators_module
from gpt_trader.errors import ValidationError
from gpt_trader.validation.base_validators import Validator
from gpt_trader.validation.config_validators import validate_config


class TestValidateConfig:
    """Test validate_config function."""

    def test_validates_simple_config(self) -> None:
        """Test validation of a simple configuration."""
        schema = {
            "name": Validator(),  # Pass-through validator
            "value": Validator(),
        }
        config = {"name": "test", "value": 42}

        result = validate_config(config, schema)

        assert result == {"name": "test", "value": 42}

    def test_validates_with_transforming_validators(self) -> None:
        """Test validation with validators that transform values."""
        schema = {
            "count": Validator(predicate=lambda x: (True, int(x))),
        }
        config = {"count": "123"}

        result = validate_config(config, schema)

        assert result["count"] == 123
        assert isinstance(result["count"], int)

    def test_raises_for_missing_required_key(self) -> None:
        """Test that missing required keys raise ValidationError."""
        schema = {
            "required_key": Validator(),
        }
        config = {}  # Missing required_key

        with pytest.raises(ValidationError, match="Missing required config key"):
            validate_config(config, schema)

    def test_error_contains_field_name(self) -> None:
        """Test that error contains the missing field name."""
        schema = {
            "api_key": Validator(),
        }
        config = {}

        with pytest.raises(ValidationError) as exc_info:
            validate_config(config, schema)

        assert exc_info.value.context["field"] == "api_key"

    def test_raises_for_invalid_value(self) -> None:
        """Test that invalid values raise ValidationError."""
        schema = {
            "port": Validator(
                predicate=lambda x: isinstance(x, int) and 1 <= x <= 65535,
                error_message="Port must be integer between 1 and 65535",
            ),
        }
        config = {"port": 100000}  # Out of range

        with pytest.raises(ValidationError, match="Port must be integer"):
            validate_config(config, schema)

    def test_logs_warning_for_extra_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that extra keys in config trigger a warning."""
        schema = {
            "name": Validator(),
        }
        config = {
            "name": "test",
            "extra_key": "ignored",
            "another_extra": 123,
        }

        mock_logger = MagicMock()
        mock_get_logger = MagicMock(return_value=mock_logger)
        monkeypatch.setattr(config_validators_module, "_get_logger", mock_get_logger)

        result = validate_config(config, schema)

        # Should still return validated config
        assert result == {"name": "test"}

        # Should log warning about extra keys
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "extra_key" in warning_message or "another_extra" in warning_message

    def test_returns_only_schema_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that result only contains keys from schema."""
        schema = {
            "a": Validator(),
            "b": Validator(),
        }
        config = {
            "a": 1,
            "b": 2,
            "c": 3,  # Extra key
        }

        monkeypatch.setattr(config_validators_module, "_get_logger", MagicMock())
        result = validate_config(config, schema)

        assert set(result.keys()) == {"a", "b"}
        assert "c" not in result

    def test_empty_schema_with_empty_config(self) -> None:
        """Test validation with empty schema and config."""
        schema: dict[str, Validator] = {}
        config: dict[str, Any] = {}

        result = validate_config(config, schema)

        assert result == {}

    def test_empty_schema_with_extra_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test validation with empty schema but extra config keys."""
        schema: dict[str, Validator] = {}
        config = {"extra": "value"}

        mock_logger = MagicMock()
        mock_get_logger = MagicMock(return_value=mock_logger)
        monkeypatch.setattr(config_validators_module, "_get_logger", mock_get_logger)

        result = validate_config(config, schema)

        assert result == {}
        mock_logger.warning.assert_called_once()
