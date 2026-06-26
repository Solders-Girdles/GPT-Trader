"""Tests for app configuration validation module."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from gpt_trader.app.config.bot_config import BotConfig
from gpt_trader.app.config.validation import (
    ConfigValidationError,
    ConfigValidationResult,
    format_validation_errors,
    validate_config,
)


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_stores_errors_list(self) -> None:
        errors = ["Error 1", "Error 2"]
        exc = ConfigValidationError(errors)
        assert exc.errors == errors

    def test_message_joins_errors(self) -> None:
        errors = ["Error 1", "Error 2"]
        exc = ConfigValidationError(errors)
        assert str(exc) == "Error 1; Error 2"

    def test_empty_errors_default_message(self) -> None:
        exc = ConfigValidationError([])
        assert str(exc) == "Invalid configuration"

    def test_single_error_message(self) -> None:
        exc = ConfigValidationError(["Single error"])
        assert str(exc) == "Single error"

    def test_is_exception_subclass(self) -> None:
        exc = ConfigValidationError(["error"])
        assert isinstance(exc, Exception)

    def test_no_args_default_message(self) -> None:
        exc = ConfigValidationError()
        assert str(exc) == "Invalid configuration"
        assert exc.errors == []


class TestConfigValidationResult:
    """Tests for ConfigValidationResult model."""

    def test_valid_result_creation(self) -> None:
        result = ConfigValidationResult(is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_invalid_result_with_errors(self) -> None:
        result = ConfigValidationResult(is_valid=False, errors=["Missing field", "Invalid type"])
        assert result.is_valid is False
        assert result.errors == ["Missing field", "Invalid type"]

    def test_result_with_warnings(self) -> None:
        result = ConfigValidationResult(is_valid=True, warnings=["Deprecated option used"])
        assert result.warnings == ["Deprecated option used"]

    def test_has_errors_property_true(self) -> None:
        result = ConfigValidationResult(is_valid=False, errors=["Error"])
        assert result.has_errors is True

    def test_has_errors_property_false(self) -> None:
        result = ConfigValidationResult(is_valid=True)
        assert result.has_errors is False

    def test_has_warnings_property_true(self) -> None:
        result = ConfigValidationResult(is_valid=True, warnings=["Warning"])
        assert result.has_warnings is True

    def test_has_warnings_property_false(self) -> None:
        result = ConfigValidationResult(is_valid=True)
        assert result.has_warnings is False

    def test_both_errors_and_warnings(self) -> None:
        result = ConfigValidationResult(is_valid=False, errors=["Error"], warnings=["Warning"])
        assert result.has_errors is True
        assert result.has_warnings is True


class _TestModel(BaseModel):
    """Test model for validation error formatting."""

    name: str
    age: int


class TestFormatValidationErrors:
    """Tests for format_validation_errors function."""

    def test_formats_single_field_error(self) -> None:
        try:
            _TestModel(name="valid", age="not_an_int")  # type: ignore[arg-type]
        except ValidationError as exc:
            errors = format_validation_errors(exc)
            assert len(errors) >= 1
            assert any("age" in error for error in errors)

    def test_formats_multiple_errors(self) -> None:
        try:
            _TestModel(name=123, age="bad")  # type: ignore[arg-type]
        except ValidationError as exc:
            errors = format_validation_errors(exc)
            assert len(errors) >= 1

    def test_field_path_included_in_message(self) -> None:
        try:
            _TestModel(name="valid", age="bad")  # type: ignore[arg-type]
        except ValidationError as exc:
            errors = format_validation_errors(exc)
            # Should have field path
            assert any(":" in error for error in errors)

    def test_missing_field_error(self) -> None:
        try:
            _TestModel(name="test")  # type: ignore[call-arg]
        except ValidationError as exc:
            errors = format_validation_errors(exc)
            assert len(errors) >= 1


class TestValidateConfigCFMConsistency:
    """Tests for CFM mode and enablement consistency validation."""

    def test_spot_only_default_is_valid(self) -> None:
        assert validate_config(BotConfig()) == []

    def test_cfm_enabled_requires_cfm_trading_mode(self) -> None:
        errors = validate_config(BotConfig(cfm_enabled=True, trading_modes=["spot"]))

        assert "cfm_enabled requires trading_modes to include 'cfm'" in errors

    def test_cfm_trading_mode_requires_cfm_enabled(self) -> None:
        errors = validate_config(BotConfig(cfm_enabled=False, trading_modes=["cfm"]))

        assert "trading_modes includes 'cfm' but cfm_enabled is false" in errors

    def test_hybrid_trading_mode_requires_cfm_enabled(self) -> None:
        errors = validate_config(BotConfig(cfm_enabled=False, trading_modes=["spot", "cfm"]))

        assert "trading_modes includes 'cfm' but cfm_enabled is false" in errors

    def test_malformed_trading_modes_none_returns_validation_errors(self) -> None:
        errors = validate_config(BotConfig(cfm_enabled=True, trading_modes=None))  # type: ignore[arg-type]

        assert "trading_modes must be a list of mode names" in errors
        assert "cfm_enabled requires trading_modes to include 'cfm'" in errors

    def test_malformed_trading_modes_items_return_validation_error(self) -> None:
        errors = validate_config(BotConfig(cfm_enabled=True, trading_modes=["cfm", 1]))  # type: ignore[list-item]

        assert "trading_modes must contain only mode names" in errors

    def test_cfm_only_configuration_is_valid(self) -> None:
        errors = validate_config(BotConfig(cfm_enabled=True, trading_modes=["cfm"]))

        assert errors == []

    def test_hybrid_cfm_configuration_is_valid(self) -> None:
        errors = validate_config(BotConfig(cfm_enabled=True, trading_modes=["spot", "cfm"]))

        assert errors == []
