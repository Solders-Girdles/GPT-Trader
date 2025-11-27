"""Tests for orchestration configuration validation module."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from gpt_trader.orchestration.configuration.validation import (
    ConfigValidationError,
    ConfigValidationResult,
    format_validation_errors,
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
