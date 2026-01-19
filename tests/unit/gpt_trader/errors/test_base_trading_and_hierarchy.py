"""Tests for TradingError base class and hierarchy helpers."""

from __future__ import annotations

import pytest

from gpt_trader.errors import (
    ConfigurationError,
    ExecutionError,
    TradingError,
    ValidationError,
    handle_error,
)


class TestTradingError:
    """Test the base TradingError class."""

    def test_trading_error_inheritance(self) -> None:
        """Test that TradingError inherits from Exception."""
        assert issubclass(TradingError, Exception)

    def test_trading_error_creation(self) -> None:
        """Test TradingError can be created with message."""
        error = TradingError("test message")
        assert str(error) == "test message"
        assert error.args == ("test message",)

    def test_trading_error_creation_without_message(self) -> None:
        """Test TradingError requires a message."""
        # TradingError now requires a message argument
        error = TradingError("")
        assert str(error) == ""

    def test_trading_error_with_context(self) -> None:
        """Test TradingError supports context."""
        error = TradingError("test", context={"key": "value"})
        assert str(error) == "test"
        assert error.context == {"key": "value"}

    def test_trading_error_captures_stack_when_no_exception(self) -> None:
        """Ensure stack trace is captured even when instantiated outside except blocks."""
        error = TradingError("stack capture check")
        assert "test_trading_error_captures_stack_when_no_exception" in error.traceback
        assert "NoneType: None" not in error.traceback


class TestHandleError:
    """Tests for the handle_error helper."""

    def test_handle_error_wraps_exception_with_traceback(self) -> None:
        """Ensure wrapped exceptions retain traceback information."""
        try:
            raise ValueError("boom")
        except ValueError as exc:
            wrapped = handle_error(exc, context={"example": "value"})

        assert isinstance(wrapped, TradingError)
        assert wrapped.error_code == "ValueError"
        assert wrapped.context["example"] == "value"
        assert "ValueError: boom" in wrapped.traceback


class TestErrorHierarchy:
    """Test error hierarchy relationships."""

    def test_error_catching_hierarchy(self) -> None:
        """Test that errors can be caught by their parent classes."""
        try:
            raise ExecutionError("test")
        except TradingError as e:
            assert isinstance(e, ExecutionError)
            assert str(e) == "test"

        try:
            raise ValidationError("test")
        except TradingError as e:
            assert isinstance(e, ValidationError)
            assert str(e) == "test"

    def test_specific_error_catching(self) -> None:
        """Test that errors can be caught specifically."""
        try:
            raise ConfigurationError("config error")
        except ConfigurationError as e:
            assert str(e) == "config error"
        except Exception:
            pytest.fail("Should have been caught by ConfigurationError")

    def test_exception_chaining(self) -> None:
        """Test that errors support exception chaining."""
        original_error = ValueError("original")
        try:
            raise original_error
        except ValueError:
            wrapped_error = ValidationError("validation failed")
            # Manually set the cause for compatibility
            wrapped_error.__cause__ = original_error

        assert wrapped_error.__cause__ is original_error
        assert str(wrapped_error) == "validation failed"
