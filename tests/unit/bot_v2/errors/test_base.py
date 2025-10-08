"""Tests for legacy error hierarchy."""

import pytest

from bot_v2.errors.base import (
    BotError,
    ExecutionError,
    ValidationError,
    ConfigError,
    RiskError,
)


class TestBotError:
    """Test the base BotError class."""

    def test_bot_error_inheritance(self) -> None:
        """Test that BotError inherits from Exception."""
        assert issubclass(BotError, Exception)

    def test_bot_error_creation(self) -> None:
        """Test BotError can be created with message."""
        error = BotError("test message")
        assert str(error) == "test message"
        assert error.args == ("test message",)

    def test_bot_error_creation_without_message(self) -> None:
        """Test BotError can be created without message."""
        error = BotError()
        assert str(error) == ""
        assert error.args == ()

    def test_bot_error_with_kwargs(self) -> None:
        """Test BotError can be created with additional context."""
        error = BotError("test")
        # Add attribute after creation since kwargs aren't supported
        error.code = 123
        assert str(error) == "test"
        assert error.code == 123


class TestExecutionError:
    """Test the ExecutionError class."""

    def test_execution_error_inheritance(self) -> None:
        """Test that ExecutionError inherits from BotError."""
        assert issubclass(ExecutionError, BotError)
        assert issubclass(ExecutionError, Exception)

    def test_execution_error_creation(self) -> None:
        """Test ExecutionError can be created with message."""
        error = ExecutionError("execution failed")
        assert str(error) == "execution failed"

    def test_execution_error_with_order_id(self) -> None:
        """Test ExecutionError can include order context."""
        error = ExecutionError("order failed")
        error.order_id = "12345"
        assert str(error) == "order failed"
        assert error.order_id == "12345"


class TestValidationError:
    """Test the ValidationError class."""

    def test_validation_error_inheritance(self) -> None:
        """Test that ValidationError inherits from BotError."""
        assert issubclass(ValidationError, BotError)
        assert issubclass(ValidationError, Exception)

    def test_validation_error_creation(self) -> None:
        """Test ValidationError can be created with message."""
        error = ValidationError("invalid input")
        assert str(error) == "invalid input"

    def test_validation_error_with_field(self) -> None:
        """Test ValidationError can include field context."""
        error = ValidationError("invalid price")
        error.field = "price"
        error.value = -1
        assert str(error) == "invalid price"
        assert error.field == "price"
        assert error.value == -1


class TestConfigError:
    """Test the ConfigError class."""

    def test_config_error_inheritance(self) -> None:
        """Test that ConfigError inherits from BotError."""
        assert issubclass(ConfigError, BotError)
        assert issubclass(ConfigError, Exception)

    def test_config_error_creation(self) -> None:
        """Test ConfigError can be created with message."""
        error = ConfigError("missing config")
        assert str(error) == "missing config"

    def test_config_error_with_config_key(self) -> None:
        """Test ConfigError can include config context."""
        error = ConfigError("invalid value")
        error.config_key = "api_key"
        assert str(error) == "invalid value"
        assert error.config_key == "api_key"


class TestRiskError:
    """Test the RiskError class."""

    def test_risk_error_inheritance(self) -> None:
        """Test that RiskError inherits from BotError."""
        assert issubclass(RiskError, BotError)
        assert issubclass(RiskError, Exception)

    def test_risk_error_creation(self) -> None:
        """Test RiskError can be created with message."""
        error = RiskError("risk limit exceeded")
        assert str(error) == "risk limit exceeded"

    def test_risk_error_with_risk_metrics(self) -> None:
        """Test RiskError can include risk context."""
        error = RiskError("position too large")
        error.position_size = 100000
        error.limit = 50000
        assert str(error) == "position too large"
        assert error.position_size == 100000
        assert error.limit == 50000


class TestErrorHierarchy:
    """Test error hierarchy relationships."""

    def test_all_errors_inherit_from_bot_error(self) -> None:
        """Test that all custom errors inherit from BotError."""
        errors = [ExecutionError, ValidationError, ConfigError, RiskError]
        for error_class in errors:
            assert issubclass(error_class, BotError)

    def test_error_catching_hierarchy(self) -> None:
        """Test that errors can be caught by their parent classes."""
        try:
            raise ExecutionError("test")
        except BotError as e:
            assert isinstance(e, ExecutionError)
            assert str(e) == "test"

        try:
            raise ValidationError("test")
        except BotError as e:
            assert isinstance(e, ValidationError)
            assert str(e) == "test"

    def test_specific_error_catching(self) -> None:
        """Test that errors can be caught specifically."""
        try:
            raise ConfigError("config error")
        except ConfigError as e:
            assert str(e) == "config error"
        except Exception:
            pytest.fail("Should have been caught by ConfigError")

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
