"""Tests for legacy error hierarchy."""

import pytest

from bot_v2.errors.base import (
    BotError,
    ConfigError,
    ExecutionError,
    RiskError,
    ValidationError,
)


class TestBotError:
    """Test BotError base exception."""

    def test_can_raise_bot_error(self):
        """BotError can be raised."""
        with pytest.raises(BotError):
            raise BotError("test error")

    def test_bot_error_message(self):
        """BotError preserves message."""
        msg = "specific error message"
        with pytest.raises(BotError, match=msg):
            raise BotError(msg)

    def test_bot_error_is_exception(self):
        """BotError is an Exception."""
        assert issubclass(BotError, Exception)


class TestExecutionError:
    """Test ExecutionError exception."""

    def test_can_raise_execution_error(self):
        """ExecutionError can be raised."""
        with pytest.raises(ExecutionError):
            raise ExecutionError("execution failed")

    def test_execution_error_message(self):
        """ExecutionError preserves message."""
        msg = "order execution failed"
        with pytest.raises(ExecutionError, match=msg):
            raise ExecutionError(msg)

    def test_execution_error_is_bot_error(self):
        """ExecutionError is a BotError."""
        assert issubclass(ExecutionError, BotError)

    def test_can_catch_as_bot_error(self):
        """ExecutionError can be caught as BotError."""
        with pytest.raises(BotError):
            raise ExecutionError("test")


class TestValidationError:
    """Test ValidationError exception."""

    def test_can_raise_validation_error(self):
        """ValidationError can be raised."""
        with pytest.raises(ValidationError):
            raise ValidationError("validation failed")

    def test_validation_error_message(self):
        """ValidationError preserves message."""
        msg = "invalid parameter"
        with pytest.raises(ValidationError, match=msg):
            raise ValidationError(msg)

    def test_validation_error_is_bot_error(self):
        """ValidationError is a BotError."""
        assert issubclass(ValidationError, BotError)

    def test_can_catch_as_bot_error(self):
        """ValidationError can be caught as BotError."""
        with pytest.raises(BotError):
            raise ValidationError("test")


class TestConfigError:
    """Test ConfigError exception."""

    def test_can_raise_config_error(self):
        """ConfigError can be raised."""
        with pytest.raises(ConfigError):
            raise ConfigError("config failed")

    def test_config_error_message(self):
        """ConfigError preserves message."""
        msg = "invalid configuration"
        with pytest.raises(ConfigError, match=msg):
            raise ConfigError(msg)

    def test_config_error_is_bot_error(self):
        """ConfigError is a BotError."""
        assert issubclass(ConfigError, BotError)

    def test_can_catch_as_bot_error(self):
        """ConfigError can be caught as BotError."""
        with pytest.raises(BotError):
            raise ConfigError("test")


class TestRiskError:
    """Test RiskError exception."""

    def test_can_raise_risk_error(self):
        """RiskError can be raised."""
        with pytest.raises(RiskError):
            raise RiskError("risk check failed")

    def test_risk_error_message(self):
        """RiskError preserves message."""
        msg = "risk limit exceeded"
        with pytest.raises(RiskError, match=msg):
            raise RiskError(msg)

    def test_risk_error_is_bot_error(self):
        """RiskError is a BotError."""
        assert issubclass(RiskError, BotError)

    def test_can_catch_as_bot_error(self):
        """RiskError can be caught as BotError."""
        with pytest.raises(BotError):
            raise RiskError("test")
