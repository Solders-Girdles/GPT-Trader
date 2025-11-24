"""Tests for comprehensive error hierarchy."""

import pytest

from gpt_trader.errors import (
    BacktestError,
    ConfigurationError,
    DataError,
    ExecutionError,
    InsufficientFundsError,
    NetworkError,
    RiskLimitExceeded,
    StrategyError,
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


class TestValidationError:
    """Test the ValidationError class."""

    def test_validation_error_inheritance(self) -> None:
        """Test that ValidationError inherits from TradingError."""
        assert issubclass(ValidationError, TradingError)
        assert issubclass(ValidationError, Exception)

    def test_validation_error_creation(self) -> None:
        """Test ValidationError can be created with message."""
        error = ValidationError("invalid input")
        assert str(error) == "invalid input"

    def test_validation_error_with_field(self) -> None:
        """Test ValidationError can include field context."""
        error = ValidationError("invalid price", field="price", value=-1)
        assert str(error) == "invalid price"
        assert error.context["field"] == "price"
        assert error.context["value"] == -1


class TestExecutionError:
    """Test the ExecutionError class."""

    def test_execution_error_inheritance(self) -> None:
        """Test that ExecutionError inherits from TradingError."""
        assert issubclass(ExecutionError, TradingError)
        assert issubclass(ExecutionError, Exception)

    def test_execution_error_creation(self) -> None:
        """Test ExecutionError can be created with message."""
        error = ExecutionError("execution failed")
        assert str(error) == "execution failed"

    def test_execution_error_with_order_id(self) -> None:
        """Test ExecutionError can include order context."""
        error = ExecutionError("order failed", order_id="12345")
        assert str(error) == "order failed"
        assert error.context["order_id"] == "12345"


class TestConfigurationError:
    """Test the ConfigurationError class."""

    def test_configuration_error_inheritance(self) -> None:
        """Test that ConfigurationError inherits from TradingError."""
        assert issubclass(ConfigurationError, TradingError)
        assert issubclass(ConfigurationError, Exception)

    def test_configuration_error_creation(self) -> None:
        """Test ConfigurationError can be created with message."""
        error = ConfigurationError("missing config")
        assert str(error) == "missing config"

    def test_configuration_error_with_config_key(self) -> None:
        """Test ConfigurationError can include config context."""
        error = ConfigurationError("invalid value", config_key="api_key")
        assert str(error) == "invalid value"
        assert error.context["config_key"] == "api_key"


class TestRiskLimitExceeded:
    """Test the RiskLimitExceeded class."""

    def test_risk_limit_exceeded_inheritance(self) -> None:
        """Test that RiskLimitExceeded inherits from TradingError."""
        assert issubclass(RiskLimitExceeded, TradingError)
        assert issubclass(RiskLimitExceeded, Exception)

    def test_risk_limit_exceeded_creation(self) -> None:
        """Test RiskLimitExceeded can be created with message."""
        error = RiskLimitExceeded(
            "risk limit exceeded", limit_type="position", limit_value=50000, current_value=100000
        )
        assert str(error) == "risk limit exceeded"

    def test_risk_limit_exceeded_with_risk_metrics(self) -> None:
        """Test RiskLimitExceeded can include risk context."""
        error = RiskLimitExceeded(
            "position too large",
            limit_type="position_size",
            limit_value=50000,
            current_value=100000,
        )
        assert str(error) == "position too large"
        assert error.context["limit_value"] == 50000
        assert error.context["current_value"] == 100000


class TestDataError:
    """Test the DataError class."""

    def test_data_error_inheritance(self) -> None:
        """Test that DataError inherits from TradingError."""
        assert issubclass(DataError, TradingError)

    def test_data_error_creation(self) -> None:
        """Test DataError can be created with message."""
        error = DataError("data issue")
        assert str(error) == "data issue"


class TestStrategyError:
    """Test the StrategyError class."""

    def test_strategy_error_inheritance(self) -> None:
        """Test that StrategyError inherits from TradingError."""
        assert issubclass(StrategyError, TradingError)

    def test_strategy_error_creation(self) -> None:
        """Test StrategyError can be created with message."""
        error = StrategyError("strategy failed")
        assert str(error) == "strategy failed"


class TestBacktestError:
    """Test the BacktestError class."""

    def test_backtest_error_inheritance(self) -> None:
        """Test that BacktestError inherits from TradingError."""
        assert issubclass(BacktestError, TradingError)

    def test_backtest_error_creation(self) -> None:
        """Test BacktestError can be created with message."""
        error = BacktestError("backtest failed")
        assert str(error) == "backtest failed"


class TestNetworkError:
    """Test the NetworkError class."""

    def test_network_error_inheritance(self) -> None:
        """Test that NetworkError inherits from TradingError."""
        assert issubclass(NetworkError, TradingError)

    def test_network_error_creation(self) -> None:
        """Test NetworkError can be created with message."""
        error = NetworkError("network issue")
        assert str(error) == "network issue"


class TestInsufficientFundsError:
    """Test the InsufficientFundsError class."""

    def test_insufficient_funds_error_inheritance(self) -> None:
        """Test that InsufficientFundsError inherits from TradingError."""
        assert issubclass(InsufficientFundsError, TradingError)

    def test_insufficient_funds_error_creation(self) -> None:
        """Test InsufficientFundsError can be created with message."""
        error = InsufficientFundsError("insufficient funds", required=1000.0, available=500.0)
        assert str(error) == "insufficient funds"
        assert error.context["required"] == 1000.0
        assert error.context["available"] == 500.0
        assert error.context["shortfall"] == 500.0


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

    def test_all_errors_inherit_from_trading_error(self) -> None:
        """Test that all custom errors inherit from TradingError."""
        errors = [
            ValidationError,
            ExecutionError,
            ConfigurationError,
            RiskLimitExceeded,
            DataError,
            StrategyError,
            BacktestError,
            NetworkError,
            InsufficientFundsError,
        ]
        for error_class in errors:
            assert issubclass(error_class, TradingError)

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
