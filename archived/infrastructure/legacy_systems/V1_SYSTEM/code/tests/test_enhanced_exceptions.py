"""Unit tests for enhanced exception system."""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest
from bot.exceptions import (
    CircuitBreaker,
    CriticalError,
    ErrorContext,
    ErrorSeverity,
    # Handler and circuit breaker
    ExceptionHandler,
    # Enhanced exceptions
    GPTTraderException,
    InsufficientCapitalError,
    OrderRejectedError,
    RecoverableError,
    RecoveryStrategy,
    RetryableError,
    RiskLimitError,
    get_exception_handler,
    monitor_performance,
    safe_execution,
    validate_inputs,
    with_recovery,
    # Decorators
    with_retry,
)


class TestErrorContext:
    """Test ErrorContext class."""

    def test_error_context_creation(self):
        """Test creating error context."""
        context = ErrorContext(
            severity=ErrorSeverity.HIGH, component="test_component", operation="test_operation"
        )

        assert context.severity == ErrorSeverity.HIGH
        assert context.component == "test_component"
        assert context.operation == "test_operation"
        assert context.retry_count == 0
        assert context.max_retries == 3
        assert context.error_id is not None
        assert isinstance(context.timestamp, datetime)

    def test_error_context_to_dict(self):
        """Test converting context to dictionary."""
        context = ErrorContext(severity=ErrorSeverity.MEDIUM, component="trading")

        context_dict = context.to_dict()
        assert context_dict["severity"] == "medium"
        assert context_dict["component"] == "trading"
        assert "error_id" in context_dict
        assert "timestamp" in context_dict


class TestGPTTraderException:
    """Test base exception class."""

    def test_exception_creation(self):
        """Test creating exception."""
        exc = GPTTraderException("Test error", component="test", operation="testing")

        assert exc.message == "Test error"
        assert exc.context.component == "test"
        assert exc.context.operation == "testing"
        assert exc.recovery_strategy == RecoveryStrategy.RETRY

    def test_can_retry(self):
        """Test retry logic."""
        exc = GPTTraderException("Test error", recovery_strategy=RecoveryStrategy.RETRY)

        assert exc.can_retry() is True

        # Exhaust retries
        for _ in range(3):
            exc.increment_retry()

        assert exc.can_retry() is False

    def test_exception_to_dict(self):
        """Test converting exception to dictionary."""
        exc = GPTTraderException("Test error", component="test")

        exc_dict = exc.to_dict()
        assert exc_dict["type"] == "GPTTraderException"
        assert exc_dict["message"] == "Test error"
        assert exc_dict["recovery_strategy"] == "retry"
        assert "context" in exc_dict


class TestRecoverableError:
    """Test RecoverableError class."""

    def test_recoverable_error_with_fallback(self):
        """Test recoverable error with fallback value."""
        error = RecoverableError("Test error", fallback_value=42)

        result = error.recover()
        assert result == 42
        assert error.context.recovery_attempted is True

    def test_recoverable_error_with_recovery_action(self):
        """Test recoverable error with recovery action."""
        recovery_action = Mock(return_value="recovered")

        error = RecoverableError("Test error", recovery_action=recovery_action)

        result = error.recover()
        assert result == "recovered"
        assert recovery_action.called
        assert error.context.recovery_successful is True

    def test_recoverable_error_with_failed_recovery(self):
        """Test recoverable error when recovery fails."""
        recovery_action = Mock(side_effect=Exception("Recovery failed"))

        error = RecoverableError(
            "Test error", recovery_action=recovery_action, fallback_value="fallback"
        )

        result = error.recover()
        assert result == "fallback"
        assert error.context.recovery_attempted is True
        assert error.context.recovery_successful is False


class TestCriticalError:
    """Test CriticalError class."""

    def test_critical_error_creation(self):
        """Test creating critical error."""
        error = CriticalError("Critical failure", component="system")

        assert error.message == "Critical failure"
        assert error.context.severity == ErrorSeverity.CRITICAL
        assert error.recovery_strategy == RecoveryStrategy.ESCALATE

    def test_critical_error_with_shutdown(self):
        """Test critical error with shutdown action."""
        shutdown_action = Mock()

        error = CriticalError("Critical failure", shutdown_action=shutdown_action)

        error.shutdown()
        assert shutdown_action.called

    def test_critical_error_with_alerts(self):
        """Test critical error with alert channels."""
        with patch("bot.exceptions.enhanced_exceptions.logger") as mock_logger:
            error = CriticalError("Critical failure", alert_channels=["email", "slack"])

            # Check that alerts were sent
            assert mock_logger.critical.call_count >= 2


class TestRetryableError:
    """Test RetryableError class."""

    def test_retryable_error_backoff(self):
        """Test exponential backoff calculation."""
        error = RetryableError(
            "Network error", backoff_base=2.0, backoff_factor=1.0, max_backoff=10.0
        )

        # First retry: 1 * 2^0 = 1
        assert error.get_backoff_time() == 1.0

        error.increment_retry()
        # Second retry: 1 * 2^1 = 2
        assert error.get_backoff_time() == 2.0

        error.increment_retry()
        # Third retry: 1 * 2^2 = 4
        assert error.get_backoff_time() == 4.0

        # Test max backoff
        for _ in range(10):
            error.increment_retry()
        assert error.get_backoff_time() <= 10.0

    def test_wait_and_retry(self):
        """Test wait and retry logic."""
        error = RetryableError(
            "Network error",
            backoff_base=2.0,
            backoff_factor=0.01,  # Very short for testing
        )

        # Should be able to retry
        assert error.can_retry() is True

        # Test wait and retry
        start_time = time.time()
        result = error.wait_and_retry()
        elapsed = time.time() - start_time

        assert result is True
        assert elapsed >= 0.01  # At least the backoff time
        assert error.context.retry_count == 1


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        breaker = CircuitBreaker(
            "test_breaker", failure_threshold=2, timeout=timedelta(seconds=0.1)
        )

        # Initially closed
        assert breaker.state == "closed"
        assert breaker.is_open() is False

        # Record failures
        breaker.record_failure()
        assert breaker.state == "closed"  # Still closed after 1 failure

        breaker.record_failure()
        assert breaker.state == "open"  # Open after 2 failures
        assert breaker.is_open() is True

        # Wait for timeout
        time.sleep(0.11)
        assert breaker.is_open() is False  # Should be half-open
        assert breaker.state == "half_open"

        # Success in half-open closes circuit
        breaker.record_success()
        assert breaker.state == "closed"

    def test_circuit_breaker_half_open(self):
        """Test half-open state behavior."""
        breaker = CircuitBreaker(
            "test_breaker",
            failure_threshold=2,
            timeout=timedelta(seconds=0.1),
            half_open_requests=2,
        )

        # Open the circuit
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state == "open"

        # Wait for timeout
        time.sleep(0.11)
        breaker.is_open()  # Triggers state check
        assert breaker.state == "half_open"

        # One success not enough
        breaker.record_success()
        assert breaker.state == "half_open"

        # Two successes closes circuit
        breaker.record_success()
        assert breaker.state == "closed"


class TestExceptionHandler:
    """Test ExceptionHandler class."""

    def test_handle_recoverable_error(self):
        """Test handling recoverable error."""
        handler = ExceptionHandler()

        error = RecoverableError("Test error", fallback_value="recovered")

        result = handler.handle(error)
        assert result == "recovered"
        assert handler.recovery_stats["total_errors"] == 1
        assert handler.recovery_stats["recovered"] == 1

    def test_handle_retryable_error(self):
        """Test handling retryable error."""
        handler = ExceptionHandler()

        error = RetryableError("Network error", component="api")
        error.context.retry_count = 0  # Can retry

        with patch.object(error, "wait_and_retry", return_value=True):
            result = handler.handle(error)
            assert result == "retry"

    def test_handle_critical_error(self):
        """Test handling critical error."""
        handler = ExceptionHandler()

        shutdown_mock = Mock()
        error = CriticalError("System failure", shutdown_action=shutdown_mock)

        with patch.object(error, "shutdown"):
            result = handler.handle(error)
            assert result is None
            assert handler.recovery_stats["critical"] == 1

    def test_handler_statistics(self):
        """Test handler statistics."""
        handler = ExceptionHandler()

        # Handle various errors
        handler.handle(RecoverableError("Error 1", fallback_value=1))
        handler.handle(RecoverableError("Error 2", fallback_value=2))

        stats = handler.get_stats()
        assert stats["total_errors"] == 2
        assert stats["recovered"] == 2
        assert stats["recovery_rate"] == 1.0
        assert "recent_errors" in stats


class TestDecorators:
    """Test exception handling decorators."""

    def test_with_retry_decorator(self):
        """Test retry decorator."""
        call_count = 0

        @with_retry(max_retries=3, backoff_factor=0.01, exceptions=(ValueError,))
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_with_recovery_decorator(self):
        """Test recovery decorator."""

        @with_recovery(fallback_value="fallback")
        def failing_function():
            raise Exception("Always fails")

        result = failing_function()
        assert result == "fallback"

    def test_with_recovery_fallback_function(self):
        """Test recovery with fallback function."""

        @with_recovery(fallback_function=lambda: "recovered")
        def failing_function():
            raise Exception("Always fails")

        result = failing_function()
        assert result == "recovered"

    def test_safe_execution_decorator(self):
        """Test safe execution decorator."""

        @safe_execution(default_return=0)
        def risky_function(x):
            if x == 0:
                raise ValueError("Cannot process zero")
            return 10 / x

        assert risky_function(2) == 5.0
        assert risky_function(0) == 0  # Returns default

    def test_monitor_performance_decorator(self):
        """Test performance monitoring decorator."""
        with patch("bot.exceptions.decorators.logger") as mock_logger:

            @monitor_performance(slow_threshold_seconds=0.01)
            def slow_function():
                time.sleep(0.02)
                return "done"

            result = slow_function()
            assert result == "done"
            # Should log warning for slow execution
            mock_logger.warning.assert_called()

    def test_validate_inputs_decorator(self):
        """Test input validation decorator."""

        @validate_inputs(x=lambda v: v > 0, y=lambda v: isinstance(v, str))
        def validated_function(x, y):
            return f"{y}: {x}"

        # Valid inputs
        assert validated_function(5, "value") == "value: 5"

        # Invalid inputs
        with pytest.raises(ValueError):
            validated_function(-1, "value")  # x <= 0

        with pytest.raises(ValueError):
            validated_function(5, 123)  # y not string


class TestTradingExceptions:
    """Test trading-specific exceptions."""

    def test_insufficient_capital_error(self):
        """Test insufficient capital error."""
        error = InsufficientCapitalError(
            "Not enough money", required_capital=10000, available_capital=5000, symbol="AAPL"
        )

        assert error.required_capital == 10000
        assert error.available_capital == 5000
        assert error.symbol == "AAPL"
        assert error.context.additional_info["shortfall"] == 5000

    def test_order_rejected_error(self):
        """Test order rejected error."""
        error = OrderRejectedError(
            "Order rejected", order_id="ORD-123", reason="Market closed", symbol="GOOGL"
        )

        assert error.order_id == "ORD-123"
        assert error.reason == "Market closed"
        assert error.symbol == "GOOGL"

    def test_risk_limit_error(self):
        """Test risk limit error."""
        error = RiskLimitError(
            "Risk too high", limit_type="daily_loss", limit_value=1000, current_value=1500
        )

        assert error.limit_type == "daily_loss"
        assert error.limit_value == 1000
        assert error.current_value == 1500
        assert error.context.additional_info["breach_amount"] == 500
        assert error.context.severity == ErrorSeverity.CRITICAL


class TestSingleton:
    """Test singleton pattern."""

    def test_exception_handler_singleton(self):
        """Test that exception handler is singleton."""
        handler1 = get_exception_handler()
        handler2 = get_exception_handler()

        assert handler1 is handler2
