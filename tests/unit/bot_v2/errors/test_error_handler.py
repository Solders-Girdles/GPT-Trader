"""Tests for error handler with retry and circuit breaker logic."""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from bot_v2.errors import NetworkError, TimeoutError, TradingError
from bot_v2.errors.handler import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ErrorHandler,
    RecoveryStrategy,
    RetryConfig,
    get_error_handler,
    set_error_handler,
    with_error_handling,
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_values(self):
        """Test default retry configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            initial_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=False,
        )
        assert config.max_attempts == 5
        assert config.initial_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter is False


class TestCircuitBreakerConfig:
    """Test circuit breaker configuration."""

    def test_default_values(self):
        """Test default circuit breaker configuration."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.expected_exception_types == (NetworkError, TimeoutError)

    def test_custom_values(self):
        """Test custom circuit breaker configuration."""
        config = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout=30.0, expected_exception_types=(TradingError,)
        )
        assert config.failure_threshold == 3
        assert config.recovery_timeout == 30.0
        assert config.expected_exception_types == (TradingError,)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in closed state."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0
        assert cb.should_attempt_call() is True

    def test_record_success_resets_failure_count(self):
        """Test that successful call resets failure count."""
        cb = CircuitBreaker(CircuitBreakerConfig())
        cb.failure_count = 3
        cb.record_success()
        assert cb.failure_count == 0

    def test_circuit_opens_after_threshold_failures(self):
        """Test that circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker(config)

        for _ in range(2):
            cb.record_failure(NetworkError("Test error"))
            assert cb.state == CircuitBreakerState.CLOSED

        cb.record_failure(NetworkError("Test error"))
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.should_attempt_call() is False

    def test_ignores_unexpected_errors(self):
        """Test that circuit breaker ignores unexpected error types."""
        config = CircuitBreakerConfig(expected_exception_types=(NetworkError,))
        cb = CircuitBreaker(config)

        cb.record_failure(ValueError("Not a network error"))
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED

    def test_transitions_to_half_open_after_timeout(self):
        """Test that circuit transitions to half-open after recovery timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(config)

        cb.record_failure(NetworkError("Test error"))
        assert cb.state == CircuitBreakerState.OPEN

        time.sleep(0.15)  # Wait for recovery timeout
        assert cb.should_attempt_call() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN

    def test_closes_after_success_in_half_open(self):
        """Test that circuit closes after 3 successes in half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker(config)

        cb.record_failure(NetworkError("Test error"))
        assert cb.state == CircuitBreakerState.OPEN

        time.sleep(0.15)
        cb.should_attempt_call()  # Transition to half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Need 3 successes to close
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED


class TestErrorHandler:
    """Test error handler with retry logic."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retry."""
        handler = ErrorHandler()
        mock_func = Mock(return_value="success")

        result = handler.with_retry(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retries_on_recoverable_error(self):
        """Test that handler retries on recoverable errors."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        handler = ErrorHandler(retry_config=config)

        mock_func = Mock(side_effect=[NetworkError("Error 1"), NetworkError("Error 2"), "success"])

        result = handler.with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 3

    def test_fails_fast_on_non_recoverable_error(self):
        """Test that handler fails fast on non-recoverable errors."""
        handler = ErrorHandler()

        error = TradingError("Fatal error", recoverable=False)
        mock_func = Mock(side_effect=error)

        with pytest.raises(TradingError) as exc_info:
            handler.with_retry(mock_func)

        assert exc_info.value.recoverable is False
        assert mock_func.call_count == 1

    def test_exhausts_retries_and_raises(self):
        """Test that all retries are exhausted before raising."""
        config = RetryConfig(max_attempts=3, initial_delay=0.01, jitter=False)
        handler = ErrorHandler(retry_config=config)

        mock_func = Mock(side_effect=NetworkError("Persistent error"))

        with pytest.raises(TradingError) as exc_info:
            handler.with_retry(mock_func)

        assert "retry attempts failed" in str(exc_info.value)
        assert mock_func.call_count == 3

    def test_exponential_backoff_delay(self):
        """Test that delay increases exponentially."""
        config = RetryConfig(initial_delay=0.1, exponential_base=2.0, jitter=False)
        handler = ErrorHandler(retry_config=config)

        delay1 = handler._calculate_delay(1)
        delay2 = handler._calculate_delay(2)
        delay3 = handler._calculate_delay(3)

        assert delay1 == 0.1
        assert delay2 == 0.2
        assert delay3 == 0.4

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(initial_delay=1.0, max_delay=5.0, exponential_base=2.0, jitter=False)
        handler = ErrorHandler(retry_config=config)

        delay10 = handler._calculate_delay(10)  # Would be 512 without cap
        assert delay10 == 5.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay."""
        config = RetryConfig(initial_delay=1.0, jitter=True)
        handler = ErrorHandler(retry_config=config)

        delays = [handler._calculate_delay(1) for _ in range(10)]

        # All delays should be different due to jitter
        assert len(set(delays)) > 1
        # All should be between 0.5 and 1.5 (1.0 * (0.5 to 1.5))
        assert all(0.5 <= d <= 1.5 for d in delays)

    def test_fallback_strategy(self):
        """Test fallback recovery strategy."""
        fallback_func = Mock(return_value="fallback_result")
        handler = ErrorHandler(fallback_handlers={NetworkError: fallback_func})

        mock_func = Mock(side_effect=NetworkError("Network down"))

        result = handler.with_retry(
            mock_func, "arg1", recovery_strategy=RecoveryStrategy.FALLBACK, kwarg1="value1"
        )

        assert result == "fallback_result"
        fallback_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_fail_fast_strategy(self):
        """Test fail-fast recovery strategy."""
        config = RetryConfig(max_attempts=3)
        handler = ErrorHandler(retry_config=config)

        mock_func = Mock(side_effect=NetworkError("Error"))

        with pytest.raises(TradingError):
            handler.with_retry(mock_func, recovery_strategy=RecoveryStrategy.FAIL_FAST)

        # Should fail on first attempt, not retry
        assert mock_func.call_count == 1

    def test_circuit_breaker_integration(self):
        """Test that circuit breaker opens after failures."""
        cb_config = CircuitBreakerConfig(failure_threshold=2)
        retry_config = RetryConfig(max_attempts=1, initial_delay=0.01)  # Fail fast to avoid retry delays
        handler = ErrorHandler(circuit_breaker_config=cb_config, retry_config=retry_config)

        mock_func = Mock(side_effect=NetworkError("Error"))

        # First call fails - records 1 failure
        with pytest.raises(TradingError):
            handler.with_retry(mock_func)

        assert handler.circuit_breaker.failure_count == 1
        assert handler.circuit_breaker.state == CircuitBreakerState.CLOSED

        # Second call fails - should open circuit
        with pytest.raises(TradingError):
            handler.with_retry(Mock(side_effect=NetworkError("Error")))

        assert handler.circuit_breaker.state == CircuitBreakerState.OPEN

        # Third call should fail immediately due to open circuit
        with pytest.raises(TradingError) as exc_info:
            handler.with_retry(Mock())

        assert "Circuit breaker is open" in str(exc_info.value)

    def test_error_history_tracking(self):
        """Test that errors are tracked in history."""
        handler = ErrorHandler()

        mock_func = Mock(side_effect=[NetworkError("Error 1"), "success"])
        handler.with_retry(mock_func)

        assert len(handler.error_history) == 1
        assert "Error 1" in str(handler.error_history[0])

    def test_error_history_size_limit(self):
        """Test that error history is limited."""
        handler = ErrorHandler()
        handler.max_history = 5

        for i in range(10):
            error = TradingError(f"Error {i}", recoverable=True)
            handler._record_error(error)

        assert len(handler.error_history) == 5
        # Should keep most recent errors
        assert "Error 9" in str(handler.error_history[-1])

    def test_get_error_stats_empty(self):
        """Test error stats when no errors occurred."""
        handler = ErrorHandler()
        stats = handler.get_error_stats()

        assert stats["total_errors"] == 0
        assert stats["error_types"] == {}
        assert stats["circuit_breaker_state"] == "closed"

    def test_get_error_stats_with_errors(self):
        """Test error stats with error history."""
        handler = ErrorHandler()

        handler._record_error(TradingError("Error 1", error_code="NET_001"))
        handler._record_error(TradingError("Error 2", error_code="NET_001"))
        handler._record_error(TradingError("Error 3", error_code="TIMEOUT"))

        stats = handler.get_error_stats()

        assert stats["total_errors"] == 3
        assert stats["error_types"]["NET_001"] == 2
        assert stats["error_types"]["TIMEOUT"] == 1
        assert stats["last_error"]["error_code"] == "TIMEOUT"

    def test_reset_circuit_breaker(self):
        """Test manual circuit breaker reset."""
        cb_config = CircuitBreakerConfig(failure_threshold=1)
        handler = ErrorHandler(circuit_breaker_config=cb_config)

        # Open the circuit
        handler.circuit_breaker.record_failure(NetworkError("Error"))
        assert handler.circuit_breaker.state == CircuitBreakerState.OPEN

        # Reset it
        handler.reset_circuit_breaker()
        assert handler.circuit_breaker.state == CircuitBreakerState.CLOSED
        assert handler.circuit_breaker.failure_count == 0

    def test_handle_error_with_degrade_strategy(self):
        """Test error handling with DEGRADE strategy."""
        handler = ErrorHandler()

        result = handler.handle_error(
            NetworkError("Service down"), recovery_strategy=RecoveryStrategy.DEGRADE
        )

        assert result is None  # Returns None to indicate degraded operation
        assert len(handler.error_history) == 1


class TestGlobalErrorHandler:
    """Test global error handler singleton."""

    def test_get_error_handler_creates_instance(self):
        """Test that get_error_handler creates a singleton instance."""
        handler1 = get_error_handler()
        handler2 = get_error_handler()

        assert handler1 is handler2

    def test_set_error_handler(self):
        """Test setting custom error handler."""
        custom_handler = ErrorHandler(retry_config=RetryConfig(max_attempts=10))
        set_error_handler(custom_handler)

        retrieved = get_error_handler()
        assert retrieved is custom_handler
        assert retrieved.retry_config.max_attempts == 10


class TestErrorHandlingDecorator:
    """Test error handling decorator."""

    def test_decorator_with_retry(self):
        """Test decorator applies retry logic."""
        call_count = 0

        @with_error_handling(recovery_strategy=RecoveryStrategy.RETRY)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        # Reset global handler
        set_error_handler(ErrorHandler(retry_config=RetryConfig(initial_delay=0.01, jitter=False)))

        result = flaky_function()
        assert result == "success"
        assert call_count == 3

    def test_decorator_with_fallback(self):
        """Test decorator with fallback function."""

        def fallback_impl():
            return "fallback_result"

        @with_error_handling(recovery_strategy=RecoveryStrategy.FALLBACK, fallback=fallback_impl)
        def failing_function():
            raise NetworkError("Always fails")

        result = failing_function()
        assert result == "fallback_result"


class TestRecoveryStrategy:
    """Test recovery strategy enum."""

    def test_all_strategies_defined(self):
        """Test that all recovery strategies are defined."""
        assert RecoveryStrategy.RETRY.value == "retry"
        assert RecoveryStrategy.FALLBACK.value == "fallback"
        assert RecoveryStrategy.CIRCUIT_BREAK.value == "circuit_break"
        assert RecoveryStrategy.FAIL_FAST.value == "fail_fast"
        assert RecoveryStrategy.DEGRADE.value == "degrade"
