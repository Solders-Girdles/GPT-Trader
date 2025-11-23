"""
Error handler with recovery strategies and circuit breaker

Provides intelligent error handling with retry logic, exponential backoff,
and circuit breaker pattern for fault tolerance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

from bot_v2.errors import (
    NetworkError,
    TimeoutError,
    TradingError,
    handle_error,
    log_error,
)

# REMOVED: from bot_v2.logging import get_log_context, get_orchestration_logger
# These are now imported lazily inside functions to break circular import
from bot_v2.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="error_handler")
# json_logger = get_orchestration_logger("error_handler")  # Removed, created lazily

P = ParamSpec("P")
T = TypeVar("T")


# Lazy import helpers to break circular dependency
def _get_json_logger():
    """Lazily get orchestration logger to avoid circular import."""
    from bot_v2.logging import get_orchestration_logger

    return get_orchestration_logger("error_handler")


def _get_log_context():
    """Lazily get log context to avoid circular import."""
    from bot_v2.logging import get_log_context

    return get_log_context()


class RecoveryStrategy(Enum):
    """Available recovery strategies"""

    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    FAIL_FAST = "fail_fast"
    DEGRADE = "degrade"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""

    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception_types: tuple[type[Exception], ...] = (NetworkError, TimeoutError)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation"""

    config: CircuitBreakerConfig
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: datetime | None = None
    success_count: int = 0

    def record_success(self) -> None:
        """Record a successful call"""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitBreakerState.CLOSED
                self.success_count = 0
                logger.info("Circuit breaker closed after successful recovery")
                _get_json_logger().info(
                    "Circuit breaker closed after successful recovery",
                    extra={"circuit_breaker_state": "closed", "event": "recovery"},
                )

    def record_failure(self, error: Exception) -> None:
        """Record a failed call"""
        if not isinstance(error, self.config.expected_exception_types):
            return  # Don't trip on unexpected errors

        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            _get_json_logger().warning(
                f"Circuit breaker opened after {self.failure_count} failures",
                extra={
                    "circuit_breaker_state": "open",
                    "failure_count": self.failure_count,
                    "error_type": error.__class__.__name__,
                },
            )

    def should_attempt_call(self) -> bool:
        """Check if we should attempt the call"""
        if self.state == CircuitBreakerState.CLOSED:
            return True

        if self.state == CircuitBreakerState.OPEN:
            # Check if we should try half-open
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure > self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    logger.info("Circuit breaker entering half-open state")
                    _get_json_logger().info(
                        "Circuit breaker entering half-open state",
                        extra={"circuit_breaker_state": "half_open", "event": "recovery_attempt"},
                    )
                    return True
            return False

        # Half-open state
        return True


class ErrorHandler:
    """Intelligent error handler with recovery strategies"""

    def __init__(
        self,
        retry_config: RetryConfig | None = None,
        circuit_breaker_config: CircuitBreakerConfig | None = None,
        fallback_handlers: dict[type[BaseException], Callable[..., Any]] | None = None,
    ) -> None:
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config or CircuitBreakerConfig())
        self.fallback_handlers: dict[type[BaseException], Callable[..., Any]] = (
            fallback_handlers.copy() if fallback_handlers else {}
        )
        self.error_history: list[TradingError] = []
        self.max_history = 100

    def with_retry(
        self,
        func: Callable[..., T],
        *args: Any,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry logic and error handling"""

        # Check circuit breaker
        if not self.circuit_breaker.should_attempt_call():
            raise TradingError(
                "Circuit breaker is open - service temporarily unavailable",
                error_code="CIRCUIT_BREAKER_OPEN",
                recoverable=False,
            )

        last_error: TradingError | None = None
        attempt = 0

        while attempt < self.retry_config.max_attempts:
            try:
                result = func(*args, **kwargs)
                self.circuit_breaker.record_success()
                return result

            except Exception as e:
                attempt += 1
                last_error = handle_error(e)
                self._record_error(last_error)

                # Check if error is recoverable
                if not last_error.recoverable:
                    self.circuit_breaker.record_failure(e)
                    raise last_error

                # Apply recovery strategy
                if recovery_strategy == RecoveryStrategy.FAIL_FAST:
                    self.circuit_breaker.record_failure(e)
                    raise last_error

                elif recovery_strategy == RecoveryStrategy.FALLBACK:
                    fallback = self._get_fallback(type(e))
                    if fallback:
                        logger.info(f"Using fallback for {e.__class__.__name__}")
                        _get_json_logger().info(
                            f"Using fallback for {e.__class__.__name__}",
                            extra={
                                "fallback_used": True,
                                "error_type": e.__class__.__name__,
                                "recovery_strategy": "fallback",
                            },
                        )
                        return cast(T, fallback(*args, **kwargs))

                # Calculate retry delay
                if attempt < self.retry_config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt}/{self.retry_config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    _get_json_logger().warning(
                        f"Attempt {attempt}/{self.retry_config.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.2f} seconds...",
                        extra={
                            "attempt": attempt,
                            "max_attempts": self.retry_config.max_attempts,
                            "error_type": e.__class__.__name__,
                            "retry_delay": delay,
                            "recovery_strategy": "retry",
                        },
                    )
                    time.sleep(delay)

        # All retries exhausted
        if last_error is None:
            last_error = TradingError(
                "Retry attempts exhausted without captured TradingError",
                error_code="UNKNOWN_RETRY_EXHAUSTION",
                recoverable=False,
            )
        self.circuit_breaker.record_failure(last_error)
        raise TradingError(
            f"All {self.retry_config.max_attempts} retry attempts failed",
            error_code="RETRY_EXHAUSTED",
            recoverable=False,
            context={"last_error": str(last_error), "attempts": attempt},
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.retry_config.initial_delay * (self.retry_config.exponential_base ** (attempt - 1)),
            self.retry_config.max_delay,
        )

        if self.retry_config.jitter:
            import random

            delay *= 0.5 + random.random()  # nosec B311

        return delay

    def _get_fallback(self, error_type: type[BaseException]) -> Callable[..., Any] | None:
        """Get fallback handler for error type"""
        for error_class, handler in self.fallback_handlers.items():
            if issubclass(error_type, error_class):
                return handler
        return None

    def _record_error(self, error: TradingError) -> None:
        """Record error in history"""
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        log_error(error)

        # Also log to JSON with correlation context
        correlation_context = _get_log_context()
        _get_json_logger().error(
            f"Error recorded: {error.error_code}",
            extra={
                "error_code": error.error_code,
                "error_message": error.message,
                "error_type": (
                    type(getattr(error, "original_error", None)).__name__
                    if getattr(error, "original_error", None)
                    else "Unknown"
                ),
                "recoverable": error.recoverable,
                "context": error.context,
                **correlation_context,
            },
        )

    def handle_error(
        self,
        error: Exception,
        context: dict[str, Any] | None = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    ) -> None:
        """Handle an error with appropriate recovery strategy"""
        trading_error = handle_error(error, context)
        self._record_error(trading_error)

        if recovery_strategy == RecoveryStrategy.DEGRADE:
            logger.warning(f"Degrading functionality due to: {trading_error.message}")
            _get_json_logger().warning(
                f"Degrading functionality due to: {trading_error.message}",
                extra={
                    "recovery_strategy": "degrade",
                    "error_code": trading_error.error_code,
                    "error_message": trading_error.message,
                    **_get_log_context(),
                },
            )
            return None  # Return None to indicate degraded operation

        if not trading_error.recoverable:
            raise trading_error

        return None

    def get_error_stats(self) -> dict[str, Any]:
        """Get error statistics"""
        if not self.error_history:
            return {
                "total_errors": 0,
                "error_types": {},
                "circuit_breaker_state": self.circuit_breaker.state.value,
            }

        error_types: dict[str, int] = {}
        for error in self.error_history:
            error_type = error.error_code
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failures": self.circuit_breaker.failure_count,
            "last_error": self.error_history[-1].to_dict() if self.error_history else None,
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker"""
        self.circuit_breaker.state = CircuitBreakerState.CLOSED
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.success_count = 0
        logger.info("Circuit breaker manually reset")
        _get_json_logger().info(
            "Circuit breaker manually reset",
            extra={
                "circuit_breaker_state": "closed",
                "event": "manual_reset",
                **_get_log_context(),
            },
        )


# Global error handler instance
_error_handler: ErrorHandler | None = None


def get_error_handler() -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """Set custom error handler"""
    global _error_handler
    _error_handler = handler


# Decorator for automatic error handling
def with_error_handling(
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
    fallback: Callable[P, T] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator to add error handling to functions"""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            handler = get_error_handler()

            if fallback:
                handler.fallback_handlers[Exception] = fallback

            return handler.with_retry(func, *args, recovery_strategy=recovery_strategy, **kwargs)

        return wrapper

    return decorator


# Export main components
__all__ = [
    "ErrorHandler",
    "RecoveryStrategy",
    "RetryConfig",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerState",
    "get_error_handler",
    "set_error_handler",
    "with_error_handling",
]
