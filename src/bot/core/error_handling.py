"""
GPT-Trader Advanced Error Handling and Recovery System

Comprehensive error management providing:
- Centralized error logging and categorization
- Automated error recovery strategies
- Error rate monitoring and alerting
- Circuit breaker patterns for failing services
- Retry mechanisms with exponential backoff
- Error correlation and trend analysis
- Integration with monitoring and alerting systems

This works with the existing exception hierarchy to provide enterprise-grade
error handling across all GPT-Trader components.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from .config import get_config
from .exceptions import (
    ComponentException,
    ConfigurationException,
    DatabaseException,
    ErrorCategory,
    ErrorSeverity,
    GPTTraderException,
    NetworkException,
    RiskException,
    ValidationException,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
F = TypeVar("F", bound=Callable)


class CircuitBreakerState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategy types"""

    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    NO_RETRY = "no_retry"


class ErrorTrendDirection(Enum):
    """Error trend analysis"""

    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


@runtime_checkable
class IErrorRecoveryStrategy(Protocol):
    """Interface for error recovery strategies"""

    def can_recover(self, error: GPTTraderException) -> bool:
        """Check if this strategy can recover from the error"""
        ...

    def recover(self, error: GPTTraderException, context: dict[str, Any]) -> bool:
        """Attempt to recover from the error"""
        ...

    def get_recovery_time_estimate(self) -> timedelta | None:
        """Estimate time needed for recovery"""
        ...


@dataclass
class ErrorStatistics:
    """Error statistics tracking"""

    total_errors: int = 0
    errors_by_severity: dict[ErrorSeverity, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_category: dict[ErrorCategory, int] = field(default_factory=lambda: defaultdict(int))
    errors_by_component: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    first_error_time: datetime | None = None
    last_error_time: datetime | None = None
    error_rate_per_hour: float = 0.0
    recovery_success_rate: float = 0.0

    def add_error(self, error: GPTTraderException) -> None:
        """Add error to statistics"""
        self.total_errors += 1
        self.errors_by_severity[error.severity] += 1
        self.errors_by_category[error.category] += 1

        if error.component:
            self.errors_by_component[error.component] += 1

        now = datetime.now()
        if not self.first_error_time:
            self.first_error_time = now
        self.last_error_time = now

        # Update error rate (errors per hour)
        if self.first_error_time:
            time_span = (now - self.first_error_time).total_seconds() / 3600
            if time_span > 0:
                self.error_rate_per_hour = self.total_errors / time_span


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: timedelta = timedelta(seconds=60)  # Time before testing recovery
    success_threshold: int = 3  # Successes needed to close
    monitor_window: timedelta = timedelta(minutes=5)  # Window for failure counting


@dataclass
class RetryConfig:
    """Retry configuration"""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True  # Add randomization to delays


class CircuitBreaker:
    """Circuit breaker for failing operations"""

    def __init__(self, name: str, config: CircuitBreakerConfig) -> None:
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED

        # Failure tracking
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: datetime | None = None
        self.state_change_time = datetime.now()

        # Recent failures for monitoring window
        self.recent_failures: deque = deque(maxlen=100)

        self._lock = threading.RLock()

        logger.info(f"Circuit breaker '{name}' initialized")

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
                else:
                    raise ComponentException(
                        f"Circuit breaker '{self.name}' is open",
                        component=self.name,
                        context={
                            "state": self.state.value,
                            "failure_count": self.failure_count,
                            "last_failure": (
                                self.last_failure_time.isoformat()
                                if self.last_failure_time
                                else None
                            ),
                        },
                    )

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result

        except Exception as e:
            self._record_failure(e)
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        return datetime.now() - self.last_failure_time > self.config.recovery_timeout

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state"""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.state_change_time = datetime.now()
        logger.info(f"Circuit breaker '{self.name}' transitioned to HALF_OPEN")

    def _record_success(self) -> None:
        """Record successful operation"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record failed operation"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            self.recent_failures.append(
                {
                    "timestamp": self.last_failure_time,
                    "error": str(error),
                    "error_type": type(error).__name__,
                }
            )

            if self.state == CircuitBreakerState.HALF_OPEN:
                self._transition_to_open()
            elif self.state == CircuitBreakerState.CLOSED:
                # Check if we should open the circuit
                failures_in_window = self._count_failures_in_window()
                if failures_in_window >= self.config.failure_threshold:
                    self._transition_to_open()

    def _count_failures_in_window(self) -> int:
        """Count failures within the monitoring window"""
        cutoff_time = datetime.now() - self.config.monitor_window
        return sum(1 for failure in self.recent_failures if failure["timestamp"] >= cutoff_time)

    def _transition_to_open(self) -> None:
        """Transition to open state"""
        self.state = CircuitBreakerState.OPEN
        self.state_change_time = datetime.now()
        logger.warning(f"Circuit breaker '{self.name}' opened due to failures")

    def _transition_to_closed(self) -> None:
        """Transition to closed state"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_change_time = datetime.now()
        logger.info(f"Circuit breaker '{self.name}' closed - service recovered")

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status"""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "last_failure_time": (
                    self.last_failure_time.isoformat() if self.last_failure_time else None
                ),
                "state_change_time": self.state_change_time.isoformat(),
                "recent_failures_count": len(self.recent_failures),
                "failures_in_window": self._count_failures_in_window(),
            }


class RetryHandler:
    """Intelligent retry handler with multiple strategies"""

    def __init__(self, config: RetryConfig) -> None:
        self.config = config

    def retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic"""
        last_exception = None

        for attempt in range(self.config.max_attempts):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.debug(
                        f"Retrying after {delay:.2f}s (attempt {attempt + 1}/{self.config.max_attempts})"
                    )
                    time.sleep(delay)

                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Don't retry certain error types
                if isinstance(e, ConfigurationException | ValidationException | RiskException):
                    logger.debug(f"Not retrying {type(e).__name__}: {str(e)}")
                    break

                logger.debug(f"Attempt {attempt + 1} failed: {str(e)}")

        # All retries failed
        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy"""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay

        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(
                self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1)),
                self.config.max_delay,
            )

        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(self.config.base_delay * attempt, self.config.max_delay)
        else:
            delay = 0

        # Add jitter to prevent thundering herd
        if self.config.jitter and delay > 0:
            import random

            jitter_amount = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0, delay)


class DatabaseRecoveryStrategy:
    """Recovery strategy for database errors"""

    def can_recover(self, error: GPTTraderException) -> bool:
        return isinstance(error, DatabaseException)

    def recover(self, error: DatabaseException, context: dict[str, Any]) -> bool:
        """Attempt database recovery"""
        try:
            # Try to reconnect to database
            from .database import get_database

            db_manager = get_database()

            # Test connection
            with db_manager.get_connection() as conn:
                conn.execute("SELECT 1")

            logger.info("Database connection recovered")
            return True

        except Exception as e:
            logger.error(f"Database recovery failed: {str(e)}")
            return False

    def get_recovery_time_estimate(self) -> timedelta | None:
        return timedelta(seconds=5)


class NetworkRecoveryStrategy:
    """Recovery strategy for network errors"""

    def can_recover(self, error: GPTTraderException) -> bool:
        return isinstance(error, NetworkException)

    def recover(self, error: NetworkException, context: dict[str, Any]) -> bool:
        """Attempt network recovery"""
        # For network errors, recovery typically involves waiting
        # and retrying the connection
        time.sleep(1)  # Brief pause for transient network issues
        return True

    def get_recovery_time_estimate(self) -> timedelta | None:
        return timedelta(seconds=2)


class ErrorTrendAnalyzer:
    """Analyze error trends for predictive insights"""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.error_history: deque = deque(maxlen=window_size)
        self._lock = threading.RLock()

    def add_error(self, error: GPTTraderException) -> None:
        """Add error to trend analysis"""
        with self._lock:
            self.error_history.append(
                {
                    "timestamp": error.timestamp,
                    "severity": error.severity,
                    "category": error.category,
                    "component": error.component,
                }
            )

    def analyze_trend(self, period: timedelta = timedelta(hours=1)) -> dict[str, Any]:
        """Analyze error trends"""
        with self._lock:
            if len(self.error_history) < 10:  # Not enough data
                return {
                    "trend_direction": ErrorTrendDirection.STABLE,
                    "confidence": 0.0,
                    "error_rate_change": 0.0,
                    "dominant_category": None,
                }

            cutoff_time = datetime.now() - period
            recent_errors = [e for e in self.error_history if e["timestamp"] >= cutoff_time]

            if len(recent_errors) < 2:
                return {
                    "trend_direction": ErrorTrendDirection.STABLE,
                    "confidence": 0.5,
                    "error_rate_change": 0.0,
                    "dominant_category": None,
                }

            # Calculate error rate trend
            half_point = len(recent_errors) // 2
            first_half = recent_errors[:half_point]
            second_half = recent_errors[half_point:]

            first_half_rate = len(first_half) / (period.total_seconds() / 2)
            second_half_rate = len(second_half) / (period.total_seconds() / 2)

            rate_change = (second_half_rate - first_half_rate) / max(first_half_rate, 0.1)

            # Determine trend direction
            if abs(rate_change) < 0.1:
                trend = ErrorTrendDirection.STABLE
            elif rate_change > 0.2:
                trend = ErrorTrendDirection.INCREASING
            elif rate_change < -0.2:
                trend = ErrorTrendDirection.DECREASING
            else:
                trend = ErrorTrendDirection.VOLATILE

            # Find dominant error category
            category_counts = defaultdict(int)
            for error in recent_errors:
                category_counts[error["category"]] += 1

            dominant_category = (
                max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
            )

            return {
                "trend_direction": trend,
                "confidence": min(len(recent_errors) / 50, 1.0),
                "error_rate_change": rate_change,
                "dominant_category": dominant_category,
                "recent_error_count": len(recent_errors),
                "categories": dict(category_counts),
            }


class ErrorManager:
    """
    Centralized error management system for GPT-Trader

    Provides comprehensive error handling including:
    - Error logging and categorization
    - Automated recovery attempts
    - Circuit breaker protection
    - Retry mechanisms
    - Trend analysis and alerting
    """

    def __init__(self) -> None:
        self.config = get_config()

        # Error tracking
        self.statistics = ErrorStatistics()
        self.error_history: list[GPTTraderException] = []

        # Recovery strategies
        self.recovery_strategies: list[IErrorRecoveryStrategy] = [
            DatabaseRecoveryStrategy(),
            NetworkRecoveryStrategy(),
        ]

        # Circuit breakers
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Trend analysis
        self.trend_analyzer = ErrorTrendAnalyzer()

        # Thread safety
        self._lock = threading.RLock()

        # Error callbacks
        self.error_callbacks: list[Callable[[GPTTraderException], None]] = []

        logger.info("Error manager initialized")

    def handle_error(
        self,
        error: GPTTraderException,
        attempt_recovery: bool = True,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Handle error with comprehensive processing"""

        with self._lock:
            # Record error statistics
            self.statistics.add_error(error)
            self.error_history.append(error)
            self.trend_analyzer.add_error(error)

            # Limit history size
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]

        # Log the error
        self._log_error(error)

        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                callback(error)
            except Exception as e:
                logger.error(f"Error callback failed: {str(e)}")

        # Attempt recovery if requested and error is recoverable
        if attempt_recovery and error.recoverable:
            return self._attempt_recovery(error, context or {})

        return False

    def _log_error(self, error: GPTTraderException) -> None:
        """Log error with appropriate level"""
        severity_levels = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL,
        }

        level = severity_levels.get(error.severity, logging.ERROR)
        logger.log(level, f"ERROR [{error.error_id}]: {error}")

        # Log full traceback for critical errors
        if error.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.FATAL):
            logger.critical(f"Error context: {error.to_dict()}")

    def _attempt_recovery(self, error: GPTTraderException, context: dict[str, Any]) -> bool:
        """Attempt to recover from error using available strategies"""

        for strategy in self.recovery_strategies:
            if strategy.can_recover(error):
                logger.info(f"Attempting recovery with {strategy.__class__.__name__}")

                try:
                    if strategy.recover(error, context):
                        logger.info(f"Recovery successful for error {error.error_id}")
                        self.statistics.recovery_success_rate = (
                            self.statistics.recovery_success_rate
                            * (self.statistics.total_errors - 1)
                            + 100
                        ) / self.statistics.total_errors
                        return True
                except Exception as e:
                    logger.error(f"Recovery attempt failed: {str(e)}")

        logger.warning(f"No recovery possible for error {error.error_id}")
        return False

    def get_circuit_breaker(
        self, name: str, config: CircuitBreakerConfig | None = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            if not config:
                config = CircuitBreakerConfig()  # Use defaults
            self.circuit_breakers[name] = CircuitBreaker(name, config)

        return self.circuit_breakers[name]

    def get_error_statistics(self) -> dict[str, Any]:
        """Get comprehensive error statistics"""
        with self._lock:
            trend_analysis = self.trend_analyzer.analyze_trend()

            circuit_breaker_stats = {}
            for name, cb in self.circuit_breakers.items():
                circuit_breaker_stats[name] = cb.get_status()

            return {
                "total_errors": self.statistics.total_errors,
                "error_rate_per_hour": self.statistics.error_rate_per_hour,
                "recovery_success_rate": self.statistics.recovery_success_rate,
                "errors_by_severity": {
                    k.value: v for k, v in self.statistics.errors_by_severity.items()
                },
                "errors_by_category": {
                    k.value: v for k, v in self.statistics.errors_by_category.items()
                },
                "errors_by_component": dict(self.statistics.errors_by_component),
                "trend_analysis": {
                    "direction": trend_analysis["trend_direction"].value,
                    "confidence": trend_analysis["confidence"],
                    "rate_change": trend_analysis["error_rate_change"],
                    "dominant_category": (
                        trend_analysis["dominant_category"].value
                        if trend_analysis["dominant_category"]
                        else None
                    ),
                },
                "circuit_breakers": circuit_breaker_stats,
                "first_error_time": (
                    self.statistics.first_error_time.isoformat()
                    if self.statistics.first_error_time
                    else None
                ),
                "last_error_time": (
                    self.statistics.last_error_time.isoformat()
                    if self.statistics.last_error_time
                    else None
                ),
            }

    def add_error_callback(self, callback: Callable[[GPTTraderException], None]) -> None:
        """Add callback for error notifications"""
        self.error_callbacks.append(callback)

    def clear_statistics(self) -> None:
        """Clear error statistics (for testing/reset)"""
        with self._lock:
            self.statistics = ErrorStatistics()
            self.error_history.clear()
            self.trend_analyzer = ErrorTrendAnalyzer()


# Global error manager instance
_error_manager: ErrorManager | None = None
_manager_lock = threading.Lock()


def get_error_manager() -> ErrorManager:
    """Get global error manager instance"""
    global _error_manager

    with _manager_lock:
        if _error_manager is None:
            _error_manager = ErrorManager()
            logger.info("Global error manager created")

        return _error_manager


# Decorator for automatic error handling
def handle_errors(
    retry_config: RetryConfig | None = None,
    circuit_breaker_name: str | None = None,
    recovery_enabled: bool = True,
):
    """Decorator for automatic error handling and recovery"""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_manager = get_error_manager()

            # Set up retry handler if configured
            if retry_config:
                retry_handler = RetryHandler(retry_config)

                def func_to_call():
                    return func(*args, **kwargs)

            else:

                def func_to_call():
                    return func(*args, **kwargs)

                retry_handler = None

            # Set up circuit breaker if configured
            if circuit_breaker_name:
                circuit_breaker = error_manager.get_circuit_breaker(circuit_breaker_name)
                if retry_handler:

                    def func_to_call():
                        return circuit_breaker.call(retry_handler.retry, func, *args, **kwargs)

                else:

                    def func_to_call():
                        return circuit_breaker.call(func, *args, **kwargs)

            elif retry_handler:

                def func_to_call():
                    return retry_handler.retry(func, *args, **kwargs)

            try:
                return func_to_call()

            except GPTTraderException as e:
                # Handle known exceptions
                error_manager.handle_error(e, attempt_recovery=recovery_enabled)
                raise

            except Exception as e:
                # Convert unknown exceptions to GPTTraderException
                gpt_error = ComponentException(
                    f"Unexpected error in {func.__name__}: {str(e)}",
                    component=func.__module__,
                    context={"function": func.__name__, "original_error": str(e)},
                )
                error_manager.handle_error(gpt_error, attempt_recovery=recovery_enabled)
                raise gpt_error from e

        return wrapper

    return decorator


@contextmanager
def error_handling_context(component_name: str, operation: str):
    """Context manager for error handling"""
    error_manager = get_error_manager()

    try:
        yield
    except GPTTraderException as e:
        # Add context information
        e.component = component_name
        e.context["operation"] = operation
        error_manager.handle_error(e)
        raise
    except Exception as e:
        # Convert to GPTTraderException
        gpt_error = ComponentException(
            f"Error in {component_name} during {operation}: {str(e)}",
            component=component_name,
            context={"operation": operation, "original_error": str(e)},
        )
        error_manager.handle_error(gpt_error)
        raise gpt_error from e


# Convenience functions


def report_error(
    error: Exception | GPTTraderException,
    component: str | None = None,
    attempt_recovery: bool = True,
) -> bool:
    """Report error to error management system"""

    if isinstance(error, GPTTraderException):
        gpt_error = error
        if component and not gpt_error.component:
            gpt_error.component = component
    else:
        gpt_error = ComponentException(
            str(error), component=component, context={"original_error_type": type(error).__name__}
        )

    return get_error_manager().handle_error(gpt_error, attempt_recovery)


def get_error_statistics() -> dict[str, Any]:
    """Get system-wide error statistics"""
    return get_error_manager().get_error_statistics()


def create_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Create circuit breaker for service protection"""
    return get_error_manager().get_circuit_breaker(name, config)


def with_retry(config: RetryConfig):
    """Decorator for retry functionality"""
    return handle_errors(retry_config=config)


def with_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None):
    """Decorator for circuit breaker protection"""
    return handle_errors(circuit_breaker_name=name)
