"""Enhanced exception hierarchy with recovery capabilities.

This module provides an advanced exception system with:
- Automatic recovery handlers
- Error context preservation
- Retry mechanisms
- Circuit breaker patterns
- Error aggregation and reporting
"""

from __future__ import annotations

import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"  # Log and continue
    MEDIUM = "medium"  # Attempt recovery
    HIGH = "high"  # Retry with backoff
    CRITICAL = "critical"  # Immediate shutdown


class RecoveryStrategy(str, Enum):
    """Recovery strategies for errors."""

    IGNORE = "ignore"  # Log and continue
    RETRY = "retry"  # Retry with backoff
    FALLBACK = "fallback"  # Use fallback value/method
    CIRCUIT_BREAK = "circuit_break"  # Stop attempting after failures
    ESCALATE = "escalate"  # Escalate to higher level
    RESTART = "restart"  # Restart component
    COMPENSATE = "compensate"  # Compensating transaction


@dataclass
class ErrorContext:
    """Context information for an error."""

    error_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    component: str = ""
    operation: str = ""
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempted: bool = False
    recovery_successful: bool = False
    additional_info: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = field(default_factory=lambda: traceback.format_exc())

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "additional_info": self.additional_info,
        }


class GPTTraderException(Exception):
    """Enhanced base exception with recovery capabilities."""

    def __init__(
        self,
        message: str,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        **kwargs,
    ) -> None:
        """Initialize enhanced exception.

        Args:
            message: Error message.
            context: Error context information.
            cause: Original exception that caused this error.
            recovery_strategy: Strategy for recovering from this error.
            **kwargs: Additional context information.
        """
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext(**kwargs)
        self.cause = cause
        self.recovery_strategy = recovery_strategy

        # Log the error
        self._log_error()

    def _log_error(self) -> None:
        """Log the error based on severity."""
        log_msg = (
            f"[{self.context.error_id}] {self.__class__.__name__}: {self.message} "
            f"(Severity: {self.context.severity.value}, Component: {self.context.component})"
        )

        if self.context.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg)
        elif self.context.severity == ErrorSeverity.HIGH:
            logger.error(log_msg)
        elif self.context.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def can_retry(self) -> bool:
        """Check if the error can be retried."""
        return (
            self.recovery_strategy == RecoveryStrategy.RETRY
            and self.context.retry_count < self.context.max_retries
        )

    def increment_retry(self) -> None:
        """Increment retry counter."""
        self.context.retry_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "recovery_strategy": self.recovery_strategy.value,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }


class RecoverableError(GPTTraderException):
    """Errors that can be automatically recovered from."""

    def __init__(
        self,
        message: str,
        recovery_action: Callable | None = None,
        fallback_value: Any = None,
        **kwargs,
    ) -> None:
        """Initialize recoverable error.

        Args:
            message: Error message.
            recovery_action: Function to call for recovery.
            fallback_value: Fallback value if recovery fails.
            **kwargs: Additional context.
        """
        super().__init__(
            message,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )
        self.recovery_action = recovery_action
        self.fallback_value = fallback_value

    def recover(self) -> Any:
        """Attempt to recover from the error.

        Returns:
            Recovery result or fallback value.
        """
        try:
            if self.recovery_action:
                logger.info(f"Attempting recovery for {self.context.error_id}")
                result = self.recovery_action()
                self.context.recovery_attempted = True
                self.context.recovery_successful = True
                logger.info(f"Recovery successful for {self.context.error_id}")
                return result
        except Exception as e:
            logger.error(f"Recovery failed for {self.context.error_id}: {e}")
            self.context.recovery_attempted = True
            self.context.recovery_successful = False

        # Return fallback value
        logger.info(f"Using fallback value for {self.context.error_id}")
        return self.fallback_value


class CriticalError(GPTTraderException):
    """Errors requiring immediate system shutdown."""

    def __init__(
        self,
        message: str,
        shutdown_action: Callable | None = None,
        alert_channels: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Initialize critical error.

        Args:
            message: Error message.
            shutdown_action: Function to call for graceful shutdown.
            alert_channels: Channels to send alerts to.
            **kwargs: Additional context.
        """
        super().__init__(
            message,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            severity=ErrorSeverity.CRITICAL,
            **kwargs,
        )
        self.shutdown_action = shutdown_action
        self.alert_channels = alert_channels or []

        # Send alerts
        self._send_alerts()

    def _send_alerts(self) -> None:
        """Send alerts through configured channels."""
        for channel in self.alert_channels:
            try:
                logger.critical(f"ALERT [{channel}]: {self.message}")
                # In production, this would send actual alerts
            except Exception as e:
                logger.error(f"Failed to send alert to {channel}: {e}")

    def shutdown(self) -> None:
        """Perform graceful shutdown."""
        logger.critical(f"Initiating shutdown due to critical error: {self.context.error_id}")

        if self.shutdown_action:
            try:
                self.shutdown_action()
            except Exception as e:
                logger.error(f"Shutdown action failed: {e}")

        logger.critical("System shutdown complete")


class RetryableError(GPTTraderException):
    """Errors that should be retried with backoff."""

    def __init__(
        self,
        message: str,
        backoff_base: float = 2.0,
        backoff_factor: float = 1.0,
        max_backoff: float = 60.0,
        **kwargs,
    ) -> None:
        """Initialize retryable error.

        Args:
            message: Error message.
            backoff_base: Base for exponential backoff.
            backoff_factor: Factor for backoff calculation.
            max_backoff: Maximum backoff time in seconds.
            **kwargs: Additional context.
        """
        super().__init__(
            message, recovery_strategy=RecoveryStrategy.RETRY, severity=ErrorSeverity.HIGH, **kwargs
        )
        self.backoff_base = backoff_base
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff

    def get_backoff_time(self) -> float:
        """Calculate backoff time for retry.

        Returns:
            Backoff time in seconds.
        """
        backoff = self.backoff_factor * (self.backoff_base**self.context.retry_count)
        return min(backoff, self.max_backoff)

    def wait_and_retry(self) -> bool:
        """Wait and check if retry should proceed.

        Returns:
            True if retry should proceed, False otherwise.
        """
        if not self.can_retry():
            return False

        backoff_time = self.get_backoff_time()
        logger.info(
            f"Waiting {backoff_time:.1f}s before retry "
            f"({self.context.retry_count + 1}/{self.context.max_retries}) "
            f"for {self.context.error_id}"
        )
        time.sleep(backoff_time)
        self.increment_retry()
        return True


class DataIntegrityError(RecoverableError):
    """Data integrity issues that can potentially be repaired."""

    def __init__(
        self, message: str, data: Any = None, repair_function: Callable | None = None, **kwargs
    ) -> None:
        """Initialize data integrity error.

        Args:
            message: Error message.
            data: The problematic data.
            repair_function: Function to repair data.
            **kwargs: Additional context.
        """
        super().__init__(
            message,
            recovery_action=lambda: repair_function(data) if repair_function else None,
            component="data_validation",
            **kwargs,
        )
        self.data = data


class NetworkError(RetryableError):
    """Network-related errors that should be retried."""

    def __init__(
        self, message: str, url: str | None = None, status_code: int | None = None, **kwargs
    ) -> None:
        """Initialize network error.

        Args:
            message: Error message.
            url: URL that failed.
            status_code: HTTP status code if applicable.
            **kwargs: Additional context.
        """
        super().__init__(message, component="network", **kwargs)
        self.url = url
        self.status_code = status_code
        self.context.additional_info["url"] = url
        self.context.additional_info["status_code"] = status_code


class ExceptionHandler:
    """Centralized exception handling with recovery."""

    def __init__(self) -> None:
        """Initialize exception handler."""
        self.error_history: list[GPTTraderException] = []
        self.recovery_stats = {
            "total_errors": 0,
            "recovered": 0,
            "failed": 0,
            "critical": 0,
        }
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

    def handle(self, error: Exception) -> Any | None:
        """Handle an exception with appropriate recovery.

        Args:
            error: Exception to handle.

        Returns:
            Recovery result if successful, None otherwise.
        """
        # Track error
        if isinstance(error, GPTTraderException):
            self.error_history.append(error)
            self.recovery_stats["total_errors"] += 1

        # Handle based on error type
        if isinstance(error, CriticalError):
            self.recovery_stats["critical"] += 1
            error.shutdown()
            return None

        elif isinstance(error, RecoverableError):
            result = error.recover()
            if error.context.recovery_successful:
                self.recovery_stats["recovered"] += 1
            else:
                self.recovery_stats["failed"] += 1
            return result

        elif isinstance(error, RetryableError):
            # Check circuit breaker
            component = error.context.component
            if component not in self.circuit_breakers:
                self.circuit_breakers[component] = CircuitBreaker(component)

            breaker = self.circuit_breakers[component]
            if breaker.is_open():
                logger.warning(f"Circuit breaker open for {component}")
                self.recovery_stats["failed"] += 1
                return None

            # Attempt retry
            if error.wait_and_retry():
                return "retry"  # Signal to retry operation
            else:
                breaker.record_failure()
                self.recovery_stats["failed"] += 1
                return None

        else:
            # Standard exception
            logger.error(f"Unhandled exception: {error}")
            self.recovery_stats["failed"] += 1
            return None

    def get_stats(self) -> dict[str, Any]:
        """Get error handling statistics.

        Returns:
            Dictionary of statistics.
        """
        return {
            **self.recovery_stats,
            "recovery_rate": (
                self.recovery_stats["recovered"] / self.recovery_stats["total_errors"]
                if self.recovery_stats["total_errors"] > 0
                else 0
            ),
            "circuit_breakers": {
                name: breaker.get_state() for name, breaker in self.circuit_breakers.items()
            },
            "recent_errors": [
                error.to_dict()
                for error in self.error_history[-10:]  # Last 10 errors
            ],
        }


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        timeout: timedelta = timedelta(minutes=1),
        half_open_requests: int = 1,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            name: Name of the component.
            failure_threshold: Failures before opening circuit.
            timeout: Time before attempting to close circuit.
            half_open_requests: Requests to try in half-open state.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_requests = half_open_requests

        self.failure_count = 0
        self.last_failure_time: datetime | None = None
        self.state = "closed"  # closed, open, half_open
        self.half_open_count = 0

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == "half_open":
            self.half_open_count += 1
            if self.half_open_count >= self.half_open_requests:
                logger.info(f"Circuit breaker {self.name} closing")
                self.state = "closed"
                self.failure_count = 0
                self.half_open_count = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == "half_open":
            logger.warning(f"Circuit breaker {self.name} reopening")
            self.state = "open"
            self.half_open_count = 0
        elif self.state == "closed" and self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker {self.name} opening")
            self.state = "open"

    def is_open(self) -> bool:
        """Check if circuit is open.

        Returns:
            True if circuit is open (blocking requests).
        """
        if self.state == "closed":
            return False

        if self.state == "open" and self.last_failure_time:
            # Check if timeout has passed
            if datetime.now() - self.last_failure_time > self.timeout:
                logger.info(f"Circuit breaker {self.name} entering half-open state")
                self.state = "half_open"
                self.half_open_count = 0
                return False

        return self.state == "open"

    def get_state(self) -> dict[str, Any]:
        """Get circuit breaker state.

        Returns:
            State information.
        """
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


# Specialized exceptions for trading operations


class TradingException(RecoverableError):
    """Base class for trading-related exceptions."""

    def __init__(self, message: str, symbol: str = "", **kwargs) -> None:
        super().__init__(message, component="trading", **kwargs)
        self.symbol = symbol
        self.context.additional_info["symbol"] = symbol


class InsufficientCapitalError(TradingException):
    """Insufficient capital for trade."""

    def __init__(
        self, message: str, required_capital: float, available_capital: float, **kwargs
    ) -> None:
        super().__init__(message, **kwargs)
        self.required_capital = required_capital
        self.available_capital = available_capital
        self.context.additional_info.update(
            {
                "required_capital": required_capital,
                "available_capital": available_capital,
                "shortfall": required_capital - available_capital,
            }
        )


class OrderRejectedError(TradingException):
    """Order rejected by broker."""

    def __init__(self, message: str, order_id: str, reason: str, **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.order_id = order_id
        self.reason = reason
        self.context.additional_info.update(
            {
                "order_id": order_id,
                "rejection_reason": reason,
            }
        )


class RiskLimitError(CriticalError):
    """Risk limit exceeded."""

    def __init__(
        self, message: str, limit_type: str, limit_value: float, current_value: float, **kwargs
    ) -> None:
        super().__init__(message, component="risk_management", **kwargs)
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value
        self.context.additional_info.update(
            {
                "limit_type": limit_type,
                "limit_value": limit_value,
                "current_value": current_value,
                "breach_amount": current_value - limit_value,
            }
        )


# Singleton exception handler
_exception_handler: ExceptionHandler | None = None


def get_exception_handler() -> ExceptionHandler:
    """Get singleton exception handler.

    Returns:
        Global exception handler instance.
    """
    global _exception_handler
    if _exception_handler is None:
        _exception_handler = ExceptionHandler()
    return _exception_handler
