"""Standardized error handling patterns for consistent error management."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from bot_v2.errors import (
    ConfigurationError,
    DataError,
    ExecutionError,
    NetworkError,
    RiskLimitExceeded,
    TradingError,
    ValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ErrorContext:
    """Context manager for standardized error handling with logging."""

    def __init__(
        self,
        operation: str,
        reraise: type[Exception] | tuple[type[Exception], ...] | None = None,
        default_return: Any = None,
        log_level: int = logging.ERROR,
        extra_context: dict[str, Any] | None = None,
    ) -> None:
        """Initialize error context.

        Args:
            operation: Description of the operation being performed
            reraise: Exception types to re-raise (None to swallow all)
            default_return: Value to return on error if not re-raising
            log_level: Logging level for errors
            extra_context: Additional context to include in logs
        """
        self.operation = operation
        self.reraise = reraise
        self.default_return = default_return
        self.log_level = log_level
        self.extra_context = extra_context or {}

    def __enter__(self) -> ErrorContext:
        return self

    def __exit__(
        self, exc_type: type[Exception] | None, exc_val: Exception | None, exc_tb: Any
    ) -> bool:
        if exc_val is None:
            return False

        # Log the error with context
        self._log_error(exc_val, exc_tb)

        # Determine whether to re-raise
        if self.reraise is None:
            return True  # Swallow the exception

        if isinstance(self.reraise, tuple):
            should_reraise = isinstance(exc_val, self.reraise)
        else:
            should_reraise = isinstance(exc_val, self.reraise)

        if should_reraise:
            return False  # Re-raise the exception

        return True  # Swallow the exception

    def _log_error(self, exc: Exception, exc_tb: Any) -> None:
        """Log the error with full context."""
        context_parts = [
            f"operation={self.operation}",
            f"error_type={type(exc).__name__}",
            f"error_message={str(exc)}",
        ]

        # Add extra context
        for key, value in self.extra_context.items():
            context_parts.append(f"{key}={value}")

        context_str = ", ".join(context_parts)

        # Log with appropriate level and traceback
        if self.log_level >= logging.ERROR:
            logger.error(f"Error in {context_str}\n{traceback.format_tb(exc_tb)}")
        else:
            logger.log(self.log_level, f"Error in {context_str}")


def handle_errors(
    operation: str,
    reraise: type[Exception] | tuple[type[Exception], ...] | None = None,
    default_return: Any = None,
    log_level: int = logging.ERROR,
    extra_context: dict[str, Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for standardized error handling.

    Args:
        operation: Description of the operation being performed
        reraise: Exception types to re-raise (None to swallow all)
        default_return: Value to return on error if not re-raising
        log_level: Logging level for errors
        extra_context: Additional context to include in logs

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with ErrorContext(
                operation=operation,
                reraise=reraise,
                default_return=default_return,
                log_level=log_level,
                extra_context=extra_context,
            ):
                return func(*args, **kwargs)

            # This should never be reached, but mypy needs it
            return default_return  # type: ignore[return-value]

        return wrapper

    return decorator


def safe_execute(
    func: Callable[..., T],
    *args: Any,
    operation: str = "unknown",
    default_return: T | None = None,
    reraise: type[Exception] | tuple[type[Exception], ...] | None = None,
    **kwargs: Any,
) -> T | None:
    """Safely execute a function with standardized error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        operation: Description of the operation
        default_return: Default return value on error
        reraise: Exception types to re-raise
        **kwargs: Function keyword arguments

    Returns:
        Function result or default_return on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        if reraise is not None and isinstance(exc, reraise):
            raise

        logger.error(f"Error in {operation}: {exc}")
        return default_return


# Specific error handling patterns for common operations


def handle_brokerage_errors(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for brokerage operations with standardized error handling.

    Args:
        operation: Description of the brokerage operation

    Returns:
        Decorated function
    """
    return handle_errors(
        operation=operation,
        reraise=(TradingError, NetworkError, ExecutionError),
        default_return=None,
    )


def handle_order_errors(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for order operations with standardized error handling.

    Args:
        operation: Description of the order operation

    Returns:
        Decorated function
    """
    return handle_errors(
        operation=operation,
        reraise=(ExecutionError, TradingError),
        default_return=None,
    )


def handle_data_errors(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for data operations with standardized error handling.

    Args:
        operation: Description of the data operation

    Returns:
        Decorated function
    """
    return handle_errors(
        operation=operation,
        reraise=(DataError, ValidationError),
        default_return=None,
    )


def handle_config_errors(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for configuration operations with standardized error handling.

    Args:
        operation: Description of the configuration operation

    Returns:
        Decorated function
    """
    return handle_errors(
        operation=operation,
        reraise=ConfigurationError,
        default_return=None,
    )


def handle_account_errors(operation: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for account operations with standardized error handling.

    Args:
        operation: Description of the account operation

    Returns:
        Decorated function
    """
    return handle_errors(
        operation=operation,
        reraise=(TradingError, RiskLimitExceeded),
        default_return=None,
    )


# Async error handling patterns


def handle_async_errors(
    operation: str,
    reraise: type[Exception] | tuple[type[Exception], ...] | None = None,
    default_return: Any = None,
    log_level: int = logging.ERROR,
    extra_context: dict[str, Any] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for async functions with standardized error handling.

    Args:
        operation: Description of the operation being performed
        reraise: Exception types to re-raise (None to swallow all)
        default_return: Value to return on error if not re-raising
        log_level: Logging level for errors
        extra_context: Additional context to include in logs

    Returns:
        Decorated async function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)  # type: ignore[misc]
            except Exception as exc:
                # Log the error with context
                context_parts = [
                    f"operation={operation}",
                    f"error_type={type(exc).__name__}",
                    f"error_message={str(exc)}",
                ]

                if extra_context:
                    for key, value in extra_context.items():
                        context_parts.append(f"{key}={value}")

                context_str = ", ".join(context_parts)

                if log_level >= logging.ERROR:
                    logger.error(f"Async error in {context_str}\n{traceback.format_exc()}")
                else:
                    logger.log(log_level, f"Async error in {context_str}")

                # Determine whether to re-raise
                if reraise is None:
                    return default_return  # type: ignore[return-value]

                if isinstance(reraise, tuple):
                    should_reraise = isinstance(exc, reraise)
                else:
                    should_reraise = isinstance(exc, reraise)

                if should_reraise:
                    raise

                return default_return  # type: ignore[return-value]

        return wrapper

    return decorator


# Utility functions for common error scenarios


def validate_and_execute(
    validator: Callable[..., bool],
    error_message: str,
    operation: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that validates before executing function.

    Args:
        validator: Validation function that returns True if valid
        error_message: Error message to raise if validation fails
        operation: Description of the operation

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not validator(*args, **kwargs):
                raise ValidationError(error_message, field=operation)

            with ErrorContext(operation=operation):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: type[Exception] | tuple[type[Exception], ...] = Exception,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries function on specified exceptions.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each attempt
        retry_on: Exception types that trigger retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            import time

            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exception = exc

                    if not isinstance(exc, retry_on):
                        raise

                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {exc}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff_factor

            # This should never be reached
            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator
