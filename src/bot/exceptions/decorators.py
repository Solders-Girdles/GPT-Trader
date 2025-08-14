"""Decorators for automatic exception handling and recovery.

This module provides decorators that can be applied to functions
to automatically handle exceptions with retry logic, circuit breaking,
and recovery mechanisms.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import Any

from .enhanced_exceptions import (
    CriticalError,
    ErrorContext,
    RecoverableError,
    RetryableError,
    get_exception_handler,
)

logger = logging.getLogger(__name__)


def with_retry(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    backoff_factor: float = 1.0,
    max_backoff: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    component: str = "",
    operation: str = "",
) -> Callable:
    """Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base for exponential backoff.
        backoff_factor: Factor for backoff calculation.
        max_backoff: Maximum backoff time in seconds.
        exceptions: Tuple of exceptions to catch and retry.
        component: Component name for error context.
        operation: Operation name for error context.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            retry_count = 0
            last_exception = None

            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if retry_count >= max_retries:
                        logger.error(
                            f"Max retries ({max_retries}) exceeded for "
                            f"{func.__name__} in {component}/{operation}"
                        )
                        raise

                    # Calculate backoff
                    backoff = backoff_factor * (backoff_base**retry_count)
                    backoff = min(backoff, max_backoff)

                    logger.warning(
                        f"Retry {retry_count + 1}/{max_retries} for {func.__name__} "
                        f"after {backoff:.1f}s (error: {str(e)})"
                    )

                    time.sleep(backoff)
                    retry_count += 1

            # Should not reach here, but handle it
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def with_recovery(
    fallback_value: Any = None,
    fallback_function: Callable | None = None,
    component: str = "",
    operation: str = "",
    log_errors: bool = True,
) -> Callable:
    """Decorator for automatic error recovery.

    Args:
        fallback_value: Value to return on error.
        fallback_function: Function to call for fallback value.
        component: Component name for error context.
        operation: Operation name for error context.
        log_errors: Whether to log errors.

    Returns:
        Decorated function with recovery logic.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__} ({component}/{operation}): {str(e)}, "
                        f"using fallback"
                    )

                # Try fallback function first
                if fallback_function:
                    try:
                        return fallback_function(*args, **kwargs)
                    except Exception as fallback_error:
                        if log_errors:
                            logger.error(
                                f"Fallback function failed: {str(fallback_error)}, "
                                f"using fallback value"
                            )

                # Return fallback value
                return fallback_value

        return wrapper

    return decorator


def with_circuit_breaker(
    failure_threshold: int = 5,
    timeout_seconds: float = 60.0,
    component: str = "",
    operation: str = "",
) -> Callable:
    """Decorator for circuit breaker pattern.

    Args:
        failure_threshold: Failures before opening circuit.
        timeout_seconds: Seconds before attempting to close circuit.
        component: Component name for circuit breaker.
        operation: Operation name for error context.

    Returns:
        Decorated function with circuit breaker.
    """

    def decorator(func: Callable) -> Callable:
        # Get the exception handler and circuit breaker
        handler = get_exception_handler()
        breaker_name = f"{component or func.__module__}.{func.__name__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if circuit is open
            if breaker_name in handler.circuit_breakers:
                breaker = handler.circuit_breakers[breaker_name]
                if breaker.is_open():
                    raise RuntimeError(
                        f"Circuit breaker open for {breaker_name}. " f"Too many failures detected."
                    )

            try:
                result = func(*args, **kwargs)

                # Record success
                if breaker_name in handler.circuit_breakers:
                    handler.circuit_breakers[breaker_name].record_success()

                return result

            except Exception:
                # Record failure
                if breaker_name in handler.circuit_breakers:
                    handler.circuit_breakers[breaker_name].record_failure()
                else:
                    # Create circuit breaker on first failure
                    from datetime import timedelta

                    from .enhanced_exceptions import CircuitBreaker

                    handler.circuit_breakers[breaker_name] = CircuitBreaker(
                        name=breaker_name,
                        failure_threshold=failure_threshold,
                        timeout=timedelta(seconds=timeout_seconds),
                    )
                    handler.circuit_breakers[breaker_name].record_failure()

                raise

        return wrapper

    return decorator


def handle_exceptions(
    recoverable_exceptions: tuple[type[Exception], ...] = (),
    retryable_exceptions: tuple[type[Exception], ...] = (),
    critical_exceptions: tuple[type[Exception], ...] = (),
    fallback_value: Any = None,
    component: str = "",
    operation: str = "",
    max_retries: int = 3,
) -> Callable:
    """Comprehensive exception handling decorator.

    Args:
        recoverable_exceptions: Exceptions to recover from.
        retryable_exceptions: Exceptions to retry.
        critical_exceptions: Exceptions that are critical.
        fallback_value: Fallback value for recoverable errors.
        component: Component name for error context.
        operation: Operation name for error context.
        max_retries: Maximum retries for retryable errors.

    Returns:
        Decorated function with exception handling.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            handler = get_exception_handler()
            context = ErrorContext(
                component=component or func.__module__,
                operation=operation or func.__name__,
                max_retries=max_retries,
            )

            retry_count = 0
            while retry_count <= max_retries:
                try:
                    return func(*args, **kwargs)

                except critical_exceptions as e:
                    # Handle critical errors
                    critical_error = CriticalError(
                        message=str(e),
                        context=context,
                        cause=e,
                    )
                    handler.handle(critical_error)
                    raise

                except retryable_exceptions as e:
                    # Handle retryable errors
                    if retry_count >= max_retries:
                        logger.error(f"Max retries exceeded for {func.__name__}")
                        raise

                    retryable_error = RetryableError(
                        message=str(e),
                        context=context,
                        cause=e,
                    )

                    result = handler.handle(retryable_error)
                    if result == "retry":
                        retry_count += 1
                        continue
                    else:
                        raise

                except recoverable_exceptions as e:
                    # Handle recoverable errors
                    recoverable_error = RecoverableError(
                        message=str(e),
                        context=context,
                        cause=e,
                        fallback_value=fallback_value,
                    )

                    result = handler.handle(recoverable_error)
                    return result if result is not None else fallback_value

                except Exception as e:
                    # Unexpected error
                    logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                    raise

            # Max retries exceeded
            raise RuntimeError(f"Max retries ({max_retries}) exceeded for {func.__name__}")

        return wrapper

    return decorator


def safe_execution(
    default_return: Any = None,
    log_errors: bool = True,
    raise_critical: bool = True,
) -> Callable:
    """Simple decorator for safe function execution.

    Args:
        default_return: Default value to return on error.
        log_errors: Whether to log errors.
        raise_critical: Whether to raise critical errors.

    Returns:
        Decorated function that won't crash on errors.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {str(e)}")

                # Check if it's a critical error
                if raise_critical and isinstance(e, CriticalError):
                    raise

                return default_return

        return wrapper

    return decorator


def monitor_performance(
    slow_threshold_seconds: float = 1.0,
    component: str = "",
    operation: str = "",
) -> Callable:
    """Decorator to monitor function performance.

    Args:
        slow_threshold_seconds: Threshold for slow execution warning.
        component: Component name for logging.
        operation: Operation name for logging.

    Returns:
        Decorated function with performance monitoring.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time

                if execution_time > slow_threshold_seconds:
                    logger.warning(
                        f"Slow execution in {func.__name__} "
                        f"({component}/{operation}): {execution_time:.2f}s"
                    )
                else:
                    logger.debug(f"Execution time for {func.__name__}: {execution_time:.3f}s")

        return wrapper

    return decorator


def validate_inputs(**validators: Callable[[Any], bool]) -> Callable:
    """Decorator for input validation.

    Args:
        **validators: Keyword arguments mapping parameter names to validation functions.

    Returns:
        Decorated function with input validation.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate inputs
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    if not validator(value):
                        raise ValueError(f"Invalid input for parameter '{param_name}': {value}")

            return func(*args, **kwargs)

        return wrapper

    return decorator
