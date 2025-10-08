"""Common utility patterns and helpers used throughout the codebase."""

from __future__ import annotations

import asyncio
import decimal
import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any, TypeVar

from ..errors import NetworkError, ValidationError

T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


async def safe_async_call(
    func: Callable[..., T],
    *args: Any,
    default: T | None = None,
    error_type: type[Exception] | None = None,
    log_level: int = logging.WARNING,
    log_message: str | None = None,
    **kwargs: Any,
) -> T | None:
    """Safely execute an async function with error handling.

    Args:
        func: The async function to execute
        *args: Positional arguments to pass to the function
        default: Default value to return on error (None if not specified)
        error_type: Specific exception type to catch (catches all if None)
        log_level: Logging level for errors (WARNING by default)
        log_message: Custom log message (auto-generated if None)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The function result or default value on error
    """
    try:
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in thread pool
            return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as exc:
        # Filter by error type if specified
        if error_type is not None and not isinstance(exc, error_type):
            raise

        # Log the error
        if log_message is None:
            log_message = f"Error in {func.__name__}: {exc}"
        logger.log(log_level, log_message)

        return default


async def safe_thread_call(
    func: Callable[..., T],
    *args: Any,
    default: T | None = None,
    error_type: type[Exception] | None = None,
    log_level: int = logging.WARNING,
    log_message: str | None = None,
    **kwargs: Any,
) -> T | None:
    """Safely execute a blocking function in a thread pool with error handling.

    Args:
        func: The blocking function to execute
        *args: Positional arguments to pass to the function
        default: Default value to return on error (None if not specified)
        error_type: Specific exception type to catch (catches all if None)
        log_level: Logging level for errors (WARNING by default)
        log_message: Custom log message (auto-generated if None)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The function result or default value on error
    """
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as exc:
        # Filter by error type if specified
        if error_type is not None and not isinstance(exc, error_type):
            raise

        # Log the error
        if log_message is None:
            log_message = f"Error in {func.__name__}: {exc}"
        logger.log(log_level, log_message)

        return default


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (NetworkError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry function calls with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
        exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function that retries on failure
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        async def async_wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            delay = base_delay

            for attempt in range(max_attempts + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return await asyncio.to_thread(func, *args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts + 1,
                            exc,
                        )
                        raise

                    logger.warning(
                        "Function %s failed (attempt %d/%d), retrying in %.1fs: %s",
                        func.__name__,
                        attempt + 1,
                        max_attempts + 1,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(min(delay, max_delay))
                    delay *= backoff_factor
                except Exception:
                    # Don't retry on non-specified exceptions
                    raise

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry decorator")

        def sync_wrapper(*args: Any, **kwargs: Any) -> T:
            import time

            last_exception = None
            delay = base_delay

            for attempt in range(max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        logger.error(
                            "Function %s failed after %d attempts: %s",
                            func.__name__,
                            max_attempts + 1,
                            exc,
                        )
                        raise

                    logger.warning(
                        "Function %s failed (attempt %d/%d), retrying in %.1fs: %s",
                        func.__name__,
                        attempt + 1,
                        max_attempts + 1,
                        delay,
                        exc,
                    )
                    time.sleep(min(delay, max_delay))
                    delay *= backoff_factor
                except Exception:
                    # Don't retry on non-specified exceptions
                    raise

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected error in retry decorator")

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def validate_decimal_positive(
    value: Any,
    field_name: str = "value",
    allow_zero: bool = False,
) -> Decimal:
    """Validate that a value is a positive decimal.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        allow_zero: Whether zero values are allowed

    Returns:
        Validated Decimal value

    Raises:
        ValidationError: If validation fails
    """
    try:
        decimal_value = Decimal(str(value))
    except (TypeError, ValueError, decimal.InvalidOperation) as exc:
        raise ValidationError(
            f"{field_name} must be a valid number", field=field_name, value=value
        ) from exc

    if allow_zero:
        if decimal_value < 0:
            raise ValidationError(
                f"{field_name} must be >= 0", field=field_name, value=decimal_value
            )
    else:
        if decimal_value <= 0:
            raise ValidationError(
                f"{field_name} must be > 0", field=field_name, value=decimal_value
            )

    return decimal_value


def validate_decimal_range(
    value: Any,
    field_name: str = "value",
    min_value: Decimal | None = None,
    max_value: Decimal | None = None,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
) -> Decimal:
    """Validate that a decimal value is within a specified range.

    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value (None for no minimum)
        max_value: Maximum allowed value (None for no maximum)
        inclusive_min: Whether minimum is inclusive
        inclusive_max: Whether maximum is inclusive

    Returns:
        Validated Decimal value

    Raises:
        ValidationError: If validation fails
    """
    try:
        decimal_value = Decimal(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            f"{field_name} must be a valid number", field=field_name, value=value
        ) from exc

    if min_value is not None:
        if inclusive_min:
            if decimal_value < min_value:
                raise ValidationError(
                    f"{field_name} must be >= {min_value}", field=field_name, value=decimal_value
                )
        else:
            if decimal_value <= min_value:
                raise ValidationError(
                    f"{field_name} must be > {min_value}", field=field_name, value=decimal_value
                )

    if max_value is not None:
        if inclusive_max:
            if decimal_value > max_value:
                raise ValidationError(
                    f"{field_name} must be <= {max_value}", field=field_name, value=decimal_value
                )
        else:
            if decimal_value >= max_value:
                raise ValidationError(
                    f"{field_name} must be < {max_value}", field=field_name, value=decimal_value
                )

    return decimal_value


def safe_decimal_division(
    numerator: Decimal | int | float | str,
    denominator: Decimal | int | float | str,
    default: Decimal | None = None,
    field_name: str = "result",
) -> Decimal:
    """Safely perform decimal division with zero-division protection.

    Args:
        numerator: The dividend
        denominator: The divisor
        default: Default value if division fails (None to raise)
        field_name: Name of the field for error messages

    Returns:
        Result of division or default value

    Raises:
        ValidationError: If division fails and no default is provided
    """
    try:
        num_decimal = Decimal(str(numerator))
        den_decimal = Decimal(str(denominator))

        if den_decimal == 0:
            if default is not None:
                return default
            raise ValidationError(f"{field_name}: Division by zero", field=field_name)

        return num_decimal / den_decimal
    except (TypeError, ValueError, decimal.InvalidOperation) as exc:
        if default is not None:
            return default
        raise ValidationError(
            f"{field_name}: Invalid division operation", field=field_name
        ) from exc


class RateLimiter:
    """Simple rate limiter for API calls or other operations."""

    def __init__(self, max_calls: int, time_window: float) -> None:
        """Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: list[float] = []

    async def acquire(self) -> None:
        """Acquire a permit, blocking if rate limit is exceeded."""
        now = asyncio.get_event_loop().time()

        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]

        # Check if we're at the limit
        if len(self.calls) >= self.max_calls:
            # Calculate sleep time until oldest call expires
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                # Re-check after sleeping
                await self.acquire()
                return

        # Record this call
        self.calls.append(now)

    def can_acquire(self) -> bool:
        """Check if a permit can be acquired without blocking."""
        now = asyncio.get_event_loop().time()
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        return len(self.calls) < self.max_calls


def format_decimal(
    value: Decimal | str | int | float,
    decimal_places: int | None = None,
    strip_trailing_zeros: bool = True,
) -> str:
    """Format a decimal value for display.

    Args:
        value: Decimal value to format (accepts multiple types)
        decimal_places: Number of decimal places (auto-detect if None)
        strip_trailing_zeros: Whether to strip trailing zeros

    Returns:
        Formatted string representation
    """
    # Convert to Decimal first
    if not isinstance(value, Decimal):
        try:
            value = Decimal(str(value))
        except (TypeError, ValueError, decimal.InvalidOperation):
            return str(value)

    result = format(value, "f")  # Use format to avoid scientific notation

    if decimal_places is not None:
        # Use Decimal quantize for proper rounding
        if decimal_places == 0:
            quantize_str = "1"
        else:
            quantize_str = "0." + "0" * (decimal_places - 1) + "1"
        value = value.quantize(Decimal(quantize_str))
        result = format(value, "f")

        # Ensure we have exactly the right number of decimal places
        if decimal_places > 0:
            if "." not in result:
                result += "." + "0" * decimal_places
            else:
                integer_part, decimal_part = result.split(".", 1)
                if len(decimal_part) < decimal_places:
                    decimal_part = decimal_part.ljust(decimal_places, "0")
                    result = f"{integer_part}.{decimal_part}"
                elif len(decimal_part) > decimal_places:
                    decimal_part = decimal_part[:decimal_places]
                    result = f"{integer_part}.{decimal_part}"

        # Don't strip zeros when decimal_places is explicitly specified
        # unless strip_trailing_zeros is True and decimal_places is 0
        if strip_trailing_zeros and decimal_places == 0 and "." in result:
            result = result.split(".")[0]
    elif strip_trailing_zeros and "." in result:
        result = result.rstrip("0").rstrip(".")

    return result


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    multiplier: float = 2.0,
    max_delay: float = 60.0,
) -> float:
    """Calculate exponential backoff delay.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        multiplier: Backoff multiplier
        max_delay: Maximum delay in seconds

    Returns:
        Calculated delay in seconds
    """
    delay = base_delay * (multiplier**attempt)
    return min(delay, max_delay)
