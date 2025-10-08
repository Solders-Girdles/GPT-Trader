"""Tests for common utility patterns and helpers."""

from __future__ import annotations

import asyncio
import decimal
import logging
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot_v2.errors import NetworkError, ValidationError
from bot_v2.utilities.common_patterns import (
    RateLimiter,
    format_decimal,
    retry_with_backoff,
    safe_async_call,
    safe_decimal_division,
    safe_thread_call,
    validate_decimal_positive,
    validate_decimal_range,
)


class TestSafeAsyncCall:
    """Test cases for safe_async_call function."""

    async def test_successful_async_call(self):
        """Test successful async function call."""
        async def test_func(x: int) -> int:
            return x * 2

        result = await safe_async_call(test_func, 5)
        assert result == 10

    async def test_successful_sync_call(self):
        """Test successful sync function call in thread pool."""
        def test_func(x: int) -> int:
            return x * 2

        result = await safe_async_call(test_func, 5)
        assert result == 10

    async def test_exception_with_default(self):
        """Test exception handling with default value."""
        async def failing_func():
            raise ValueError("Test error")

        result = await safe_async_call(failing_func, default="fallback")
        assert result == "fallback"

    async def test_exception_without_default(self):
        """Test exception handling without default value."""
        async def failing_func():
            raise ValueError("Test error")

        result = await safe_async_call(failing_func)
        assert result is None

    async def test_specific_exception_type(self):
        """Test filtering by specific exception type."""
        async def failing_func():
            raise ValueError("Test error")

        # Should catch ValueError
        result = await safe_async_call(failing_func, error_type=ValueError, default="caught")
        assert result == "caught"

        # Should not catch TypeError
        async def type_error_func():
            raise TypeError("Type error")

        with pytest.raises(TypeError):
            await safe_async_call(type_error_func, error_type=ValueError)

    async def test_custom_log_message(self):
        """Test custom log message."""
        async def failing_func():
            raise ValueError("Test error")

        with patch("bot_v2.utilities.common_patterns.logger") as mock_logger:
            await safe_async_call(
                failing_func,
                log_message="Custom error message",
                log_level=logging.ERROR
            )
            mock_logger.log.assert_called_once_with(
                logging.ERROR,
                "Custom error message"
            )


class TestSafeThreadCall:
    """Test cases for safe_thread_call function."""

    async def test_successful_call(self):
        """Test successful function call."""
        def test_func(x: int) -> int:
            return x * 2

        result = await safe_thread_call(test_func, 5)
        assert result == 10

    async def test_exception_handling(self):
        """Test exception handling."""
        def failing_func():
            raise ValueError("Test error")

        result = await safe_thread_call(failing_func, default="fallback")
        assert result == "fallback"


class TestRetryWithBackoff:
    """Test cases for retry_with_backoff decorator."""

    async def test_successful_function(self):
        """Test function that succeeds on first try."""
        @retry_with_backoff(max_attempts=3)
        async def test_func():
            return "success"

        result = await test_func()
        assert result == "success"

    async def test_eventual_success(self):
        """Test function that fails initially then succeeds."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = await test_func()
        assert result == "success"
        assert call_count == 3

    async def test_max_attempts_exceeded(self):
        """Test function that never succeeds."""
        @retry_with_backoff(max_attempts=2, base_delay=0.01)
        async def test_func():
            raise NetworkError("Persistent failure")

        with pytest.raises(NetworkError, match="Persistent failure"):
            await test_func()

    async def test_sync_function(self):
        """Test decorator with sync function."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        # For sync functions, we call them directly (not await)
        result = test_func()
        assert result == "success"
        assert call_count == 3

    async def test_non_retryable_exception(self):
        """Test that non-specified exceptions are not retried."""
        @retry_with_backoff(max_attempts=3, exceptions=(NetworkError,))
        async def test_func():
            raise ValueError("Non-retryable error")

        with pytest.raises(ValueError, match="Non-retryable error"):
            await test_func()


class TestValidateDecimalPositive:
    """Test cases for validate_decimal_positive function."""

    def test_valid_positive_decimal(self):
        """Test valid positive decimal."""
        result = validate_decimal_positive("10.5", "test_value")
        assert result == Decimal("10.5")

    def test_valid_zero_with_allow(self):
        """Test zero value when allowed."""
        result = validate_decimal_positive("0", "test_value", allow_zero=True)
        assert result == Decimal("0")

    def test_invalid_zero_without_allow(self):
        """Test zero value when not allowed."""
        with pytest.raises(ValidationError, match="test_value must be > 0"):
            validate_decimal_positive("0", "test_value", allow_zero=False)

    def test_invalid_negative(self):
        """Test negative value."""
        with pytest.raises(ValidationError, match="test_value must be > 0"):
            validate_decimal_positive("-5", "test_value")

    def test_invalid_string(self):
        """Test invalid string value."""
        with pytest.raises(ValidationError, match="test_value must be a valid number"):
            validate_decimal_positive("invalid", "test_value")

    def test_various_input_types(self):
        """Test various input types."""
        assert validate_decimal_positive(10, "test") == Decimal("10")
        assert validate_decimal_positive(10.5, "test") == Decimal("10.5")
        assert validate_decimal_positive(Decimal("5.5"), "test") == Decimal("5.5")


class TestValidateDecimalRange:
    """Test cases for validate_decimal_range function."""

    def test_valid_in_range(self):
        """Test value within valid range."""
        result = validate_decimal_range(
            "5.5",
            "test_value",
            min_value=Decimal("1"),
            max_value=Decimal("10")
        )
        assert result == Decimal("5.5")

    def test_below_minimum(self):
        """Test value below minimum."""
        with pytest.raises(ValidationError, match="test_value must be >= 1"):
            validate_decimal_range(
                "0.5",
                "test_value",
                min_value=Decimal("1"),
                max_value=Decimal("10")
            )

    def test_above_maximum(self):
        """Test value above maximum."""
        with pytest.raises(ValidationError, match="test_value must be <= 10"):
            validate_decimal_range(
                "15",
                "test_value",
                min_value=Decimal("1"),
                max_value=Decimal("10")
            )

    def test_exclusive_bounds(self):
        """Test exclusive bounds."""
        # Should fail at exact minimum
        with pytest.raises(ValidationError, match="test_value must be > 1"):
            validate_decimal_range(
                "1",
                "test_value",
                min_value=Decimal("1"),
                max_value=Decimal("10"),
                inclusive_min=False
            )

        # Should fail at exact maximum
        with pytest.raises(ValidationError, match="test_value must be < 10"):
            validate_decimal_range(
                "10",
                "test_value",
                min_value=Decimal("1"),
                max_value=Decimal("10"),
                inclusive_max=False
            )

    def test_no_bounds(self):
        """Test with no bounds specified."""
        result = validate_decimal_range("5.5", "test_value")
        assert result == Decimal("5.5")


class TestSafeDecimalDivision:
    """Test cases for safe_decimal_division function."""

    def test_successful_division(self):
        """Test successful division."""
        result = safe_decimal_division("10", "2")
        assert result == Decimal("5")

    def test_division_by_zero_with_default(self):
        """Test division by zero with default value."""
        result = safe_decimal_division("10", "0", default=Decimal("-1"))
        assert result == Decimal("-1")

    def test_division_by_zero_without_default(self):
        """Test division by zero without default value."""
        with pytest.raises(ValidationError, match="result: Division by zero"):
            safe_decimal_division("10", "0")

    def test_invalid_inputs_with_default(self):
        """Test invalid inputs with default value."""
        result = safe_decimal_division("invalid", "2", default=Decimal("-1"))
        assert result == Decimal("-1")

    def test_invalid_inputs_without_default(self):
        """Test invalid inputs without default value."""
        with pytest.raises(ValidationError, match="result: Invalid division operation"):
            safe_decimal_division("invalid", "2")

    def test_various_input_types(self):
        """Test various input types."""
        assert safe_decimal_division(10, 2) == Decimal("5")
        assert safe_decimal_division(10.0, 2.0) == Decimal("5")
        assert safe_decimal_division(Decimal("10"), Decimal("2")) == Decimal("5")


class TestRateLimiter:
    """Test cases for RateLimiter class."""

    async def test_acquire_within_limit(self):
        """Test acquiring permits within rate limit."""
        limiter = RateLimiter(max_calls=2, time_window=1.0)
        
        # Should allow first two calls
        await limiter.acquire()
        await limiter.acquire()
        
        # Should be able to acquire without blocking
        assert limiter.can_acquire() is False  # At limit now

    async def test_blocking_when_limit_exceeded(self):
        """Test blocking when rate limit is exceeded."""
        limiter = RateLimiter(max_calls=1, time_window=0.1)
        
        # First call should succeed
        await limiter.acquire()
        
        # Second call should block
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()
        
        # Should have waited at least the time window
        assert end_time - start_time >= 0.1

    async def test_can_acquire(self):
        """Test can_acquire method."""
        limiter = RateLimiter(max_calls=2, time_window=1.0)
        
        assert limiter.can_acquire() is True
        
        await limiter.acquire()
        assert limiter.can_acquire() is True
        
        await limiter.acquire()
        assert limiter.can_acquire() is False

    async def test_time_window_reset(self):
        """Test that time window resets correctly."""
        limiter = RateLimiter(max_calls=1, time_window=0.1)
        
        await limiter.acquire()
        assert limiter.can_acquire() is False
        
        # Wait for time window to pass
        await asyncio.sleep(0.15)
        assert limiter.can_acquire() is True


class TestFormatDecimal:
    """Test cases for format_decimal function."""

    def test_basic_formatting(self):
        """Test basic decimal formatting."""
        result = format_decimal(Decimal("10.5000"))
        assert result == "10.5"

    def test_with_decimal_places(self):
        """Test formatting with specific decimal places."""
        result = format_decimal(Decimal("10.5"), decimal_places=4)
        assert result == "10.5000"

    def test_strip_trailing_zeros_false(self):
        """Test formatting without stripping trailing zeros."""
        result = format_decimal(Decimal("10.5000"), strip_trailing_zeros=False)
        assert result == "10.5000"

    def test_integer_value(self):
        """Test formatting integer value."""
        result = format_decimal(Decimal("10"))
        assert result == "10"

    def test_various_input_types(self):
        """Test various input types."""
        assert format_decimal("10.5000") == "10.5"
        assert format_decimal(10.5000) == "10.5"
        assert format_decimal(10) == "10"

    def test_very_small_decimal(self):
        """Test formatting very small decimal."""
        result = format_decimal(Decimal("0.0001000"))
        assert result == "0.0001"

    def test_rounding_with_decimal_places(self):
        """Test rounding when specifying decimal places."""
        result = format_decimal(Decimal("10.5678"), decimal_places=2)
        assert result == "10.57"
