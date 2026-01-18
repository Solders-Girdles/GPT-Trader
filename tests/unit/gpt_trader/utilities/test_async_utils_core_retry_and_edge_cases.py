"""Tests for core async retry utilities and edge cases."""

from __future__ import annotations

import time

import pytest

from gpt_trader.utilities.async_tools import (  # naming: allow
    AsyncBatchProcessor,
    AsyncCache,
    AsyncRateLimiter,
    AsyncRetry,
    async_retry,
)


class TestAsyncRetry:
    """Test AsyncRetry functionality."""

    @pytest.mark.asyncio
    async def test_async_retry_success(self) -> None:
        """Test retry with eventual success."""
        call_count = 0

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        retry = AsyncRetry(max_attempts=3, base_delay=0.01)
        result = await retry.execute(failing_func)

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_failure(self) -> None:
        """Test retry with eventual failure."""

        async def always_failing_func():
            raise ValueError("Permanent failure")

        retry = AsyncRetry(max_attempts=2, base_delay=0.01)

        with pytest.raises(ValueError, match="Permanent failure"):
            await retry.execute(always_failing_func)

    @pytest.mark.asyncio
    async def test_async_retry_decorator(self) -> None:
        """Test async retry decorator."""
        call_count = 0

        @async_retry(max_attempts=3, base_delay=0.01)
        async def retry_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "success"

        result = await retry_func()
        assert result == "success"
        assert call_count == 2


class TestAsyncEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_rate_limiter_edge_cases(self) -> None:
        """Test rate limiter edge cases."""
        # Very high rate limit should not delay
        limiter = AsyncRateLimiter(rate_limit=1000.0, burst_limit=10)

        start_time = time.time()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.time() - start_time

        # Should be very fast
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_cache_edge_cases(self) -> None:
        """Test cache edge cases."""
        cache = AsyncCache(ttl=0.001)  # Very short TTL

        await cache.set("key", "value")

        # Get immediately after set
        value = await cache.get("key")
        assert value == "value"

        # Cleanup expired should remove expired entries
        removed = await cache.cleanup_expired()
        assert removed >= 0

    @pytest.mark.asyncio
    async def test_batch_processor_empty(self) -> None:
        """Test batch processor with empty operations."""
        processor = AsyncBatchProcessor()
        results = await processor.process_batch([])
        assert results == []

    @pytest.mark.asyncio
    async def test_retry_zero_attempts(self) -> None:
        """Test retry with zero attempts."""

        async def func():
            return "result"

        retry = AsyncRetry(max_attempts=1)  # Only one attempt
        result = await retry.execute(func)
        assert result == "result"
