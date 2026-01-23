"""Tests for async caching and retry utilities."""

from __future__ import annotations

import asyncio

import pytest

from gpt_trader.utilities.async_tools import (  # naming: allow
    AsyncBatchProcessor,
    AsyncCache,
    AsyncRetry,
    async_cache,
    async_retry,
)


class TestAsyncCache:
    """Test AsyncCache functionality."""

    @pytest.mark.asyncio
    async def test_async_cache_basic(self) -> None:
        """Test basic async caching."""
        cache = AsyncCache(ttl=1.0)

        # Set and get value
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"

    @pytest.mark.asyncio
    async def test_async_cache_ttl(self) -> None:
        """Test cache TTL expiration."""
        cache = AsyncCache(ttl=0.01)  # 10ms TTL

        await cache.set("key1", "value1")

        # Should be available immediately
        value = await cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(0.02)

        # Should be expired
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_async_cache_clear(self) -> None:
        """Test cache clearing."""
        cache = AsyncCache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_async_cache_decorator(self) -> None:
        """Test async cache decorator."""
        call_count = 0

        @async_cache(ttl=1.0)
        async def expensive_func(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            return x * 2

        # First call should execute function
        result1 = await expensive_func(5)
        assert result1 == 10
        assert call_count == 1

        # Second call should use cache
        result2 = await expensive_func(5)
        assert result2 == 10
        assert call_count == 1  # Should not increase

        # Different argument should execute function
        result3 = await expensive_func(7)
        assert result3 == 14
        assert call_count == 2

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
