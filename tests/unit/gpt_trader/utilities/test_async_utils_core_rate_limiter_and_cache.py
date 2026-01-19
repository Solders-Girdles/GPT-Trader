"""Tests for core async rate limiting and caching utilities."""

from __future__ import annotations

import asyncio
import time

import pytest

from gpt_trader.utilities.async_tools import (  # naming: allow
    AsyncCache,
    AsyncRateLimiter,
    async_cache,
    async_rate_limit,
)


class TestAsyncRateLimiter:
    """Test AsyncRateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self) -> None:
        """Test basic rate limiting."""
        limiter = AsyncRateLimiter(rate_limit=10.0, burst_limit=2)

        start_time = time.time()

        # Should allow first two requests immediately
        await limiter.acquire()
        await limiter.acquire()

        # Third request should be delayed
        await limiter.acquire()

        elapsed = time.time() - start_time
        assert elapsed >= 0.1  # Should be delayed

    @pytest.mark.asyncio
    async def test_rate_limiter_context_manager(self) -> None:
        """Test rate limiter as context manager."""
        limiter = AsyncRateLimiter(rate_limit=5.0, burst_limit=1)

        async with limiter:
            # Should acquire token
            pass
        assert limiter.tokens == 0.0

    @pytest.mark.asyncio
    async def test_rate_limit_decorator(self) -> None:
        """Test rate limit decorator."""

        @async_rate_limit(rate_limit=10.0, burst_limit=2)
        async def rate_limited_func(x: int) -> int:
            return x * 2

        # Should not raise exception
        result = await rate_limited_func(5)
        assert result == 10


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
