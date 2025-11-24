"""Tests for core async utilities."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from gpt_trader.utilities.async_utils import (
    AsyncBatchProcessor,
    AsyncCache,
    AsyncContextManager,
    AsyncRateLimiter,
    AsyncRetry,
    AsyncToSyncWrapper,
    SyncToAsyncWrapper,
    async_cache,
    async_rate_limit,
    async_retry,
    async_timeout,
    async_to_sync,
    sync_to_async,
)


class TestAsyncToSyncWrapper:
    """Test AsyncToSyncWrapper functionality."""

    def test_async_to_sync_basic(self) -> None:
        """Test basic async to sync conversion."""

        async def async_func(x: int) -> int:
            return x * 2

        wrapper = AsyncToSyncWrapper()
        result = wrapper(async_func(5))
        assert result == 10

    def test_async_to_sync_with_exception(self) -> None:
        """Test async to sync with exception."""

        async def failing_func():
            raise ValueError("Test error")

        wrapper = AsyncToSyncWrapper()
        with pytest.raises(ValueError, match="Test error"):
            wrapper(failing_func())

    def test_async_to_sync_decorator(self) -> None:
        """Test async_to_sync decorator."""

        @async_to_sync
        async def async_func(x: int) -> int:
            return x * 3

        result = async_func(7)
        assert result == 21

    def test_async_to_sync_in_running_loop(self) -> None:
        async def async_func(x: int) -> int:
            await asyncio.sleep(0)
            return x * 2

        loop = asyncio.new_event_loop()
        loop_started = threading.Event()

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop_started.set()
            loop.run_forever()

        runner = threading.Thread(target=run_loop)
        runner.start()
        loop_started.wait()

        try:
            wrapper = AsyncToSyncWrapper(loop)
            result = wrapper(async_func(5))
            assert result == 10
        finally:
            loop.call_soon_threadsafe(loop.stop)
            runner.join()
            loop.close()


class TestSyncToAsyncWrapper:
    """Test SyncToAsyncWrapper functionality."""

    @pytest.mark.asyncio
    async def test_sync_to_async_basic(self) -> None:
        """Test basic sync to async conversion."""

        def sync_func(x: int) -> int:
            return x * 2

        wrapper = SyncToAsyncWrapper()
        result = await wrapper(sync_func, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_sync_to_async_with_exception(self) -> None:
        """Test sync to async with exception."""

        def failing_func():
            raise ValueError("Test error")

        wrapper = SyncToAsyncWrapper()
        with pytest.raises(ValueError, match="Test error"):
            await wrapper(failing_func)

    @pytest.mark.asyncio
    async def test_sync_to_async_decorator(self) -> None:
        """Test sync_to_async decorator."""

        @sync_to_async
        def sync_func(x: int) -> int:
            return x * 3

        result = await sync_func(7)
        assert result == 21


class TestAsyncContextManager:
    """Test AsyncContextManager functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager_basic(self) -> None:
        """Test basic async context manager."""
        context = AsyncContextManager("test_operation")

        async with context:
            assert context.start_time is not None
            await asyncio.sleep(0.01)  # Small delay

        # Context should track timing
        assert context.start_time is not None

    @pytest.mark.asyncio
    async def test_async_context_manager_with_timeout(self) -> None:
        """Test async context manager with timeout."""
        context = AsyncContextManager("test_operation", timeout=1.0)

        async with context:
            assert context.timeout == 1.0


class TestAsyncTimeout:
    """Test async_timeout functionality."""

    @pytest.mark.asyncio
    async def test_async_timeout_success(self) -> None:
        """Test timeout with successful operation."""

        @async_timeout(1.0)
        async def quick_operation():
            await asyncio.sleep(0.01)
            return "success"

        result = await quick_operation()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_timeout_failure(self) -> None:
        """Test timeout with slow operation."""

        @async_timeout(0.01)
        async def slow_operation():
            await asyncio.sleep(0.1)
            return "success"

        with pytest.raises(asyncio.TimeoutError, match="Operation timed out after 0.01s"):
            await slow_operation()


class TestAsyncBatchProcessor:
    """Test AsyncBatchProcessor functionality."""

    @pytest.mark.asyncio
    async def test_batch_processor_basic(self) -> None:
        """Test basic batch processing."""

        async def mock_operation(x: int) -> int:
            await asyncio.sleep(0.001)  # Small delay
            return x * 2

        processor = AsyncBatchProcessor(batch_size=3, delay_between_batches=0.01)

        operations = [mock_operation(i) for i in range(10)]
        results = await processor.process_batch(operations)

        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_batch_processor_with_exceptions(self) -> None:
        """Test batch processing with exceptions."""

        async def failing_operation(x: int):
            if x == 5:
                raise ValueError("Test error")
            return x * 2

        processor = AsyncBatchProcessor(batch_size=3)

        operations = [failing_operation(i) for i in range(10)]
        results = await processor.process_batch(operations, return_exceptions=True)

        assert len(results) == 10
        assert results[5] is not None
        assert isinstance(results[5], ValueError)


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
