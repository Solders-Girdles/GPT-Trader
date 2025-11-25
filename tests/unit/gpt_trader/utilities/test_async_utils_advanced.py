"""Tests for advanced async utilities."""

from __future__ import annotations

import asyncio
import time

import pytest

from gpt_trader.utilities.async_utils import (  # naming: allow
    gather_with_concurrency,
    is_async_func,
    run_async_if_needed,
    wait_for_first,
)


class TestAsyncUtilities:
    """Test general async utility functions."""

    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self) -> None:
        """Test gather with concurrency limit."""

        async def slow_operation(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        operations = [slow_operation(i) for i in range(10)]
        results = await gather_with_concurrency(operations, max_concurrency=3)

        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_wait_for_first(self) -> None:
        """Test waiting for first completed coroutine."""

        async def fast_operation():
            await asyncio.sleep(0.01)
            return "fast"

        async def slow_operation():
            await asyncio.sleep(0.1)
            return "slow"

        result = await wait_for_first([fast_operation(), slow_operation()])
        assert result == "fast"

    def test_is_async_func(self) -> None:
        """Test async function detection."""

        async def async_func():
            pass

        def sync_func():
            pass

        assert is_async_func(async_func)
        assert not is_async_func(sync_func)

    @pytest.mark.asyncio
    async def test_run_async_if_needed(self) -> None:
        """Test running async functions if needed."""

        async def async_func(x: int) -> int:
            return x * 2

        def sync_func(x: int) -> int:
            return x * 3

        # Async function should return coroutine
        result = run_async_if_needed(async_func, 5)
        assert asyncio.iscoroutine(result)
        assert await result == 10

        # Sync function should return result directly
        result = run_async_if_needed(sync_func, 5)
        assert result == 15


class TestAsyncPerformance:
    """Test performance-related async utilities."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_performance(self) -> None:
        """Test that concurrent execution is faster than sequential."""

        async def slow_operation(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        # Create fresh coroutines for each test
        operations1 = [slow_operation(i) for i in range(5)]
        operations2 = [slow_operation(i) for i in range(5)]

        # Sequential execution
        start_time = time.time()
        sequential_results = []
        for op in operations1:
            sequential_results.append(await op)
        sequential_time = time.time() - start_time

        # Concurrent execution
        start_time = time.time()
        concurrent_results = await asyncio.gather(*operations2)
        concurrent_time = time.time() - start_time

        # Concurrent should be faster
        assert concurrent_time < sequential_time
        assert sequential_results == concurrent_results

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_performance(self) -> None:
        """Test that gather_with_concurrency respects limits."""

        async def tracked_operation(operation_id: int, active_counter: list) -> int:
            active_counter[0] += 1
            max_active = active_counter[1]
            max_active[0] = max(max_active[0], active_counter[0])

            await asyncio.sleep(0.01)

            active_counter[0] -= 1
            return operation_id * 2

        # Track concurrent operations
        max_active = [0]  # Maximum concurrent operations seen
        active_counter = [0, max_active]  # Current active operations and max tracker

        operations = [tracked_operation(i, active_counter) for i in range(10)]
        results = await gather_with_concurrency(operations, max_concurrency=3)

        # Should not exceed concurrency limit
        assert max_active[0] <= 3
        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_wait_for_first_performance(self) -> None:
        """Test wait_for_first with different timing scenarios."""
        results = []

        async def operation_with_delay(delay: int, result: str):
            await asyncio.sleep(delay)
            results.append(result)
            return result

        # Test with operations that complete at different times
        coroutines = [
            operation_with_delay(0.05, "slow"),
            operation_with_delay(0.01, "fast"),
            operation_with_delay(0.03, "medium"),
        ]

        first_result = await wait_for_first(coroutines)
        assert first_result == "fast"

        # Wait a bit more to ensure other operations complete
        await asyncio.sleep(0.1)

        # Should have completed all operations
        assert len(results) == 3
        assert "fast" in results


class TestAsyncEdgeCasesAdvanced:
    """Test advanced edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_empty(self) -> None:
        """Test gather with empty operations list."""
        results = await gather_with_concurrency([], max_concurrency=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_single(self) -> None:
        """Test gather with single operation."""

        async def single_op():
            return "single"

        results = await gather_with_concurrency([single_op()], max_concurrency=5)
        assert results == ["single"]

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_exceptions(self) -> None:
        """Test gather with exceptions in operations."""

        async def failing_op(op_id: int):
            if op_id == 5:
                raise ValueError(f"Operation {op_id} failed")
            await asyncio.sleep(0.01)
            return op_id * 2

        operations = [failing_op(i) for i in range(10)]

        # Should propagate exceptions
        with pytest.raises(ValueError, match="Operation 5 failed"):
            await gather_with_concurrency(operations, max_concurrency=3, return_exceptions=False)

    @pytest.mark.asyncio
    async def test_wait_for_first_empty(self) -> None:
        """Test wait_for_first with empty list."""
        with pytest.raises(ValueError, match="At least one coroutine must be provided"):
            await wait_for_first([])

    @pytest.mark.asyncio
    async def test_wait_for_first_all_fail(self) -> None:
        """Test wait_for_first when all operations fail."""

        async def failing_op():
            raise ValueError("All fail")

        coroutines = [failing_op(), failing_op()]

        with pytest.raises(ValueError, match="All fail"):
            await wait_for_first(coroutines)

    @pytest.mark.asyncio
    async def test_wait_for_first_cancellation(self) -> None:
        """Test wait_for_first with cancellation."""

        async def slow_op():
            await asyncio.sleep(1.0)
            return "slow"

        async def fast_op():
            await asyncio.sleep(0.01)
            return "fast"

        # Create task that we can cancel
        task = asyncio.create_task(wait_for_first([slow_op(), fast_op()]))

        # Wait for fast operation to complete
        await asyncio.sleep(0.02)

        # Task should be completed with fast result
        assert task.done()
        assert task.result() == "fast"

    def test_is_async_func_edge_cases(self) -> None:
        """Test async function detection edge cases."""

        # Test with lambda functions
        async def async_lambda():
            pass

        def sync_lambda() -> None:
            return None

        assert is_async_func(async_lambda)
        assert not is_async_func(sync_lambda)

        # Test with methods
        class TestClass:
            async def async_method(self):
                pass

            def sync_method(self):
                pass

        obj = TestClass()
        assert is_async_func(obj.async_method)
        assert not is_async_func(obj.sync_method)

    @pytest.mark.asyncio
    async def test_run_async_if_needed_edge_cases(self) -> None:
        """Test run_async_if_needed edge cases."""
        # Test with None
        result = run_async_if_needed(lambda: None)
        assert result is None

        # Test with complex objects
        async def async_complex():
            return {"key": "value", "nested": [1, 2, 3]}

        result = run_async_if_needed(async_complex)
        assert asyncio.iscoroutine(result)
        assert await result == {"key": "value", "nested": [1, 2, 3]}


class TestAsyncIntegrationScenarios:
    """Test integration scenarios combining multiple async utilities."""

    @pytest.mark.asyncio
    async def test_rate_limiting_with_batch_processing(self) -> None:
        """Test combining rate limiting with batch processing."""
        from gpt_trader.utilities.async_utils import (  # naming: allow
            AsyncBatchProcessor,
            AsyncRateLimiter,
        )

        limiter = AsyncRateLimiter(rate_limit=50.0, burst_limit=5)
        processor = AsyncBatchProcessor(batch_size=3, delay_between_batches=0.01)

        async def rate_limited_operation(x: int) -> int:
            async with limiter:
                await asyncio.sleep(0.001)
                return x * 2

        operations = [rate_limited_operation(i) for i in range(15)]
        results = await processor.process_batch(operations)

        assert len(results) == 15
        assert results == [i * 2 for i in range(15)]

    @pytest.mark.asyncio
    async def test_caching_with_retry(self) -> None:
        """Test combining caching with retry logic."""
        from gpt_trader.utilities.async_utils import AsyncCache, AsyncRetry  # naming: allow

        cache = AsyncCache(ttl=1.0)
        retry = AsyncRetry(max_attempts=3, base_delay=0.01)

        call_count = 0

        async def cached_retry_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1

            # Check cache first
            cache_key = f"op_{x}"
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Simulate occasional failure
            if call_count == 2 and x == 5:
                raise ValueError("Temporary failure")

            result = x * 3
            await cache.set(cache_key, result)
            return result

        # Test multiple calls
        results = []
        for i in range(5):
            for _ in range(2):  # Call each twice
                result = await retry.execute(cached_retry_operation, i)
                results.append(result)

        # Should have cached results for second calls
        assert len(results) == 10
        assert results[0] == results[1]  # First operation cached
        assert results[2] == results[3]  # Second operation cached

    @pytest.mark.asyncio
    async def test_timeout_with_concurrency(self) -> None:
        """Test combining timeout with concurrent execution."""
        from gpt_trader.utilities.async_utils import async_timeout  # naming: allow

        @async_timeout(0.1)
        async def timed_operation(x: int) -> int:
            await asyncio.sleep(0.02)  # Fast enough
            return x * 2

        operations = [timed_operation(i) for i in range(10)]
        results = await gather_with_concurrency(operations, max_concurrency=5)

        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_complex_async_workflow(self) -> None:
        """Test a complex workflow combining multiple utilities."""
        from gpt_trader.utilities.async_utils import (  # naming: allow
            AsyncCache,
            AsyncRateLimiter,
            AsyncRetry,
            async_timeout,
            gather_with_concurrency,
        )

        # Setup components
        cache = AsyncCache(ttl=2.0)
        limiter = AsyncRateLimiter(rate_limit=20.0, burst_limit=3)
        retry = AsyncRetry(max_attempts=2, base_delay=0.01)

        # Complex operation
        @async_timeout(0.1)
        async def complex_operation(item_id: int) -> dict:
            async with limiter:
                # Check cache
                cache_key = f"complex_{item_id}"
                cached = await cache.get(cache_key)
                if cached:
                    return cached

                # Simulate processing with potential failure
                if item_id == 7:  # Simulate failure for specific item
                    raise ValueError(f"Processing failed for item {item_id}")

                await asyncio.sleep(0.01)  # Simulate work
                result = {
                    "id": item_id,
                    "processed": True,
                    "timestamp": time.time(),
                    "value": item_id * 10,
                }

                await cache.set(cache_key, result)
                return result

        # Execute workflow
        item_ids = list(range(10))

        async def process_item_with_retry(item_id: int):
            return await retry.execute(complex_operation, item_id)

        operations = [process_item_with_retry(i) for i in item_ids]
        results = await gather_with_concurrency(operations, max_concurrency=3)

        # Should have processed 9 items (item 7 should fail after retry)
        assert len(results) == 10

        # Check that successful items were processed correctly
        successful_results = [r for r in results if isinstance(r, dict)]
        assert len(successful_results) == 9

        # Check cache usage
        cache_hits = 0
        for result in successful_results:
            if result["id"] != 7:  # Exclude failed item
                cached = await cache.get(f"complex_{result['id']}")
                assert cached is not None
                cache_hits += 1

        assert cache_hits == 9
