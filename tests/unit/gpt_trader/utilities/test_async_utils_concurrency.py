"""Tests for async concurrency helpers."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from gpt_trader.utilities.async_tools import (  # naming: allow
    AsyncBatchProcessor,
    AsyncContextManager,
    BoundedToThread,
    async_timeout,
)


class TestBoundedToThread:
    """Test BoundedToThread concurrency limiting."""

    def test_invalid_concurrency(self) -> None:
        with pytest.raises(ValueError, match="max_concurrency"):
            BoundedToThread(max_concurrency=0)

    @pytest.mark.asyncio
    async def test_concurrency_is_bounded(self) -> None:
        limit = 3
        limiter = BoundedToThread(max_concurrency=limit)

        lock = threading.Lock()
        started = threading.Event()
        release = threading.Event()
        state = {"current": 0, "max": 0}

        def work(value: int) -> int:
            with lock:
                state["current"] += 1
                state["max"] = max(state["max"], state["current"])
                if state["current"] == limit:
                    started.set()

            # Keep threads busy until the test releases them.
            release.wait(timeout=2)

            with lock:
                state["current"] -= 1
            return value

        tasks = [asyncio.create_task(limiter.run(work, idx)) for idx in range(10)]

        # Wait until we observe the concurrency peak.
        await asyncio.wait_for(asyncio.to_thread(started.wait, 2), timeout=2.5)
        release.set()
        results = await asyncio.gather(*tasks)

        assert results == list(range(10))
        assert state["max"] == limit

    @pytest.mark.asyncio
    async def test_custom_executor_path(self) -> None:
        class RecordingExecutor(ThreadPoolExecutor):
            def __init__(self) -> None:
                super().__init__(max_workers=1)
                self.submit_calls = 0

            def submit(self, *args, **kwargs):
                self.submit_calls += 1
                return super().submit(*args, **kwargs)

        executor = RecordingExecutor()
        limiter = BoundedToThread(max_concurrency=1, executor=executor)

        result = await limiter.run(lambda: "ok")

        assert result == "ok"
        assert executor.submit_calls >= 1

        limiter.shutdown()
        assert executor.submit(lambda: "still-alive").result() == "still-alive"
        executor.shutdown(wait=True)

    @pytest.mark.asyncio
    async def test_dedicated_executor_shutdown(self) -> None:
        limiter = BoundedToThread(max_concurrency=1, use_dedicated_executor=True)

        result = await limiter.run(lambda: 42)

        assert result == 42
        assert limiter._executor is not None
        limiter.shutdown()


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
