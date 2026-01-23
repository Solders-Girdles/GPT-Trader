"""Tests for gather_with_concurrency and wait_for_first async helpers."""

from __future__ import annotations

import asyncio
import time

import pytest

from gpt_trader.utilities.async_tools import gather_with_concurrency, wait_for_first


class TestGatherWithConcurrency:
    """Test gather_with_concurrency functionality."""

    @pytest.mark.asyncio
    async def test_gather_with_concurrency(self) -> None:
        async def slow_operation(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        operations = [slow_operation(i) for i in range(10)]
        results = await gather_with_concurrency(operations, max_concurrency=3)

        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_performance(self) -> None:
        async def tracked_operation(operation_id: int, active_counter: list) -> int:
            active_counter[0] += 1
            max_active = active_counter[1]
            max_active[0] = max(max_active[0], active_counter[0])

            await asyncio.sleep(0.01)

            active_counter[0] -= 1
            return operation_id * 2

        max_active = [0]
        active_counter = [0, max_active]

        operations = [tracked_operation(i, active_counter) for i in range(10)]
        results = await gather_with_concurrency(operations, max_concurrency=3)

        assert max_active[0] <= 3
        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_empty(self) -> None:
        results = await gather_with_concurrency([], max_concurrency=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_single(self) -> None:
        async def single_op():
            return "single"

        results = await gather_with_concurrency([single_op()], max_concurrency=5)
        assert results == ["single"]

    @pytest.mark.asyncio
    async def test_gather_with_concurrency_exceptions(self) -> None:
        async def failing_op(op_id: int):
            if op_id == 5:
                raise ValueError(f"Operation {op_id} failed")
            await asyncio.sleep(0.01)
            return op_id * 2

        operations = [failing_op(i) for i in range(10)]

        with pytest.raises(ValueError, match="Operation 5 failed"):
            await gather_with_concurrency(operations, max_concurrency=3, return_exceptions=False)


class TestWaitForFirst:
    """Test wait_for_first functionality."""

    @pytest.mark.asyncio
    async def test_wait_for_first(self) -> None:
        slow_release = asyncio.Event()
        slow_done = asyncio.Event()

        async def slow_operation() -> str:
            # Block until explicitly released so we can assert wait_for_first returns
            # without waiting on the slow task.
            await slow_release.wait()
            slow_done.set()
            return "slow"

        fast_release = asyncio.Event()

        async def fast_operation() -> str:
            await fast_release.wait()
            return "fast"

        # Make the fast task immediately completable.
        fast_release.set()

        result = await wait_for_first([fast_operation(), slow_operation()])
        assert result == "fast"
        assert slow_done.is_set() is False

        slow_release.set()
        await slow_done.wait()

    @pytest.mark.asyncio
    async def test_wait_for_first_performance(self) -> None:
        results: list[str] = []

        async def operation_with_gate(
            gate: asyncio.Event,
            started: asyncio.Event,
            done: asyncio.Event,
            result: str,
        ) -> str:
            started.set()
            await gate.wait()
            results.append(result)
            done.set()
            return result

        slow_gate = asyncio.Event()
        fast_gate = asyncio.Event()
        medium_gate = asyncio.Event()

        slow_started = asyncio.Event()
        fast_started = asyncio.Event()
        medium_started = asyncio.Event()

        slow_done = asyncio.Event()
        fast_done = asyncio.Event()
        medium_done = asyncio.Event()

        coroutines = [
            operation_with_gate(slow_gate, slow_started, slow_done, "slow"),
            operation_with_gate(fast_gate, fast_started, fast_done, "fast"),
            operation_with_gate(medium_gate, medium_started, medium_done, "medium"),
        ]

        wait_task = asyncio.create_task(wait_for_first(coroutines))
        await asyncio.gather(slow_started.wait(), fast_started.wait(), medium_started.wait())

        fast_gate.set()
        first_result = await wait_task
        assert first_result == "fast"

        medium_gate.set()
        slow_gate.set()

        # wait_for_first doesn't cancel the remaining work; ensure it completes.
        await asyncio.gather(slow_done.wait(), fast_done.wait(), medium_done.wait())

        assert len(results) == 3
        assert "fast" in results

    @pytest.mark.asyncio
    async def test_wait_for_first_empty(self) -> None:
        with pytest.raises(ValueError, match="At least one coroutine must be provided"):
            await wait_for_first([])

    @pytest.mark.asyncio
    async def test_wait_for_first_all_fail(self) -> None:
        async def failing_op():
            raise ValueError("All fail")

        coroutines = [failing_op(), failing_op()]

        with pytest.raises(ValueError, match="All fail"):
            await wait_for_first(coroutines)

    @pytest.mark.asyncio
    async def test_wait_for_first_cancellation(self) -> None:
        slow_release = asyncio.Event()
        slow_done = asyncio.Event()

        async def slow_op() -> str:
            await slow_release.wait()
            slow_done.set()
            return "slow"

        fast_release = asyncio.Event()

        async def fast_op() -> str:
            await fast_release.wait()
            return "fast"

        task = asyncio.create_task(wait_for_first([slow_op(), fast_op()]))

        fast_release.set()
        result = await task

        assert task.done()
        assert result == "fast"

        slow_release.set()
        await slow_done.wait()


class TestAsyncPerformance:
    """Test async performance characteristics."""

    @pytest.mark.asyncio
    async def test_concurrent_execution_performance(self) -> None:
        async def slow_operation(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        operations1 = [slow_operation(i) for i in range(5)]
        operations2 = [slow_operation(i) for i in range(5)]

        start_time = time.time()
        sequential_results = []
        for op in operations1:
            sequential_results.append(await op)
        sequential_time = time.time() - start_time

        start_time = time.time()
        concurrent_results = await asyncio.gather(*operations2)
        concurrent_time = time.time() - start_time

        assert concurrent_time < sequential_time
        assert sequential_results == concurrent_results
