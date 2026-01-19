"""Tests for gather_with_concurrency and wait_for_first async helpers."""

from __future__ import annotations

import asyncio
import time

import pytest

from gpt_trader.utilities.async_tools import gather_with_concurrency, wait_for_first


class TestGatherWithConcurrency:
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
    @pytest.mark.asyncio
    async def test_wait_for_first(self) -> None:
        async def fast_operation():
            await asyncio.sleep(0.01)
            return "fast"

        async def slow_operation():
            await asyncio.sleep(0.1)
            return "slow"

        result = await wait_for_first([fast_operation(), slow_operation()])
        assert result == "fast"

    @pytest.mark.asyncio
    async def test_wait_for_first_performance(self) -> None:
        results: list[str] = []

        async def operation_with_delay(delay: float, result: str):
            await asyncio.sleep(delay)
            results.append(result)
            return result

        coroutines = [
            operation_with_delay(0.05, "slow"),
            operation_with_delay(0.01, "fast"),
            operation_with_delay(0.03, "medium"),
        ]

        first_result = await wait_for_first(coroutines)
        assert first_result == "fast"

        await asyncio.sleep(0.1)

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
        async def slow_op():
            await asyncio.sleep(1.0)
            return "slow"

        async def fast_op():
            await asyncio.sleep(0.01)
            return "fast"

        task = asyncio.create_task(wait_for_first([slow_op(), fast_op()]))

        await asyncio.sleep(0.02)

        assert task.done()
        assert task.result() == "fast"


class TestAsyncPerformance:
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
