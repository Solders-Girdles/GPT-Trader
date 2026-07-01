"""Tests for higher-level async utility composition scenarios."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest

from gpt_trader.utilities.async_tools import gather_with_concurrency


class TestAsyncScenarios:
    @dataclass
    class FakeClock:
        current: float = 1_000_000.0

        def time(self) -> float:
            return self.current

        def sleep(self, seconds: float) -> None:
            self.current += seconds

    @pytest.fixture(autouse=True)
    def _deterministic_time(self, monkeypatch: pytest.MonkeyPatch) -> TestAsyncScenarios.FakeClock:
        clock = self.FakeClock()
        monkeypatch.setattr(time, "time", clock.time)

        original_sleep = asyncio.sleep

        async def fake_sleep(delay: float = 0.0, result: Any | None = None) -> Any | None:
            clock.sleep(delay)
            await original_sleep(0)
            return result

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)
        return clock

    @pytest.mark.asyncio
    async def test_rate_limiting_with_batch_processing(self) -> None:
        from gpt_trader.utilities.async_tools import (  # naming: allow
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
    async def test_timeout_with_concurrency(self) -> None:
        from gpt_trader.utilities.async_tools import async_timeout  # naming: allow

        @async_timeout(0.1)
        async def timed_operation(x: int) -> int:
            await asyncio.sleep(0.02)
            return x * 2

        operations = [timed_operation(i) for i in range(10)]
        results = await gather_with_concurrency(operations, max_concurrency=5)

        assert len(results) == 10
        assert results == [i * 2 for i in range(10)]

    @pytest.mark.asyncio
    async def test_complex_async_workflow(self) -> None:
        from gpt_trader.utilities.async_tools import (  # naming: allow
            AsyncCache,
            AsyncRateLimiter,
            async_timeout,
            gather_with_concurrency,
        )

        cache = AsyncCache(ttl=2.0)
        limiter = AsyncRateLimiter(rate_limit=20.0, burst_limit=3)

        @async_timeout(0.1)
        async def complex_operation(item_id: int) -> dict:
            async with limiter:
                cache_key = f"complex_{item_id}"
                cached = await cache.get(cache_key)
                if cached:
                    return cached

                if item_id == 7:
                    raise ValueError(f"Processing failed for item {item_id}")

                await asyncio.sleep(0.01)
                result = {
                    "id": item_id,
                    "processed": True,
                    "timestamp": time.time(),
                    "value": item_id * 10,
                }

                await cache.set(cache_key, result)
                return result

        item_ids = list(range(10))

        operations = [complex_operation(i) for i in item_ids]
        results = await gather_with_concurrency(operations, max_concurrency=3)

        assert len(results) == 10

        successful_results = [r for r in results if isinstance(r, dict)]
        assert len(successful_results) == 9

        cache_hits = 0
        for result in successful_results:
            if result["id"] != 7:
                cached = await cache.get(f"complex_{result['id']}")
                assert cached is not None
                cache_hits += 1

        assert cache_hits == 9
