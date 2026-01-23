"""Tests for async rate limiting utilities."""

from __future__ import annotations

import pytest

from gpt_trader.utilities.async_tools import AsyncRateLimiter, async_rate_limit  # naming: allow


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class TestAsyncRateLimiter:
    """Test AsyncRateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self) -> None:
        """Test basic rate limiting."""
        clock = _FakeClock()
        limiter = AsyncRateLimiter(
            rate_limit=10.0, burst_limit=2, time_fn=clock.time, sleep=clock.sleep
        )

        # Should allow first two requests immediately.
        await limiter.acquire()
        await limiter.acquire()

        # Third request should be delayed.
        await limiter.acquire()

        assert len(clock.slept) == 1
        assert clock.slept[0] == pytest.approx(0.1)
        assert clock.now == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_rate_limiter_context_manager(self) -> None:
        """Test rate limiter as context manager."""
        limiter = AsyncRateLimiter(rate_limit=5.0, burst_limit=1)

        async with limiter:
            # Should acquire token.
            pass
        assert limiter.tokens == 0.0

    @pytest.mark.asyncio
    async def test_rate_limit_decorator(self) -> None:
        """Test rate limit decorator."""

        @async_rate_limit(rate_limit=10.0, burst_limit=2)
        async def rate_limited_func(x: int) -> int:
            return x * 2

        # Should not raise exception.
        result = await rate_limited_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_rate_limiter_edge_cases(self) -> None:
        """Test rate limiter edge cases."""
        # Very high rate limit should not delay.
        clock = _FakeClock()
        limiter = AsyncRateLimiter(
            rate_limit=1000.0, burst_limit=10, time_fn=clock.time, sleep=clock.sleep
        )
        for _ in range(5):
            await limiter.acquire()

        assert clock.slept == []
        assert limiter.tokens == 5.0
