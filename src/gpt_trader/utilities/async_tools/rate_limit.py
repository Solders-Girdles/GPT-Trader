"""Async rate limiting helpers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRateLimiter:
    """Rate limiter for async operations."""

    def __init__(
        self,
        rate_limit: float,
        burst_limit: int = 1,
        *,
        time_fn: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.rate_limit = rate_limit
        self.burst_limit = burst_limit
        self.tokens: float = float(burst_limit)
        # Use monotonic time for rate limiting so we aren't sensitive to system clock changes.
        self._time_fn = time_fn or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self.last_update = self._time_fn()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self._time_fn()
            time_passed = now - self.last_update
            burst_cap = float(self.burst_limit)
            self.tokens = min(burst_cap, self.tokens + time_passed * self.rate_limit)
            self.last_update = now
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate_limit
                await self._sleep(wait_time)
                # Advance last_update to avoid double-counting time slept as token regeneration.
                self.last_update = self._time_fn()
                self.tokens = 0.0
            else:
                self.tokens -= 1

    async def __aenter__(self) -> None:
        await self.acquire()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


def async_rate_limit(
    rate_limit: float, burst_limit: int = 1
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    limiter = AsyncRateLimiter(rate_limit, burst_limit)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


__all__ = ["AsyncRateLimiter", "async_rate_limit"]
