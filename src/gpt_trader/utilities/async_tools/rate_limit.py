"""Async rate limiting helpers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRateLimiter:
    """Rate limiter for async operations."""

    def __init__(self, rate_limit: float, burst_limit: int = 1) -> None:
        self.rate_limit = rate_limit
        self.burst_limit = burst_limit
        self.tokens: float = float(burst_limit)
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update
            burst_cap = float(self.burst_limit)
            self.tokens = min(burst_cap, self.tokens + time_passed * self.rate_limit)
            self.last_update = now
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate_limit
                await asyncio.sleep(wait_time)
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
