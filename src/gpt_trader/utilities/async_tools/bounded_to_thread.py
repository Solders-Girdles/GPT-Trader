"""Concurrency-limited helper for running sync callables via asyncio.to_thread()."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from .rate_limit import AsyncRateLimiter

T = TypeVar("T")


class BoundedToThread:
    """Run blocking callables in a thread with bounded concurrency.

    This is a thin wrapper over ``asyncio.to_thread`` that adds:
    - An ``asyncio.Semaphore`` to cap concurrent in-flight calls
    - Optional async token-bucket rate limiting

    Intended usage is for sync broker clients (requests/time.sleep) from async code.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 5,
        rate_limit_per_sec: float | None = None,
        burst_limit: int = 1,
        limiter: AsyncRateLimiter | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if rate_limit_per_sec is not None and rate_limit_per_sec <= 0:
            raise ValueError("rate_limit_per_sec must be positive when provided")

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._limiter = limiter or (
            AsyncRateLimiter(rate_limit_per_sec, burst_limit)
            if rate_limit_per_sec is not None
            else None
        )

    async def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        async with self._semaphore:
            if self._limiter is None:
                return await asyncio.to_thread(func, *args, **kwargs)
            async with self._limiter:
                return await asyncio.to_thread(func, *args, **kwargs)

    async def __call__(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await self.run(func, *args, **kwargs)


__all__ = ["BoundedToThread"]
