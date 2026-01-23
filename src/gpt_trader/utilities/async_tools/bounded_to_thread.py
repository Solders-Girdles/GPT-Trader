"""Concurrency-limited helper for running sync callables via asyncio.to_thread()."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from .rate_limit import AsyncRateLimiter

T = TypeVar("T")


class BoundedToThread:
    """Run blocking callables in a thread with bounded concurrency.

    This is a thin wrapper over ``asyncio.to_thread`` that adds:
    - An ``asyncio.Semaphore`` to cap concurrent in-flight calls
    - Optional async token-bucket rate limiting
    - Optional dedicated ``ThreadPoolExecutor`` for isolation

    Intended usage is for sync broker clients (requests/time.sleep) from async code.
    """

    def __init__(
        self,
        *,
        max_concurrency: int = 5,
        rate_limit_per_sec: float | None = None,
        burst_limit: int = 1,
        limiter: AsyncRateLimiter | None = None,
        executor: ThreadPoolExecutor | None = None,
        use_dedicated_executor: bool = False,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if rate_limit_per_sec is not None and rate_limit_per_sec <= 0:
            raise ValueError("rate_limit_per_sec must be positive when provided")
        if executor is not None and use_dedicated_executor:
            raise ValueError("Provide executor or set use_dedicated_executor, not both")

        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._limiter = limiter or (
            AsyncRateLimiter(rate_limit_per_sec, burst_limit)
            if rate_limit_per_sec is not None
            else None
        )
        self._owns_executor = False
        if executor is not None:
            self._executor: ThreadPoolExecutor | None = executor
        elif use_dedicated_executor:
            self._executor = ThreadPoolExecutor(max_workers=max_concurrency)
            self._owns_executor = True
        else:
            self._executor = None

    async def _run_in_thread(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        if self._executor is None:
            return await asyncio.to_thread(func, *args, **kwargs)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(func, *args, **kwargs),
        )

    async def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        async with self._semaphore:
            if self._limiter is None:
                return await self._run_in_thread(func, *args, **kwargs)
            async with self._limiter:
                return await self._run_in_thread(func, *args, **kwargs)

    async def __call__(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await self.run(func, *args, **kwargs)

    def shutdown(self) -> None:
        """Shutdown the dedicated executor if this instance owns it."""
        if self._executor is not None and self._owns_executor:
            self._executor.shutdown(wait=True)
            self._executor = None


__all__ = ["BoundedToThread"]
