"""Retry helpers for async operations."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncRetry:
    """Retry mechanism for async operations."""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    async def execute(
        self, func: Callable[..., Coroutine[Any, Any, T]], *args: Any, **kwargs: Any
    ) -> T:
        last_exception: Exception | None = None
        delay = self.base_delay
        for attempt in range(self.max_attempts):
            try:
                return await func(*args, **kwargs)
            except self.exceptions as exc:
                last_exception = exc
                if attempt == self.max_attempts - 1:
                    raise
                await asyncio.sleep(min(delay, self.max_delay))
                delay *= self.backoff_factor
        if last_exception is None:
            raise RuntimeError("Retry loop completed without exception - this should not happen")
        raise last_exception


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    retry = AsyncRetry(max_attempts, base_delay, max_delay, backoff_factor, exceptions)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry.execute(func, *args, **kwargs)

        return functools.wraps(func)(wrapper)

    return decorator


__all__ = ["AsyncRetry", "async_retry"]
