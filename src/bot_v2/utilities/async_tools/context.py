"""Context helpers for async operations."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncContextManager:
    """Context manager for async operations."""

    def __init__(self, name: str, timeout: float | None = None) -> None:
        self.name = name
        self.timeout = timeout
        self.start_time: float | None = None

    async def __aenter__(self) -> AsyncContextManager:
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time is not None:
            _ = time.time() - self.start_time


def async_timeout(
    timeout: float,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    """Decorator providing a timeout for async functions."""

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                async with asyncio.timeout(timeout):
                    return await func(*args, **kwargs)
            except TimeoutError as exc:
                raise TimeoutError(f"Operation timed out after {timeout}s") from exc

        return wrapper

    return decorator


__all__ = ["AsyncContextManager", "async_timeout"]
