"""Async-friendly caching utilities."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar, cast

T = TypeVar("T")


class AsyncCache:
    """Simple async-safe cache with TTL."""

    def __init__(self, ttl: float = 300.0) -> None:
        self.ttl = ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            if key not in self._cache:
                return None
            value, timestamp = self._cache[key]
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._cache[key] = (value, time.time())

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key
                for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


def async_cache(
    ttl: float = 300.0,
) -> Callable[[Callable[..., Coroutine[Any, Any, T]]], Callable[..., Coroutine[Any, Any, T]]]:
    cache = AsyncCache(ttl)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            cached = await cache.get(key)
            if cached is not None:
                return cast(T, cached)
            result = await func(*args, **kwargs)
            await cache.set(key, result)
            return result

        return wrapper

    return decorator


__all__ = ["AsyncCache", "async_cache"]
