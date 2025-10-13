"""Wrappers for bridging async/sync boundaries."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")


class AsyncToSyncWrapper:
    """Wrapper to make async functions callable from sync code."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.loop = loop

    def __call__(self, coro: Coroutine[Any, Any, T]) -> T:
        if self.loop is not None:
            if self.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, self.loop)
                return future.result()
            return self.loop.run_until_complete(coro)

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            raise RuntimeError("AsyncToSyncWrapper cannot be used from within a running event loop")

        return asyncio.run(coro)


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    wrapper = AsyncToSyncWrapper()

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        return wrapper(func(*args, **kwargs))

    return sync_wrapper


class SyncToAsyncWrapper:
    """Wrapper to run sync functions from async code."""

    def __init__(self, max_workers: int = 4) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def __call__(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, functools.partial(func, *args, **kwargs))

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)


_sync_to_async = SyncToAsyncWrapper()


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        return await _sync_to_async(func, *args, **kwargs)

    return async_wrapper


class AsyncBatchProcessor:
    """Process multiple async operations in batches."""

    def __init__(self, batch_size: int = 10, delay_between_batches: float = 0.1) -> None:
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches

    async def process_batch(
        self,
        operations: list[Coroutine[Any, Any, T]],
        return_exceptions: bool = False,
    ) -> list[T | BaseException]:
        results: list[T | BaseException] = []
        for i in range(0, len(operations), self.batch_size):
            batch = operations[i : i + self.batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=return_exceptions)
            results.extend(batch_results)
            if i + self.batch_size < len(operations):
                await asyncio.sleep(self.delay_between_batches)
        return results


__all__ = [
    "AsyncToSyncWrapper",
    "SyncToAsyncWrapper",
    "AsyncBatchProcessor",
    "async_to_sync",
    "sync_to_async",
]
