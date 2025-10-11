"""Miscellaneous async helpers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


async def gather_with_concurrency(
    coroutines: list[Coroutine[Any, Any, T]], max_concurrency: int = 10
) -> list[T]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(bounded(c) for c in coroutines))


async def wait_for_first(
    coroutines: list[Coroutine[Any, Any, T]], timeout: float | None = None
) -> T:
    if not coroutines:
        raise ValueError("No coroutines provided")

    async def iterate():
        for coro in asyncio.as_completed(coroutines):
            return await coro

    if timeout is None:
        return await iterate()
    async with asyncio.timeout(timeout):
        return await iterate()


def is_async_func(func: Callable[..., Any]) -> bool:
    return asyncio.iscoroutinefunction(func)


def run_async_if_needed(
    func: Callable[..., T], *args: Any, **kwargs: Any
) -> T | Coroutine[Any, Any, T]:
    if is_async_func(func):
        return func(*args, **kwargs)  # type: ignore[return-value]
    return func(*args, **kwargs)


def create_task_manager() -> asyncio.TaskGroup:
    return asyncio.TaskGroup()


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))


__all__ = [
    "gather_with_concurrency",
    "wait_for_first",
    "is_async_func",
    "run_async_if_needed",
    "create_task_manager",
    "run_in_thread",
]
