"""Miscellaneous async helpers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


async def gather_with_concurrency(
    coroutines: Iterable[Awaitable[T]],
    max_concurrency: int = 10,
    *,
    return_exceptions: bool = True,
) -> list[T]:
    pending = list(coroutines)
    if not pending:
        return []

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[Any] = [None] * len(pending)
    first_exception: BaseException | None = None

    async def runner(index: int, coro: Awaitable[T]) -> None:
        nonlocal first_exception
        async with semaphore:
            try:
                results[index] = await coro
            except BaseException as exc:  # pragma: no cover - exercised in tests
                if return_exceptions:
                    results[index] = exc
                else:
                    if first_exception is None:
                        first_exception = exc

    await asyncio.gather(*(runner(idx, coro) for idx, coro in enumerate(pending)))

    if first_exception is not None:
        raise first_exception

    return results  # type: ignore[return-value]


async def wait_for_first(
    coroutines: Iterable[Awaitable[T]],
    timeout: float | None = None,
) -> T:
    pending = list(coroutines)
    if not pending:
        raise ValueError("At least one coroutine must be provided")

    async def iterate():
        for coro in asyncio.as_completed(pending):
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
