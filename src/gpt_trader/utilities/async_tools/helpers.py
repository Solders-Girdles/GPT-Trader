"""Miscellaneous async helpers."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable, Coroutine, Iterable
from typing import Any, TypeVar, cast

T = TypeVar("T")


async def gather_with_concurrency(
    coroutines: Iterable[Awaitable[T]],
    max_concurrency: int = 10,
    *,
    return_exceptions: bool = True,
) -> list[T | BaseException]:
    pending = list(coroutines)
    if not pending:
        return []

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    semaphore = asyncio.Semaphore(max_concurrency)
    results: list[T | BaseException | None] = [None] * len(pending)
    first_exception: BaseException | None = None

    async def runner(index: int, coro: Awaitable[T]) -> None:
        nonlocal first_exception
        async with semaphore:
            try:
                results[index] = await coro
            except BaseException as exc:  # pragma: no cover - exercised in tests
                results[index] = exc
                if not return_exceptions and first_exception is None:
                    first_exception = exc

    await asyncio.gather(*(runner(idx, coro) for idx, coro in enumerate(pending)))

    if first_exception is not None:
        raise first_exception

    if not all(result is not None for result in results):
        raise RuntimeError("Not all coroutines completed - this should not happen")
    return [cast(T | BaseException, result) for result in results]


async def wait_for_first(
    coroutines: Iterable[Awaitable[T]],
    timeout: float | None = None,
) -> T:
    pending = list(coroutines)
    if not pending:
        raise ValueError("At least one coroutine must be provided")

    async def iterate() -> T:
        for coro in asyncio.as_completed(pending):
            return await coro
        raise RuntimeError("asyncio.as_completed produced no results")

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
        async_func = cast(Callable[..., Coroutine[Any, Any, T]], func)
        return async_func(*args, **kwargs)
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
