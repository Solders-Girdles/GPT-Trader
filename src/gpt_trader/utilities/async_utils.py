"""Backwards-compatible shim for async utilities."""

from __future__ import annotations

from .async_tools import (
    AsyncBatchProcessor,
    AsyncCache,
    AsyncContextManager,
    AsyncRateLimiter,
    AsyncRetry,
    AsyncToSyncWrapper,
    SyncToAsyncWrapper,
    async_cache,
    async_rate_limit,
    async_retry,
    async_timeout,
    async_to_sync,
    create_task_manager,
    gather_with_concurrency,
    is_async_func,
    run_async_if_needed,
    run_in_thread,
    sync_to_async,
    wait_for_first,
)

__all__ = [
    "AsyncToSyncWrapper",
    "SyncToAsyncWrapper",
    "AsyncBatchProcessor",
    "AsyncRateLimiter",
    "AsyncContextManager",
    "AsyncCache",
    "AsyncRetry",
    "async_to_sync",
    "sync_to_async",
    "async_timeout",
    "async_rate_limit",
    "async_cache",
    "async_retry",
    "gather_with_concurrency",
    "wait_for_first",
    "is_async_func",
    "run_async_if_needed",
    "create_task_manager",
    "run_in_thread",
]
