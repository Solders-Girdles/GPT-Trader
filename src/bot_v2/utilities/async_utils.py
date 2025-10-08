"""Utilities for managing async/sync boundaries and concurrency."""

from __future__ import annotations

import asyncio
import functools
import threading
import time
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class AsyncToSyncWrapper:
    """Wrapper to make async functions callable from sync code."""

    def __init__(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Initialize wrapper.

        Args:
            loop: Event loop to use (creates new if None)
        """
        self.loop = loop
        self._thread_local = threading.local()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self.loop:
            return self.loop

        try:
            # Try to get current loop
            loop = asyncio.get_running_loop()
            return loop
        except RuntimeError:
            # No running loop, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop

    def __call__(self, coro: Coroutine[Any, Any, T]) -> T:
        """Call async coroutine from sync code.

        Args:
            coro: Coroutine to execute

        Returns:
            Result of coroutine
        """
        loop = self._get_loop()

        try:
            # If we're in the same thread as the loop
            if asyncio.get_running_loop() == loop:
                # Run coroutine directly
                return loop.run_until_complete(coro)
            else:
                # Submit to loop from different thread
                future = asyncio.run_coroutine_threadsafe(coro, loop)
                return future.result()
        except RuntimeError:
            # No running loop, create temporary one
            return asyncio.run(coro)


def async_to_sync(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Decorator to make async function callable from sync code.

    Args:
        func: Async function to wrap

    Returns:
        Sync wrapper function
    """
    wrapper = AsyncToSyncWrapper()

    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> T:
        coro = func(*args, **kwargs)
        return wrapper(coro)

    return sync_wrapper


class SyncToAsyncWrapper:
    """Wrapper to run sync functions in async context."""

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize wrapper.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def __call__(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Call sync function from async code.

        Args:
            func: Sync function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Result of function
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, functools.partial(func, *args, **kwargs))

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)


# Global instance for sync-to-async wrapping
_sync_to_async = SyncToAsyncWrapper()


def sync_to_async(func: Callable[..., T]) -> Callable[..., Coroutine[Any, Any, T]]:
    """Decorator to make sync function callable from async code.

    Args:
        func: Sync function to wrap

    Returns:
        Async wrapper function
    """

    @functools.wraps(func)
    async def async_wrapper(*args: Any, **kwargs: Any) -> T:
        return await _sync_to_async(func, *args, **kwargs)

    return async_wrapper


class AsyncContextManager:
    """Context manager for async operations."""

    def __init__(self, name: str, timeout: float | None = None) -> None:
        """Initialize context manager.

        Args:
            name: Context name for logging
            timeout: Optional timeout in seconds
        """
        self.name = name
        self.timeout = timeout
        self.start_time: float | None = None

    async def __aenter__(self) -> AsyncContextManager:
        """Enter async context."""
        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        if self.start_time:
            duration = time.time() - self.start_time
            # Could log timing here if needed
            # For now, we just calculate it to avoid unused variable warning
            _ = duration


def async_timeout(timeout: float) -> Callable[..., Any]:
    """Decorator for adding timeout to async functions.

    Args:
        timeout: Timeout in seconds

    Returns:
        Decorated function

    Raises:
        asyncio.TimeoutError: If operation times out
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                async with asyncio.timeout(timeout):
                    return await func(*args, **kwargs)
            except TimeoutError as e:
                raise TimeoutError(f"Operation timed out after {timeout}s") from e

        return wrapper

    return decorator


class AsyncBatchProcessor:
    """Process multiple async operations in batches."""

    def __init__(self, batch_size: int = 10, delay_between_batches: float = 0.1) -> None:
        """Initialize batch processor.

        Args:
            batch_size: Number of operations per batch
            delay_between_batches: Delay between batches in seconds
        """
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches

    async def process_batch(
        self,
        operations: list[Coroutine[Any, Any, T]],
        return_exceptions: bool = False,
    ) -> list[T | Exception]:
        """Process operations in batches.

        Args:
            operations: List of coroutines to execute
            return_exceptions: Whether to return exceptions instead of raising

        Returns:
            List of results or exceptions
        """
        results = []

        for i in range(0, len(operations), self.batch_size):
            batch = operations[i : i + self.batch_size]

            batch_results = await asyncio.gather(*batch, return_exceptions=return_exceptions)
            results.extend(batch_results)

            # Add delay between batches (except for last batch)
            if i + self.batch_size < len(operations):
                await asyncio.sleep(self.delay_between_batches)

        return results


class AsyncRateLimiter:
    """Rate limiter for async operations."""

    def __init__(self, rate_limit: float, burst_limit: int = 1) -> None:
        """Initialize rate limiter.

        Args:
            rate_limit: Maximum operations per second
            burst_limit: Maximum burst size
        """
        self.rate_limit = rate_limit
        self.burst_limit = burst_limit
        self.tokens = burst_limit
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token from the rate limiter."""
        async with self._lock:
            now = time.time()
            time_passed = now - self.last_update

            # Add tokens based on time passed
            self.tokens = min(self.burst_limit, self.tokens + time_passed * self.rate_limit)
            self.last_update = now

            if self.tokens < 1:
                # Wait for token to become available
                wait_time = (1 - self.tokens) / self.rate_limit
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1

    async def __aenter__(self) -> None:
        """Enter rate limited context."""
        await self.acquire()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit rate limited context."""
        pass


def async_rate_limit(rate_limit: float, burst_limit: int = 1) -> Callable[..., Any]:
    """Decorator for rate limiting async functions.

    Args:
        rate_limit: Maximum calls per second
        burst_limit: Maximum burst size

    Returns:
        Decorated function
    """
    limiter = AsyncRateLimiter(rate_limit, burst_limit)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with limiter:
                return await func(*args, **kwargs)

        return wrapper

    return decorator


class AsyncCache:
    """Simple async-safe cache with TTL."""

    def __init__(self, ttl: float = 300.0) -> None:
        """Initialize cache.

        Args:
            ttl: Time to live in seconds
        """
        self.ttl = ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]

            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None

            return value

    async def set(self, key: str, value: Any) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            self._cache[key] = (value, time.time())

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed
        """
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


def async_cache(ttl: float = 300.0) -> Callable[..., Any]:
    """Decorator for caching async function results.

    Args:
        ttl: Time to live in seconds

    Returns:
        Decorated function
    """
    cache = AsyncCache(ttl)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache
            cached_result = await cache.get(key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(key, result)

            return result

        return wrapper

    return decorator


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
        """Initialize retry mechanism.

        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Initial delay between retries
            max_delay: Maximum delay between retries
            backoff_factor: Multiplier for delay after each attempt
            exceptions: Exception types to retry on
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions

    async def execute(
        self, coro: Callable[..., Coroutine[Any, Any, T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute coroutine with retry logic.

        Args:
            coro: Coroutine function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Result of coroutine

        Raises:
            Last exception if all attempts fail
        """
        last_exception = None
        current_delay = self.base_delay

        for attempt in range(self.max_attempts):
            try:
                return await coro(*args, **kwargs)
            except self.exceptions as exc:
                last_exception = exc

                if attempt == self.max_attempts - 1:
                    # Last attempt, re-raise
                    raise

                # Wait before retry
                await asyncio.sleep(min(current_delay, self.max_delay))
                current_delay *= self.backoff_factor

        # This should never be reached
        raise last_exception  # type: ignore[misc]


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., Any]:
    """Decorator for retrying async functions.

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for delay after each attempt
        exceptions: Exception types to retry on

    Returns:
        Decorated function
    """
    retry = AsyncRetry(max_attempts, base_delay, max_delay, backoff_factor, exceptions)

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]]
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry.execute(func, *args, **kwargs)

        return wrapper

    return decorator


# Utility functions for common async patterns


async def gather_with_concurrency(
    coroutines: list[Coroutine[Any, Any, T]],
    max_concurrency: int = 10,
) -> list[T]:
    """Gather coroutines with concurrency limit.

    Args:
        coroutines: List of coroutines to execute
        max_concurrency: Maximum concurrent executions

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def bounded_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*(bounded_coro(c) for c in coroutines))


async def wait_for_first(
    coroutines: list[Coroutine[Any, Any, T]],
    timeout: float | None = None,
) -> T:
    """Wait for first coroutine to complete.

    Args:
        coroutines: List of coroutines
        timeout: Optional timeout

    Returns:
        Result of first completed coroutine

    Raises:
        asyncio.TimeoutError: If timeout is reached
    """
    if not coroutines:
        raise ValueError("No coroutines provided")

    try:
        if timeout:
            async with asyncio.timeout(timeout):
                for coro in asyncio.as_completed(coroutines):
                    return await coro
        else:
            for coro in asyncio.as_completed(coroutines):
                return await coro
    except TimeoutError:
        raise


def is_async_func(func: Callable[..., Any]) -> bool:
    """Check if function is async.

    Args:
        func: Function to check

    Returns:
        True if function is async
    """
    return asyncio.iscoroutinefunction(func)


def run_async_if_needed(
    func: Callable[..., T], *args: Any, **kwargs: Any
) -> T | Coroutine[Any, Any, T]:
    """Run function, handling both sync and async.

    Args:
        func: Function to run
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result or coroutine if async
    """
    if is_async_func(func):
        return func(*args, **kwargs)  # type: ignore[return-value]
    else:
        return func(*args, **kwargs)


def create_task_manager() -> asyncio.TaskGroup:
    """Create a new task group for managing async tasks.

    Returns:
        AsyncIO TaskGroup instance
    """
    return asyncio.TaskGroup()


async def run_in_thread(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a synchronous function in a thread pool from async context.

    Args:
        func: Synchronous function to run
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Result of the function
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
