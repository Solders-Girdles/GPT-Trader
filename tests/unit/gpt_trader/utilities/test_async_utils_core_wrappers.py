"""Tests for core async utility wrappers."""

from __future__ import annotations

import asyncio
import threading

import pytest

from gpt_trader.utilities.async_tools import (  # naming: allow
    AsyncToSyncWrapper,
    SyncToAsyncWrapper,
    async_to_sync,
    sync_to_async,
)


class TestAsyncToSyncWrapper:
    """Test AsyncToSyncWrapper functionality."""

    def test_async_to_sync_basic(self) -> None:
        """Test basic async to sync conversion."""

        async def async_func(x: int) -> int:
            return x * 2

        wrapper = AsyncToSyncWrapper()
        result = wrapper(async_func(5))
        assert result == 10

    def test_async_to_sync_with_exception(self) -> None:
        """Test async to sync with exception."""

        async def failing_func():
            raise ValueError("Test error")

        wrapper = AsyncToSyncWrapper()
        with pytest.raises(ValueError, match="Test error"):
            wrapper(failing_func())

    def test_async_to_sync_decorator(self) -> None:
        """Test async_to_sync decorator."""

        @async_to_sync
        async def async_func(x: int) -> int:
            return x * 3

        result = async_func(7)
        assert result == 21

    def test_async_to_sync_in_running_loop(self) -> None:
        async def async_func(x: int) -> int:
            await asyncio.sleep(0)
            return x * 2

        loop = asyncio.new_event_loop()
        loop_started = threading.Event()

        def run_loop() -> None:
            asyncio.set_event_loop(loop)
            loop_started.set()
            loop.run_forever()

        runner = threading.Thread(target=run_loop)
        runner.start()
        loop_started.wait()

        try:
            wrapper = AsyncToSyncWrapper(loop)
            result = wrapper(async_func(5))
            assert result == 10
        finally:
            loop.call_soon_threadsafe(loop.stop)
            runner.join()
            loop.close()


class TestSyncToAsyncWrapper:
    """Test SyncToAsyncWrapper functionality."""

    @pytest.mark.asyncio
    async def test_sync_to_async_basic(self) -> None:
        """Test basic sync to async conversion."""

        def sync_func(x: int) -> int:
            return x * 2

        wrapper = SyncToAsyncWrapper()
        result = await wrapper(sync_func, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_sync_to_async_with_exception(self) -> None:
        """Test sync to async with exception."""

        def failing_func():
            raise ValueError("Test error")

        wrapper = SyncToAsyncWrapper()
        with pytest.raises(ValueError, match="Test error"):
            await wrapper(failing_func)

    @pytest.mark.asyncio
    async def test_sync_to_async_decorator(self) -> None:
        """Test sync_to_async decorator."""

        @sync_to_async
        def sync_func(x: int) -> int:
            return x * 3

        result = await sync_func(7)
        assert result == 21
