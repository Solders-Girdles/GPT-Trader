"""Tests for is_async_func and run_async_if_needed helpers."""

from __future__ import annotations

import asyncio

import pytest

from gpt_trader.utilities.async_tools import is_async_func, run_async_if_needed


class TestAsyncFunctionDetection:
    def test_is_async_func(self) -> None:
        async def async_func():
            return None

        def sync_func():
            return None

        assert is_async_func(async_func)
        assert not is_async_func(sync_func)

    def test_is_async_func_edge_cases(self) -> None:
        async def async_lambda():
            return None

        def sync_lambda() -> None:
            return None

        assert is_async_func(async_lambda)
        assert not is_async_func(sync_lambda)

        class TestClass:
            async def async_method(self):
                return None

            def sync_method(self):
                return None

        obj = TestClass()
        assert is_async_func(obj.async_method)
        assert not is_async_func(obj.sync_method)


class TestRunAsyncIfNeeded:
    @pytest.mark.asyncio
    async def test_run_async_if_needed(self) -> None:
        async def async_func(x: int) -> int:
            return x * 2

        def sync_func(x: int) -> int:
            return x * 3

        result = run_async_if_needed(async_func, 5)
        assert asyncio.iscoroutine(result)
        assert await result == 10

        result = run_async_if_needed(sync_func, 5)
        assert result == 15

    @pytest.mark.asyncio
    async def test_run_async_if_needed_edge_cases(self) -> None:
        result = run_async_if_needed(lambda: None)
        assert result is None

        async def async_complex():
            return {"key": "value", "nested": [1, 2, 3]}

        result = run_async_if_needed(async_complex)
        assert asyncio.iscoroutine(result)
        assert await result == {"key": "value", "nested": [1, 2, 3]}
