from __future__ import annotations

import logging

import pandas as pd
import pytest

from gpt_trader.errors.error_patterns import handle_async_errors


@pytest.mark.asyncio
async def test_handle_async_errors_returns_default_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @handle_async_errors("collect", default_return=pd.DataFrame({"a": [1]}))
    async def failing_async_call() -> pd.DataFrame:
        raise RuntimeError("async boom")

    with caplog.at_level(logging.ERROR):
        result = await failing_async_call()

    assert isinstance(result, pd.DataFrame)
    assert "async boom" in caplog.text


@pytest.mark.asyncio
async def test_handle_async_errors_reraises_when_specified() -> None:
    @handle_async_errors("collect", reraise=ValueError)
    async def failing_async_call() -> int:
        raise ValueError("bad async")

    with pytest.raises(ValueError):
        await failing_async_call()


@pytest.mark.asyncio
async def test_handle_async_errors_with_extra_context(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @handle_async_errors("test_op", extra_context={"key": "value"})
    async def failing_async() -> int:
        raise RuntimeError("error")

    with caplog.at_level(logging.ERROR):
        await failing_async()

    assert "key=value" in caplog.text


@pytest.mark.asyncio
async def test_handle_async_errors_reraises_tuple() -> None:
    @handle_async_errors("test_op", reraise=(ValueError, TypeError))
    async def failing_async() -> int:
        raise TypeError("type error")

    with pytest.raises(TypeError):
        await failing_async()


@pytest.mark.asyncio
async def test_handle_async_errors_swallows_non_matching_tuple(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @handle_async_errors("test_op", reraise=(ValueError, TypeError), default_return=-1)
    async def failing_async() -> int:
        raise RuntimeError("runtime error")

    with caplog.at_level(logging.ERROR):
        result = await failing_async()

    assert result == -1


@pytest.mark.asyncio
async def test_handle_async_errors_low_log_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @handle_async_errors("test_op", log_level=logging.WARNING, default_return=-1)
    async def failing_async() -> int:
        raise RuntimeError("warning")

    with caplog.at_level(logging.WARNING):
        await failing_async()

    assert "Async error" in caplog.text
