from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pytest

from gpt_trader.errors import ExecutionError
from gpt_trader.errors.error_patterns import (
    ErrorContext,
    handle_async_errors,
    handle_errors,
    safe_execute,
)


def test_error_context_swallow_logs_extra_context(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        with ErrorContext(
            operation="store-position",
            extra_context={"symbol": "BTC-USD", "attempt": 1},
        ):
            raise RuntimeError("boom")

    assert "operation=store-position" in caplog.text
    assert "symbol=BTC-USD" in caplog.text
    assert "RuntimeError" in caplog.text


def test_error_context_reraise_for_matching_type() -> None:
    with pytest.raises(ValueError):
        with ErrorContext(operation="load", reraise=ValueError):
            raise ValueError("bad")


def test_handle_errors_returns_default_for_swallowed_exception() -> None:
    @handle_errors("fetch", default_return=42)
    def failing_call() -> int:
        raise RuntimeError("failure")

    assert failing_call() == 42


def test_handle_errors_reraises_specified_exception() -> None:
    @handle_errors("fetch", reraise=(KeyError, ValueError), default_return={"ok": False})
    def failing_call() -> dict[str, Any]:
        raise KeyError("missing")

    with pytest.raises(KeyError):
        failing_call()


def test_safe_execute_respects_reraise() -> None:
    with pytest.raises(ExecutionError):
        safe_execute(
            lambda: (_ for _ in ()).throw(ExecutionError("stop")),
            operation="order-placement",
            reraise=ExecutionError,
        )


def test_safe_execute_returns_default_when_swallowed(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        result = safe_execute(
            lambda: (_ for _ in ()).throw(RuntimeError("stop")),
            operation="order-placement",
            default_return="fallback",
        )

    assert result == "fallback"
    assert "Error in order-placement" in caplog.text


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
