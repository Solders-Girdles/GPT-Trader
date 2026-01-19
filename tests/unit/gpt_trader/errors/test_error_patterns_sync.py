from __future__ import annotations

import logging
from typing import Any

import pytest

from gpt_trader.errors import ExecutionError
from gpt_trader.errors.error_patterns import ErrorContext, handle_errors, safe_execute


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


class TestErrorContextExtended:
    """Extended tests for ErrorContext."""

    def test_exit_no_exception(self) -> None:
        ctx = ErrorContext(operation="test")
        result = ctx.__exit__(None, None, None)
        assert result is False

    def test_reraise_tuple_matching(self) -> None:
        with pytest.raises(ValueError):
            with ErrorContext(operation="test", reraise=(ValueError, TypeError)):
                raise ValueError("match")

    def test_reraise_tuple_non_matching(self) -> None:
        try:
            with ErrorContext(operation="test", reraise=(ValueError, TypeError)):
                raise RuntimeError("no match")
        except RuntimeError:
            pytest.fail("RuntimeError should be suppressed for non-matching reraise tuple")

    def test_log_level_below_error(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            with ErrorContext(operation="test", log_level=logging.WARNING):
                raise RuntimeError("warning level")
        assert "test" in caplog.text


class TestHandleBrokerageErrors:
    """Tests for handle_brokerage_errors."""

    def test_returns_result_on_success(self) -> None:
        from gpt_trader.errors.error_patterns import handle_brokerage_errors

        @handle_brokerage_errors("get_balance")
        def get_balance() -> float:
            return 1000.0

        assert get_balance() == 1000.0


class TestHandleOrderErrors:
    """Tests for handle_order_errors."""

    def test_returns_result_on_success(self) -> None:
        from gpt_trader.errors.error_patterns import handle_order_errors

        @handle_order_errors("place_order")
        def place_order() -> str:
            return "order-123"

        assert place_order() == "order-123"


class TestHandleDataErrors:
    """Tests for handle_data_errors."""

    def test_returns_result_on_success(self) -> None:
        from gpt_trader.errors.error_patterns import handle_data_errors

        @handle_data_errors("fetch_data")
        def fetch_data() -> list:
            return [1, 2, 3]

        assert fetch_data() == [1, 2, 3]


class TestHandleConfigErrors:
    """Tests for handle_config_errors."""

    def test_returns_result_on_success(self) -> None:
        from gpt_trader.errors.error_patterns import handle_config_errors

        @handle_config_errors("load_config")
        def load_config() -> dict:
            return {"key": "value"}

        assert load_config() == {"key": "value"}


class TestHandleAccountErrors:
    """Tests for handle_account_errors."""

    def test_returns_result_on_success(self) -> None:
        from gpt_trader.errors.error_patterns import handle_account_errors

        @handle_account_errors("get_account")
        def get_account() -> dict:
            return {"balance": 1000}

        assert get_account() == {"balance": 1000}


class TestRetryOnError:
    """Tests for retry_on_error decorator."""

    def test_succeeds_without_retry(self) -> None:
        from gpt_trader.errors.error_patterns import retry_on_error

        call_count = 0

        @retry_on_error(max_attempts=3)
        def succeed() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self) -> None:
        from gpt_trader.errors.error_patterns import retry_on_error

        call_count = 0

        @retry_on_error(max_attempts=3, delay=0.01, retry_on=RuntimeError)
        def fail_then_succeed() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("temporary failure")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 2
