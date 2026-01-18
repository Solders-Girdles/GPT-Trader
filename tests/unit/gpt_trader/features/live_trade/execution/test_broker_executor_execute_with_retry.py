"""Tests for execute_with_retry helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.features.live_trade.execution.broker_executor import (
    RetryPolicy,
    execute_with_retry,
)


class TestExecuteWithRetry:
    """Tests for execute_with_retry function."""

    def test_success_on_first_attempt(self) -> None:
        """Test successful execution on first attempt."""
        func = MagicMock(return_value="success")
        policy = RetryPolicy(max_attempts=3)
        sleep_calls: list[float] = []

        result = execute_with_retry(
            func,
            policy,
            client_order_id="test-123",
            sleep_fn=sleep_calls.append,
        )

        assert result == "success"
        func.assert_called_once()
        assert len(sleep_calls) == 0  # No sleep on first attempt

    def test_retry_on_retryable_exception(self) -> None:
        """Test retry occurs on retryable exception."""
        func = MagicMock(side_effect=[ConnectionError("failed"), "success"])
        policy = RetryPolicy(max_attempts=3, jitter=0)
        sleep_calls: list[float] = []

        result = execute_with_retry(
            func,
            policy,
            client_order_id="test-123",
            sleep_fn=sleep_calls.append,
        )

        assert result == "success"
        assert func.call_count == 2
        assert len(sleep_calls) == 1  # One delay before retry

    def test_max_attempts_respected(self) -> None:
        """Test that max_attempts is respected."""
        func = MagicMock(side_effect=ConnectionError("always fails"))
        policy = RetryPolicy(max_attempts=3, jitter=0)
        sleep_calls: list[float] = []

        with pytest.raises(ConnectionError, match="always fails"):
            execute_with_retry(
                func,
                policy,
                client_order_id="test-123",
                sleep_fn=sleep_calls.append,
            )

        assert func.call_count == 3
        assert len(sleep_calls) == 2  # Delays before attempts 2 and 3

    def test_non_retryable_exception_not_retried(self) -> None:
        """Test that non-retryable exceptions are not retried."""
        func = MagicMock(side_effect=ValueError("bad input"))
        policy = RetryPolicy(max_attempts=3)
        sleep_calls: list[float] = []

        with pytest.raises(ValueError, match="bad input"):
            execute_with_retry(
                func,
                policy,
                client_order_id="test-123",
                sleep_fn=sleep_calls.append,
            )

        func.assert_called_once()  # No retries
        assert len(sleep_calls) == 0

    def test_timeout_error_is_retryable(self) -> None:
        """Test that TimeoutError triggers retry."""
        func = MagicMock(side_effect=[TimeoutError("timed out"), "success"])
        policy = RetryPolicy(max_attempts=3, jitter=0)
        sleep_calls: list[float] = []

        result = execute_with_retry(
            func,
            policy,
            client_order_id="test-123",
            sleep_fn=sleep_calls.append,
        )

        assert result == "success"
        assert func.call_count == 2

    @patch("gpt_trader.features.live_trade.execution.broker_executor.logger")
    def test_client_order_id_logged_consistently(self, mock_logger: MagicMock) -> None:
        """Test that the same client_order_id is used for all attempts."""
        func = MagicMock(side_effect=[ConnectionError("fail"), "success"])
        policy = RetryPolicy(max_attempts=3, jitter=0)

        result = execute_with_retry(
            func,
            policy,
            client_order_id="stable-id-123",
            operation="test_op",
            sleep_fn=lambda _: None,
        )

        assert result == "success"
        assert func.call_count == 2

        warning_kwargs = mock_logger.warning.call_args.kwargs
        assert warning_kwargs["client_order_id"] == "stable-id-123"

        for call in mock_logger.info.call_args_list:
            call_kwargs = call.kwargs
            if "client_order_id" in call_kwargs:
                assert call_kwargs["client_order_id"] == "stable-id-123"
