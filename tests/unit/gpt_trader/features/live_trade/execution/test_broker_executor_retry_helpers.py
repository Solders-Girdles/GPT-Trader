"""Tests for BrokerExecutor retry primitives (RetryPolicy + execute_with_retry)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.broker_executor as broker_executor_module
from gpt_trader.features.live_trade.execution.broker_executor import (
    RetryPolicy,
    execute_with_retry,
)


class TestRetryPolicy:
    def test_default_values(self) -> None:
        policy = RetryPolicy()
        assert policy.max_attempts == 3
        assert policy.base_delay == 0.5
        assert policy.max_delay == 5.0
        assert policy.timeout_seconds == 30.0
        assert policy.jitter == 0.1

    def test_calculate_delay_first_attempt_is_zero(self) -> None:
        policy = RetryPolicy(jitter=0)
        assert policy.calculate_delay(1) == 0.0

    def test_calculate_delay_exponential_backoff(self) -> None:
        policy = RetryPolicy(base_delay=1.0, max_delay=10.0, jitter=0)
        assert policy.calculate_delay(2) == 1.0
        assert policy.calculate_delay(3) == 2.0
        assert policy.calculate_delay(4) == 4.0

    def test_calculate_delay_respects_max_delay(self) -> None:
        policy = RetryPolicy(base_delay=1.0, max_delay=2.0, jitter=0)
        assert policy.calculate_delay(10) == 2.0

    def test_jitter_clamping(self) -> None:
        assert RetryPolicy(jitter=-0.5).jitter == 0.0
        assert RetryPolicy(jitter=1.5).jitter == 1.0

    def test_jitter_valid_range_unchanged(self) -> None:
        for jitter_val in [0.0, 0.1, 0.5, 1.0]:
            assert RetryPolicy(jitter=jitter_val).jitter == jitter_val


class TestExecuteWithRetry:
    def test_success_on_first_attempt(self) -> None:
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
        assert len(sleep_calls) == 0

    def test_retry_on_retryable_exception(self) -> None:
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
        assert len(sleep_calls) == 1

    def test_max_attempts_respected(self) -> None:
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
        assert len(sleep_calls) == 2

    def test_non_retryable_exception_not_retried(self) -> None:
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

        func.assert_called_once()
        assert len(sleep_calls) == 0

    def test_timeout_error_is_retryable(self) -> None:
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

    def test_client_order_id_logged_consistently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr(broker_executor_module, "logger", mock_logger)

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

        assert mock_logger.warning.call_args.kwargs["client_order_id"] == "stable-id-123"
        for call in mock_logger.info.call_args_list:
            if "client_order_id" in call.kwargs:
                assert call.kwargs["client_order_id"] == "stable-id-123"
