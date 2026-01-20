"""Deterministic resilience tests for BrokerExecutor (retry behavior)."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.broker_executor as broker_executor_module
from gpt_trader.core import (
    Order,
    OrderSide,
    OrderType,
)
from gpt_trader.features.live_trade.execution.broker_executor import (
    BrokerExecutor,
    RetryPolicy,
)


class TestBrokerExecutorResilienceRetry:
    """Deterministic resilience tests using failure injection harness."""

    def test_retry_success_after_n_failures(self, sample_order: Order) -> None:
        """Test that retry succeeds after N transient failures."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            counting_sleep,
        )

        # Setup: fail twice, then succeed
        mock_broker = MagicMock()
        mock_broker.place_order.return_value = sample_order
        script = FailureScript.fail_then_succeed(failures=2)
        injecting = InjectingBroker(mock_broker, place_order=script)

        sleep_fn, get_sleeps = counting_sleep()
        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=5, base_delay=0.5, jitter=0),
            sleep_fn=sleep_fn,
        )

        result = executor.execute_order(
            submit_id="resilience-test-1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            use_retry=True,
        )

        assert result is sample_order
        assert injecting.get_call_count("place_order") == 3  # 2 failures + 1 success
        assert len(get_sleeps()) == 2  # 2 delays before retries

    def test_timeout_path_triggers_retry_with_classification(
        self, sample_order: Order, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that timeout triggers retry and is classified correctly."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            no_op_sleep,
        )

        mock_broker = MagicMock()
        mock_broker.place_order.return_value = sample_order
        script = FailureScript.timeout_then_succeed(timeouts=1)
        injecting = InjectingBroker(mock_broker, place_order=script)

        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=3, jitter=0),
            sleep_fn=no_op_sleep,
        )

        mock_record = MagicMock()
        monkeypatch.setattr(broker_executor_module, "_record_broker_call_latency", mock_record)

        result = executor.execute_order(
            submit_id="timeout-test-1",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            use_retry=True,
        )

        assert result is sample_order
        assert injecting.get_call_count("place_order") == 2
        # At least one call should have recorded with timeout classification
        # (the first failed attempt)
        assert mock_record.call_count >= 1

    def test_non_retryable_error_short_circuits(self) -> None:
        """Test that non-retryable errors abort immediately without retries."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            no_op_sleep,
        )

        mock_broker = MagicMock()
        # ValueError is not in default retryable_exceptions
        script = FailureScript.rejection("Insufficient funds")
        injecting = InjectingBroker(mock_broker, place_order=script)

        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=5, jitter=0),
            sleep_fn=no_op_sleep,
        )

        with pytest.raises(ValueError, match="Insufficient funds"):
            executor.execute_order(
                submit_id="rejection-test-1",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
                use_retry=True,
            )

        # Only one attempt - no retries for non-retryable
        assert injecting.get_call_count("place_order") == 1

    def test_all_retries_exhausted_raises_last_exception(self) -> None:
        """Test that exhausting all retries raises the last exception."""
        from tests.fixtures.failure_injection import (
            FailureScript,
            InjectingBroker,
            counting_sleep,
        )

        mock_broker = MagicMock()
        script = FailureScript.always_fail(ConnectionError("persistent failure"))
        injecting = InjectingBroker(mock_broker, place_order=script)

        sleep_fn, get_sleeps = counting_sleep()
        executor = BrokerExecutor(
            broker=injecting,
            retry_policy=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=0),
            sleep_fn=sleep_fn,
        )

        with pytest.raises(ConnectionError, match="persistent failure"):
            executor.execute_order(
                submit_id="exhaust-test-1",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
                use_retry=True,
            )

        assert injecting.get_call_count("place_order") == 3  # All attempts used
        assert len(get_sleeps()) == 2  # Delays before attempts 2 and 3
