"""Deterministic resilience tests for BrokerExecutor retry/backoff behavior."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.broker_executor as broker_executor_module
from gpt_trader.core import Order, OrderSide, OrderType
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor, RetryPolicy
from tests.fixtures.failure_injection import (
    FailureScript,
    InjectingBroker,
    counting_sleep,
    no_op_sleep,
)


def _execute_market_buy(executor: BrokerExecutor, submit_id: str) -> Order:
    return executor.execute_order(
        submit_id=submit_id,
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


def _make_executor(
    script: FailureScript,
    *,
    sample_order: Order | None = None,
    retry_policy: RetryPolicy,
    sleep_fn,
) -> tuple[BrokerExecutor, InjectingBroker]:
    mock_broker = MagicMock()
    if sample_order is not None:
        mock_broker.place_order.return_value = sample_order
    injecting = InjectingBroker(mock_broker, place_order=script)
    return (
        BrokerExecutor(broker=injecting, retry_policy=retry_policy, sleep_fn=sleep_fn),
        injecting,
    )


def test_retry_success_after_n_failures(sample_order: Order) -> None:
    script = FailureScript.fail_then_succeed(failures=2)
    sleep_fn, get_sleeps = counting_sleep()
    executor, injecting = _make_executor(
        script,
        sample_order=sample_order,
        retry_policy=RetryPolicy(max_attempts=5, base_delay=0.5, jitter=0),
        sleep_fn=sleep_fn,
    )

    result = _execute_market_buy(executor, submit_id="resilience-test-1")

    assert result is sample_order
    assert injecting.get_call_count("place_order") == 3
    assert len(get_sleeps()) == 2


def test_timeout_path_triggers_retry_with_classification(
    sample_order: Order, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = FailureScript.timeout_then_succeed(timeouts=1)
    executor, injecting = _make_executor(
        script,
        sample_order=sample_order,
        retry_policy=RetryPolicy(max_attempts=3, jitter=0),
        sleep_fn=no_op_sleep,
    )

    mock_record = MagicMock()
    monkeypatch.setattr(broker_executor_module, "_record_broker_call_latency", mock_record)

    result = _execute_market_buy(executor, submit_id="timeout-test-1")

    assert result is sample_order
    assert injecting.get_call_count("place_order") == 2
    assert mock_record.call_count >= 1


def test_non_retryable_error_short_circuits() -> None:
    script = FailureScript.rejection("Insufficient funds")
    executor, injecting = _make_executor(
        script,
        retry_policy=RetryPolicy(max_attempts=5, jitter=0),
        sleep_fn=no_op_sleep,
    )

    with pytest.raises(ValueError, match="Insufficient funds"):
        _execute_market_buy(executor, submit_id="rejection-test-1")

    assert injecting.get_call_count("place_order") == 1


def test_all_retries_exhausted_raises_last_exception() -> None:
    script = FailureScript.always_fail(ConnectionError("persistent failure"))
    sleep_fn, get_sleeps = counting_sleep()
    executor, injecting = _make_executor(
        script,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.1, jitter=0),
        sleep_fn=sleep_fn,
    )

    with pytest.raises(ConnectionError, match="persistent failure"):
        _execute_market_buy(executor, submit_id="exhaust-test-1")

    assert injecting.get_call_count("place_order") == 3
    assert len(get_sleeps()) == 2


def test_exponential_backoff_delays(sample_order: Order) -> None:
    script = FailureScript.fail_then_succeed(failures=3)
    sleep_fn, get_sleeps = counting_sleep()
    executor, _ = _make_executor(
        script,
        sample_order=sample_order,
        retry_policy=RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=10.0, jitter=0),
        sleep_fn=sleep_fn,
    )

    _execute_market_buy(executor, submit_id="backoff-test-1")

    assert get_sleeps() == [0.5, 1.0, 2.0]


def test_non_retryable_mid_sequence_stops_immediately() -> None:
    script = FailureScript.from_outcomes(
        ConnectionError("transient 1"),
        ConnectionError("transient 2"),
        ValueError("insufficient funds"),
        None,
    )
    sleep_fn, get_sleeps = counting_sleep()
    executor, injecting = _make_executor(
        script,
        retry_policy=RetryPolicy(max_attempts=5, base_delay=0.1, jitter=0),
        sleep_fn=sleep_fn,
    )

    with pytest.raises(ValueError, match="insufficient funds"):
        _execute_market_buy(executor, submit_id="mid-stop-test-1")

    assert injecting.get_call_count("place_order") == 3
    assert len(get_sleeps()) == 2
