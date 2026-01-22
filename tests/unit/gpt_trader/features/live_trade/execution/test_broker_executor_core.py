"""Tests for BrokerExecutor core setup, metrics, and retry helpers."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.broker_executor as broker_executor_module
from gpt_trader.core import Order, OrderSide, OrderType
from gpt_trader.features.live_trade.execution.broker_executor import (
    BrokerExecutor,
    RetryPolicy,
    execute_with_retry,
)

ORDER_KWARGS = {
    "submit_id": "client-123",
    "symbol": "BTC-USD",
    "side": OrderSide.BUY,
    "order_type": OrderType.MARKET,
    "quantity": Decimal("1.0"),
    "price": None,
    "stop_price": None,
    "tif": None,
    "reduce_only": False,
    "leverage": None,
}


def _submit_market_order(executor: BrokerExecutor) -> None:
    executor.execute_order(**ORDER_KWARGS)


@pytest.fixture
def record_latency_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_record_latency = MagicMock()
    monkeypatch.setattr(
        broker_executor_module,
        "_record_broker_call_latency",
        mock_record_latency,
    )
    return mock_record_latency


def test_init_stores_broker(mock_broker: MagicMock) -> None:
    executor = BrokerExecutor(broker=mock_broker)
    assert executor._broker is mock_broker


@pytest.mark.parametrize(
    ("integration_mode", "expected"),
    [
        (None, False),
        (True, True),
    ],
)
def test_init_integration_mode(
    mock_broker: MagicMock,
    integration_mode: bool | None,
    expected: bool,
) -> None:
    if integration_mode is None:
        executor = BrokerExecutor(broker=mock_broker)
    else:
        executor = BrokerExecutor(broker=mock_broker, integration_mode=integration_mode)
    assert executor._integration_mode is expected


@pytest.mark.parametrize(
    ("side_effect", "outcome", "reason"),
    [
        (None, "success", None),
        (RuntimeError("Broker error"), "failure", "error"),
    ],
)
def test_broker_call_metrics(
    record_latency_mock: MagicMock,
    mock_broker: MagicMock,
    sample_order: Order,
    side_effect: Exception | None,
    outcome: str,
    reason: str | None,
) -> None:
    if side_effect is None:
        mock_broker.place_order.return_value = sample_order
    else:
        mock_broker.place_order.side_effect = side_effect
    executor = BrokerExecutor(broker=mock_broker)

    if side_effect is None:
        _submit_market_order(executor)
    else:
        with pytest.raises(RuntimeError):
            _submit_market_order(executor)

    call_kwargs = record_latency_mock.call_args.kwargs
    assert call_kwargs["operation"] == "submit"
    assert call_kwargs["outcome"] == outcome
    if outcome == "success":
        assert call_kwargs["latency_seconds"] >= 0
    else:
        assert call_kwargs["reason"] == reason


@pytest.mark.parametrize(
    ("error_message", "expected_reason"),
    [
        ("Request timeout exceeded", "timeout"),
        ("429 rate limit exceeded", "rate_limit"),
        ("Connection refused", "network"),
    ],
)
def test_error_reason_classification(
    record_latency_mock: MagicMock,
    mock_broker: MagicMock,
    error_message: str,
    expected_reason: str,
) -> None:
    mock_broker.place_order.side_effect = RuntimeError(error_message)
    executor = BrokerExecutor(broker=mock_broker)

    with pytest.raises(RuntimeError):
        _submit_market_order(executor)

    call_kwargs = record_latency_mock.call_args.kwargs
    assert call_kwargs["reason"] == expected_reason


def test_retry_policy_default_values() -> None:
    policy = RetryPolicy()
    assert policy.max_attempts == 3
    assert policy.base_delay == 0.5
    assert policy.max_delay == 5.0
    assert policy.timeout_seconds == 30.0
    assert policy.jitter == 0.1


@pytest.mark.parametrize(
    ("policy_kwargs", "attempt", "expected"),
    [
        ({"jitter": 0}, 1, 0.0),
        ({"base_delay": 1.0, "max_delay": 10.0, "jitter": 0}, 2, 1.0),
        ({"base_delay": 1.0, "max_delay": 10.0, "jitter": 0}, 3, 2.0),
        ({"base_delay": 1.0, "max_delay": 10.0, "jitter": 0}, 4, 4.0),
        ({"base_delay": 1.0, "max_delay": 2.0, "jitter": 0}, 10, 2.0),
    ],
)
def test_calculate_delay(policy_kwargs: dict[str, float], attempt: int, expected: float) -> None:
    policy = RetryPolicy(**policy_kwargs)
    assert policy.calculate_delay(attempt) == expected


@pytest.mark.parametrize(
    ("jitter", "expected"),
    [
        (-0.5, 0.0),
        (1.5, 1.0),
        (0.0, 0.0),
        (0.1, 0.1),
        (0.5, 0.5),
        (1.0, 1.0),
    ],
)
def test_jitter_bounds(jitter: float, expected: float) -> None:
    assert RetryPolicy(jitter=jitter).jitter == expected


def test_execute_with_retry_success_on_first_attempt() -> None:
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


def test_execute_with_retry_retries_once() -> None:
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


@pytest.mark.parametrize(
    ("side_effect", "exc_type", "match", "expected_calls", "expected_sleeps"),
    [
        (ConnectionError("always fails"), ConnectionError, "always fails", 3, 2),
        (ValueError("bad input"), ValueError, "bad input", 1, 0),
    ],
)
def test_execute_with_retry_stop_conditions(
    side_effect: Exception,
    exc_type: type[Exception],
    match: str,
    expected_calls: int,
    expected_sleeps: int,
) -> None:
    func = MagicMock(side_effect=side_effect)
    policy = RetryPolicy(max_attempts=3, jitter=0)
    sleep_calls: list[float] = []

    with pytest.raises(exc_type, match=match):
        execute_with_retry(
            func,
            policy,
            client_order_id="test-123",
            sleep_fn=sleep_calls.append,
        )

    assert func.call_count == expected_calls
    assert len(sleep_calls) == expected_sleeps
