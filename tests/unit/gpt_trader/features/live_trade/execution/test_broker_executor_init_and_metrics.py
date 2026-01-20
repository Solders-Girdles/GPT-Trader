"""Tests for BrokerExecutor initialization and metrics recording."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.broker_executor as broker_executor_module
from gpt_trader.core import Order, OrderSide, OrderType
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor


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


def test_init_defaults_integration_mode_false(mock_broker: MagicMock) -> None:
    executor = BrokerExecutor(broker=mock_broker)
    assert executor._integration_mode is False


def test_init_accepts_integration_mode_true(mock_broker: MagicMock) -> None:
    executor = BrokerExecutor(broker=mock_broker, integration_mode=True)
    assert executor._integration_mode is True


class TestBrokerCallMetrics:
    def test_successful_call_records_latency_with_success_outcome(
        self,
        record_latency_mock: MagicMock,
        mock_broker: MagicMock,
        sample_order: Order,
    ) -> None:
        mock_broker.place_order.return_value = sample_order
        executor = BrokerExecutor(broker=mock_broker)

        executor.execute_order(
            submit_id="client-123",
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
        )

        record_latency_mock.assert_called_once()
        call_kwargs = record_latency_mock.call_args.kwargs
        assert call_kwargs["operation"] == "submit"
        assert call_kwargs["outcome"] == "success"
        assert call_kwargs["latency_seconds"] >= 0

    def test_failed_call_records_latency_with_failure_outcome(
        self,
        record_latency_mock: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        mock_broker.place_order.side_effect = RuntimeError("Broker error")
        executor = BrokerExecutor(broker=mock_broker)

        with pytest.raises(RuntimeError):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
            )

        record_latency_mock.assert_called_once()
        call_kwargs = record_latency_mock.call_args.kwargs
        assert call_kwargs["operation"] == "submit"
        assert call_kwargs["outcome"] == "failure"
        assert call_kwargs["reason"] == "error"

    @pytest.mark.parametrize(
        ("error_message", "expected_reason"),
        [
            ("Request timeout exceeded", "timeout"),
            ("429 rate limit exceeded", "rate_limit"),
            ("Connection refused", "network"),
        ],
    )
    def test_error_reason_classification(
        self,
        record_latency_mock: MagicMock,
        mock_broker: MagicMock,
        error_message: str,
        expected_reason: str,
    ) -> None:
        mock_broker.place_order.side_effect = RuntimeError(error_message)
        executor = BrokerExecutor(broker=mock_broker)

        with pytest.raises(RuntimeError):
            executor.execute_order(
                submit_id="client-123",
                symbol="BTC-USD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1.0"),
                price=None,
                stop_price=None,
                tif=None,
                reduce_only=False,
                leverage=None,
            )

        call_kwargs = record_latency_mock.call_args.kwargs
        assert call_kwargs["reason"] == expected_reason
