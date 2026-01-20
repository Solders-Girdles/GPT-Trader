"""Order submission latency metrics tests."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
import gpt_trader.features.live_trade.execution.order_submission as submission_module
from gpt_trader.core import Order, OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


@pytest.fixture(autouse=True)
def monitoring_logger(monkeypatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    monkeypatch.setattr(recorder_module, "emit_metric", MagicMock())
    return mock_logger


@pytest.fixture()
def record_latency_mock(monkeypatch) -> MagicMock:
    mock_latency = MagicMock()
    monkeypatch.setattr(submission_module, "_record_order_submission_latency", mock_latency)
    return mock_latency


@pytest.fixture(autouse=True)
def record_metric_mock(monkeypatch) -> MagicMock:
    mock_metric = MagicMock()
    monkeypatch.setattr(submission_module, "_record_order_submission_metric", mock_metric)
    return mock_metric


class TestOrderSubmissionLatencyMetrics:
    """Tests for order submission latency metrics recording."""

    def test_successful_submission_records_latency_histogram(
        self,
        record_latency_mock: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        mock_order: Order,
    ) -> None:
        """Test that successful submission records latency histogram."""
        mock_broker.place_order.return_value = mock_order

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            order_quantity=Decimal("1.0"),
            price=None,
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        record_latency_mock.assert_called_once()
        call_kwargs = record_latency_mock.call_args[1]
        assert call_kwargs["result"] == "success"
        assert call_kwargs["side"].lower() == "buy"
        assert call_kwargs["latency_seconds"] >= 0

    def test_failed_submission_records_latency_with_failure_result(
        self,
        record_latency_mock: MagicMock,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
    ) -> None:
        """Test that failed submission records latency with failure result."""
        mock_broker.place_order.side_effect = RuntimeError("Connection error")

        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
        )

        submitter.submit_order(
            symbol="BTC-USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            order_quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            stop_price=None,
            tif=None,
            reduce_only=False,
            leverage=None,
            client_order_id=None,
        )

        record_latency_mock.assert_called_once()
        call_kwargs = record_latency_mock.call_args[1]
        assert call_kwargs["result"] == "failed"
        assert call_kwargs["side"].lower() == "sell"
