"""Tests for broker call metrics recording in BrokerExecutor."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from gpt_trader.core import (
    OrderSide,
    OrderType,
)
from gpt_trader.features.live_trade.execution.broker_executor import BrokerExecutor


class TestBrokerCallMetrics:
    """Tests for broker call metrics recording."""

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_successful_call_records_latency_with_success_outcome(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that successful broker call records latency with success outcome."""
        mock_order = MagicMock()
        mock_order.id = "order-123"
        mock_broker.place_order.return_value = mock_order

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

        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["operation"] == "submit"
        assert call_kwargs["outcome"] == "success"
        assert call_kwargs["latency_seconds"] >= 0

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_failed_call_records_latency_with_failure_outcome(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that failed broker call records latency with failure outcome."""
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

        mock_record_latency.assert_called_once()
        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["operation"] == "submit"
        assert call_kwargs["outcome"] == "failure"
        assert call_kwargs["reason"] == "error"

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_timeout_error_classified_correctly(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that timeout errors are classified correctly in metrics."""
        mock_broker.place_order.side_effect = RuntimeError("Request timeout exceeded")

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

        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["reason"] == "timeout"

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_rate_limit_error_classified_correctly(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that rate limit errors are classified correctly in metrics."""
        mock_broker.place_order.side_effect = RuntimeError("429 rate limit exceeded")

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

        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["reason"] == "rate_limit"

    @patch("gpt_trader.features.live_trade.execution.broker_executor._record_broker_call_latency")
    def test_network_error_classified_correctly(
        self,
        mock_record_latency: MagicMock,
        mock_broker: MagicMock,
    ) -> None:
        """Test that network errors are classified correctly in metrics."""
        mock_broker.place_order.side_effect = RuntimeError("Connection refused")

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

        call_kwargs = mock_record_latency.call_args[1]
        assert call_kwargs["reason"] == "network"
