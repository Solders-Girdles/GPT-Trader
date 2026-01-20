"""Tests for OrderEventRecorder.record_success."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as order_event_recorder_module
from gpt_trader.core import OrderSide
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordSuccess:
    """Tests for record_success method."""

    def test_record_success_logs_order_info(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that successful order is logged."""
        mock_logger = MagicMock()
        monkeypatch.setattr(order_event_recorder_module, "logger", mock_logger)

        # record_success only logs, doesn't interact with event_store
        # Just verify it doesn't raise
        order_event_recorder.record_success(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            display_price=Decimal("50000"),
            reduce_only=False,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["symbol"] == "BTC-USD"
        assert call_kwargs["reduce_only"] is False

    def test_record_success_handles_reduce_only(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that reduce_only flag is handled."""
        mock_logger = MagicMock()
        monkeypatch.setattr(order_event_recorder_module, "logger", mock_logger)

        order_event_recorder.record_success(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.SELL,
            quantity=Decimal("0.5"),
            display_price=Decimal("51000"),
            reduce_only=True,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["reduce_only"] is True

    def test_record_success_handles_market_price(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that 'market' as display_price is handled."""
        mock_logger = MagicMock()
        monkeypatch.setattr(order_event_recorder_module, "logger", mock_logger)

        order_event_recorder.record_success(
            order=order_event_mock_order,
            symbol="BTC-USD",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            display_price="market",
            reduce_only=False,
        )
        mock_logger.info.assert_called_once()
        call_kwargs = mock_logger.info.call_args.kwargs
        assert call_kwargs["price"] == "market"
