"""Tests for OrderEventRecorder preview and success methods."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


@pytest.fixture
def recorder_mocks(monkeypatch: pytest.MonkeyPatch) -> dict[str, MagicMock]:
    monitoring_logger = MagicMock()
    get_monitoring_logger = MagicMock(return_value=monitoring_logger)
    emit_metric = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", get_monitoring_logger)
    monkeypatch.setattr(recorder_module, "emit_metric", emit_metric)
    return {
        "monitoring_logger": monitoring_logger,
        "get_monitoring_logger": get_monitoring_logger,
        "emit_metric": emit_metric,
    }


class TestRecordPreview:
    """Tests for record_preview method."""

    def test_record_preview_skips_when_preview_is_none(
        self,
        order_event_recorder: OrderEventRecorder,
        recorder_mocks: dict[str, MagicMock],
    ) -> None:
        """Test that record_preview does nothing when preview is None."""
        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview=None,
        )

        recorder_mocks["emit_metric"].assert_not_called()
        recorder_mocks["get_monitoring_logger"].assert_not_called()

    def test_record_preview_emits_metric(
        self,
        order_event_recorder: OrderEventRecorder,
        mock_event_store: MagicMock,
        recorder_mocks: dict[str, MagicMock],
    ) -> None:
        """Test that record_preview emits metric with correct data."""
        preview = {"fee": "0.001", "estimated_fill": "50000"}

        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
            preview=preview,
        )

        emit_metric = recorder_mocks["emit_metric"]
        emit_metric.assert_called_once()
        call_args = emit_metric.call_args
        assert call_args[0][0] is mock_event_store
        assert call_args[0][1] == "test-bot-123"
        metric_data = call_args[0][2]
        assert metric_data["event_type"] == "order_preview"
        assert metric_data["symbol"] == "BTC-USD"
        assert metric_data["side"] == "BUY"
        assert metric_data["order_type"] == "LIMIT"
        assert metric_data["quantity"] == "1.5"
        assert metric_data["price"] == "50000"
        assert metric_data["preview"] == preview

    def test_record_preview_uses_market_for_none_price(
        self,
        order_event_recorder: OrderEventRecorder,
        recorder_mocks: dict[str, MagicMock],
    ) -> None:
        """Test that 'market' is used when price is None."""
        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            preview={"some": "data"},
        )

        call_args = recorder_mocks["emit_metric"].call_args
        metric_data = call_args[0][2]
        assert metric_data["price"] == "market"

    def test_record_preview_logs_event(
        self,
        order_event_recorder: OrderEventRecorder,
        recorder_mocks: dict[str, MagicMock],
    ) -> None:
        """Test that record_preview logs to monitoring logger."""
        order_event_recorder.record_preview(
            symbol="ETH-USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("2.0"),
            price=None,
            preview={"data": "value"},
        )

        monitoring_logger = recorder_mocks["monitoring_logger"]
        monitoring_logger.log_event.assert_called_once()
        call_kwargs = monitoring_logger.log_event.call_args.kwargs
        assert call_kwargs["event_type"] == "order_preview"
        assert call_kwargs["symbol"] == "ETH-USD"
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["component"] == "TradingEngine"

    def test_record_preview_handles_log_exception(
        self,
        order_event_recorder: OrderEventRecorder,
        recorder_mocks: dict[str, MagicMock],
    ) -> None:
        """Test that record_preview handles logging exceptions gracefully."""
        monitoring_logger = recorder_mocks["monitoring_logger"]
        monitoring_logger.log_event.side_effect = RuntimeError("Log failure")

        # Should not raise
        order_event_recorder.record_preview(
            symbol="BTC-USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"data": "value"},
        )
        recorder_mocks["emit_metric"].assert_called_once()
        monitoring_logger.log_event.assert_called_once()


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
        monkeypatch.setattr(recorder_module, "logger", mock_logger)

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
        monkeypatch.setattr(recorder_module, "logger", mock_logger)

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
        monkeypatch.setattr(recorder_module, "logger", mock_logger)

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
