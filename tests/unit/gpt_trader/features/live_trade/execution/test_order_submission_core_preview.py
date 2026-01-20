"""Core unit tests for recording order previews."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide, OrderType
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


@pytest.fixture
def emit_metric_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_emit = MagicMock()
    monkeypatch.setattr(recorder_module, "emit_metric", mock_emit)
    return mock_emit


@pytest.fixture
def monitoring_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    return mock_logger


class TestRecordPreview:
    """Tests for record_preview method."""

    def test_record_preview_with_preview_data(
        self,
        submitter: OrderSubmitter,
        emit_metric_mock: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test recording a preview with data."""
        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"estimated_fee": "0.1"},
        )

        emit_metric_mock.assert_called_once()
        call_args = emit_metric_mock.call_args[0]
        assert call_args[1] == "test-bot-123"
        assert call_args[2]["event_type"] == "order_preview"

    def test_record_preview_with_none_preview_skips(
        self,
        submitter: OrderSubmitter,
        emit_metric_mock: MagicMock,
    ) -> None:
        """Test that None preview is skipped."""
        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview=None,
        )

        emit_metric_mock.assert_not_called()

    def test_record_preview_market_price(
        self,
        submitter: OrderSubmitter,
        emit_metric_mock: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test recording a preview with market price."""
        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=None,
            preview={"estimated_fee": "0.1"},
        )

        call_args = emit_metric_mock.call_args[0]
        assert call_args[2]["price"] == "market"

    def test_record_preview_handles_logger_exception(
        self,
        submitter: OrderSubmitter,
        emit_metric_mock: MagicMock,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that logger exceptions are suppressed."""
        monitoring_logger.log_event.side_effect = RuntimeError("Log error")

        submitter.record_preview(
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            preview={"estimated_fee": "0.1"},
        )

        emit_metric_mock.assert_called_once()
        monitoring_logger.log_event.assert_called_once()
