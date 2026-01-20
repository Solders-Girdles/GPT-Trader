"""Core unit tests for recording order rejections."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter


@pytest.fixture(autouse=True)
def monitoring_logger(monkeypatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    return mock_logger


@pytest.fixture()
def emit_metric_mock(monkeypatch) -> MagicMock:
    mock_emit_metric = MagicMock()
    monkeypatch.setattr(recorder_module, "emit_metric", mock_emit_metric)
    return mock_emit_metric


class TestRecordRejection:
    """Tests for record_rejection method."""

    def test_record_rejection_logs_and_emits(
        self,
        emit_metric_mock: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that rejection is logged and metric emitted."""
        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        emit_metric_mock.assert_called_once()
        call_args = emit_metric_mock.call_args[0]
        assert call_args[2]["event_type"] == "order_rejected"
        assert call_args[2]["reason"] == "insufficient_funds"
        assert call_args[2]["reason_detail"] == "insufficient_margin"

    def test_record_rejection_includes_client_order_id(
        self,
        emit_metric_mock: MagicMock,
        submitter: OrderSubmitter,
        monitoring_logger: MagicMock,
    ) -> None:
        """Test that client_order_id is included in rejection telemetry."""
        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
            client_order_id="custom-order-123",
        )

        emit_metric_mock.assert_called_once()
        call_args = emit_metric_mock.call_args[0]
        assert call_args[2]["client_order_id"] == "custom-order-123"

        monitoring_logger.log_order_status_change.assert_called_once()
        log_call_kwargs = monitoring_logger.log_order_status_change.call_args[1]
        assert log_call_kwargs["client_order_id"] == "custom-order-123"
        assert log_call_kwargs["order_id"] == "custom-order-123"

    def test_record_rejection_without_client_order_id_uses_empty_string(
        self,
        emit_metric_mock: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test that missing client_order_id defaults to empty string."""
        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            reason="insufficient_margin",
        )

        emit_metric_mock.assert_called_once()
        call_args = emit_metric_mock.call_args[0]
        assert call_args[2]["client_order_id"] == ""

    def test_record_rejection_with_none_price(
        self,
        emit_metric_mock: MagicMock,
        submitter: OrderSubmitter,
    ) -> None:
        """Test recording rejection with None price."""
        submitter.record_rejection(
            symbol="BTC-PERP",
            side="BUY",
            quantity=Decimal("1.0"),
            price=None,
            reason="min_notional",
        )

        call_args = emit_metric_mock.call_args[0]
        assert call_args[2]["price"] == "market"
