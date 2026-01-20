"""Core unit tests for OrderSubmitter rejection processing and recording."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

import pytest

import gpt_trader.features.live_trade.execution.order_event_recorder as recorder_module
from gpt_trader.core import OrderSide, OrderType, TimeInForce
from gpt_trader.features.live_trade.execution.order_submission import OrderSubmitter
from gpt_trader.persistence.orders_store import OrderStatus as StoreOrderStatus


@pytest.fixture
def monitoring_logger(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_logger = MagicMock()
    monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)
    return mock_logger


@pytest.fixture
def emit_metric_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_emit = MagicMock()
    monkeypatch.setattr(recorder_module, "emit_metric", mock_emit)
    return mock_emit


class TestProcessRejection:
    """Tests for _process_rejection method."""

    def test_integration_mode_stores_event(
        self,
        mock_broker: MagicMock,
        mock_event_store: MagicMock,
        open_orders: list[str],
        rejected_order: MagicMock,
    ) -> None:
        """Test that integration mode stores rejection event."""
        submitter = OrderSubmitter(
            broker=mock_broker,
            event_store=mock_event_store,
            bot_id="test-bot",
            open_orders=open_orders,
            integration_mode=True,
        )

        rejected_order.status = MagicMock()
        rejected_order.status.value = "REJECTED"

        result = submitter._process_rejection(
            order=rejected_order,
            status_name="REJECTED",
            symbol="BTC-PERP",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            effective_price=Decimal("50000"),
            tif=TimeInForce.GTC,
            reduce_only=False,
            leverage=None,
            submit_id="test-id",
            store_status=StoreOrderStatus.REJECTED,
        )

        assert result is rejected_order
        mock_event_store.store_event.assert_called_once_with(
            "order_rejected",
            {
                "order_id": "rejected-order",
                "symbol": "BTC-PERP",
                "status": "REJECTED",
            },
        )

    def test_normal_mode_raises_error(
        self,
        submitter: OrderSubmitter,
        rejected_order: MagicMock,
        monitoring_logger: MagicMock,
        emit_metric_mock: MagicMock,
    ) -> None:
        """Test that normal mode raises RuntimeError."""
        rejected_order.status = MagicMock()
        rejected_order.status.value = "CANCELLED"

        with pytest.raises(RuntimeError, match="CANCELLED"):
            submitter._process_rejection(
                order=rejected_order,
                status_name="CANCELLED",
                symbol="BTC-PERP",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=None,
                effective_price=Decimal("50000"),
                tif=TimeInForce.GTC,
                reduce_only=False,
                leverage=None,
                submit_id="test-id",
                store_status=StoreOrderStatus.CANCELLED,
            )


class TestRecordRejection:
    """Tests for record_rejection method."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_logger = MagicMock()
        monkeypatch.setattr(recorder_module, "get_monitoring_logger", lambda: mock_logger)

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
