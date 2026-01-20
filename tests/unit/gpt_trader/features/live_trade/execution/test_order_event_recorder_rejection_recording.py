"""Tests for OrderEventRecorder.record_integration_rejection."""

from __future__ import annotations

from unittest.mock import MagicMock

from gpt_trader.features.live_trade.execution.order_event_recorder import OrderEventRecorder


class TestRecordIntegrationRejection:
    """Tests for record_integration_rejection method."""

    def test_record_integration_rejection_stores_event(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that integration rejection stores event."""
        order_event_recorder.record_integration_rejection(
            order=order_event_mock_order,
            symbol="BTC-USD",
            status_name="CANCELLED",
        )

        mock_event_store.store_event.assert_called_once_with(
            "order_rejected",
            {
                "order_id": "order-123",
                "symbol": "BTC-USD",
                "status": "CANCELLED",
            },
        )

    def test_record_integration_rejection_with_different_status(
        self,
        order_event_recorder: OrderEventRecorder,
        order_event_mock_order: MagicMock,
        mock_event_store: MagicMock,
    ) -> None:
        """Test that different status names are handled."""
        order_event_mock_order.id = "order-456"

        order_event_recorder.record_integration_rejection(
            order=order_event_mock_order,
            symbol="ETH-USD",
            status_name="FAILED",
        )

        mock_event_store.store_event.assert_called_once_with(
            "order_rejected",
            {
                "order_id": "order-456",
                "symbol": "ETH-USD",
                "status": "FAILED",
            },
        )
