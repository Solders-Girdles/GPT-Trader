"\"\"\"Unit tests for typed order event schema normalization.\"\"\""

from __future__ import annotations

from decimal import Decimal

import pytest

from gpt_trader.features.live_trade.execution.order_event_schema import (
    OrderEventSchemaError,
    OrderPreviewEvent,
    OrderRejectionEvent,
    OrderSubmissionAttemptEvent,
)


class TestOrderPreviewEvent:
    def test_serialize_with_valid_data(self) -> None:
        payload = OrderPreviewEvent(
            symbol=\"BTC-USD\",
            side=\"BUY\",
            order_type=\"LIMIT\",
            quantity=Decimal(\"1.25\"),
            price=Decimal(\"50000\"),
            preview={\"fee\": \"0.001\"},
        ).serialize()

        assert payload[\"event_type\"] == \"order_preview\"
        assert payload[\"symbol\"] == \"BTC-USD\"
        assert payload[\"price\"] == \"50000\"
        assert payload[\"quantity\"] == \"1.25\"
        assert payload[\"preview\"] == {\"fee\": \"0.001\"}

    def test_serialize_handles_missing_quantity(self) -> None:
        with pytest.raises(OrderEventSchemaError):
            OrderPreviewEvent(
                symbol=\"BTC-USD\",
                side=\"BUY\",
                order_type=\"LIMIT\",
                quantity=None,
                price=Decimal(\"50000\"),
                preview={\"fee\": \"0.001\"},
            ).serialize()

    def test_serialize_handles_non_mapping_preview(self) -> None:
        with pytest.raises(OrderEventSchemaError):
            OrderPreviewEvent(
                symbol=\"BTC-USD\",
                side=\"BUY\",
                order_type=\"LIMIT\",
                quantity=Decimal(\"1.0\"),
                price=Decimal(\"50000\"),
                preview=[1, 2, 3],  # not a mapping
            ).serialize()


class TestOrderRejectionEvent:
    def test_serializes_reason_and_market_price(self) -> None:
        payload = OrderRejectionEvent(
            symbol=\"ETH-USD\",
            side=\"SELL\",
            quantity=Decimal(\"2.0\"),
            price=None,
            reason=\"insufficient_funds\",
            reason_detail=\"balance=0\",
            client_order_id=\"client-abc\",
        ).serialize()

        assert payload[\"event_type\"] == \"order_rejected\"
        assert payload[\"price\"] == \"market\"
        assert payload[\"reason\"] == \"insufficient_funds\"
        assert payload[\"reason_detail\"] == \"balance=0\"

    def test_serialize_rejects_missing_reason(self) -> None:
        with pytest.raises(OrderEventSchemaError):
            OrderRejectionEvent(
                symbol=\"ETH-USD\",
                side=\"SELL\",
                quantity=Decimal(\"2.0\"),
                price=Decimal(\"3000\"),
                reason=None,
                reason_detail=None,
                client_order_id=\"client-abc\",
            ).serialize()


class TestOrderSubmissionAttemptEvent:
    def test_serializes_allowing_empty_client_id(self) -> None:
        payload = OrderSubmissionAttemptEvent(
            client_order_id=\"\",
            symbol=\"BTC-USD\",
            side=\"BUY\",
            order_type=\"MARKET\",
            quantity=Decimal(\"0.5\"),
            price=None,
        ).serialize()

        assert payload[\"event_type\"] == \"order_submission_attempt\"
        assert payload[\"price\"] == \"market\"
        assert payload[\"quantity\"] == \"0.5\"

    def test_serialize_rejects_missing_symbol(self) -> None:
        with pytest.raises(OrderEventSchemaError):
            OrderSubmissionAttemptEvent(
                client_order_id=\"client-1\",
                symbol=None,
                side=\"BUY\",
                order_type=\"MARKET\",
                quantity=Decimal(\"0.5\"),
                price=Decimal(\"1\"),
            ).serialize()
