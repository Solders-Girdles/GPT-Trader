"""Tests for OrderStatus enum."""

from __future__ import annotations

from gpt_trader.persistence.orders_store import OrderStatus


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_status_values(self) -> None:
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_status_from_string(self) -> None:
        status = OrderStatus("pending")
        assert status == OrderStatus.PENDING
