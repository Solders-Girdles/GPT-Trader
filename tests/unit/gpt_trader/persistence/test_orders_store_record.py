"""Tests for OrderRecord persistence."""

from __future__ import annotations

from decimal import Decimal

from gpt_trader.persistence.orders_store import OrderRecord, OrderStatus
from tests.unit.gpt_trader.persistence.orders_store_test_helpers import create_test_order


class TestOrderRecord:
    """Tests for OrderRecord dataclass."""

    def test_to_dict_and_from_dict(self) -> None:
        order = create_test_order(
            price=Decimal("50000"),
            metadata={"source": "test"},
        )

        order_dict = order.to_dict()
        restored = OrderRecord.from_dict(order_dict)

        assert restored.order_id == order.order_id
        assert restored.symbol == order.symbol
        assert restored.quantity == order.quantity
        assert restored.price == order.price
        assert restored.metadata == order.metadata

    def test_compute_checksum_deterministic(self) -> None:
        order = create_test_order()
        checksum1 = order.compute_checksum()
        checksum2 = order.compute_checksum()
        assert checksum1 == checksum2

    def test_compute_checksum_changes_with_critical_fields(self) -> None:
        order1 = create_test_order(quantity=Decimal("1.0"))
        order2 = create_test_order(quantity=Decimal("2.0"))
        assert order1.compute_checksum() != order2.compute_checksum()

    def test_is_terminal_pending(self) -> None:
        order = create_test_order(status=OrderStatus.PENDING)
        assert order.is_terminal() is False

    def test_is_terminal_open(self) -> None:
        order = create_test_order(status=OrderStatus.OPEN)
        assert order.is_terminal() is False

    def test_is_terminal_filled(self) -> None:
        order = create_test_order(status=OrderStatus.FILLED)
        assert order.is_terminal() is True

    def test_is_terminal_cancelled(self) -> None:
        order = create_test_order(status=OrderStatus.CANCELLED)
        assert order.is_terminal() is True
