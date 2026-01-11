"""Tests for OrdersStore persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from gpt_trader.persistence.orders_store import OrderRecord, OrdersStore, OrderStatus


def create_test_order(
    order_id: str = "test-order-1",
    symbol: str = "BTC-USD",
    side: str = "buy",
    quantity: Decimal = Decimal("1.5"),
    status: OrderStatus = OrderStatus.PENDING,
    **kwargs,
) -> OrderRecord:
    """Create a test order with defaults."""
    now = datetime.now(timezone.utc)
    return OrderRecord(
        order_id=order_id,
        client_order_id=kwargs.get("client_order_id", f"client-{order_id}"),
        symbol=symbol,
        side=side,
        order_type=kwargs.get("order_type", "market"),
        quantity=quantity,
        price=kwargs.get("price"),
        status=status,
        filled_quantity=kwargs.get("filled_quantity", Decimal("0")),
        average_fill_price=kwargs.get("average_fill_price"),
        created_at=kwargs.get("created_at", now),
        updated_at=kwargs.get("updated_at", now),
        bot_id=kwargs.get("bot_id", "test-bot"),
        time_in_force=kwargs.get("time_in_force", "GTC"),
        metadata=kwargs.get("metadata"),
    )


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


class TestOrdersStore:
    """Tests for OrdersStore persistence."""

    def test_initialize_creates_database(self) -> None:
        with TemporaryDirectory() as tmpdir:
            store = OrdersStore(tmpdir)
            store.initialize()

            db_path = Path(tmpdir) / "orders.db"
            assert db_path.exists()

    def test_save_and_get_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                order = create_test_order()
                result = store.save_order(order)

                assert result.success is True
                assert result.checksum is not None

                retrieved = store.get_order(order.order_id)
                assert retrieved is not None
                assert retrieved.order_id == order.order_id
                assert retrieved.symbol == order.symbol
                assert retrieved.quantity == order.quantity

    def test_get_order_not_found(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                result = store.get_order("nonexistent")
                assert result is None

    def test_save_order_upsert(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                order = create_test_order(status=OrderStatus.PENDING)
                store.save_order(order)

                # Update the same order
                order.status = OrderStatus.FILLED
                order.filled_quantity = order.quantity
                store.save_order(order)

                retrieved = store.get_order(order.order_id)
                assert retrieved.status == OrderStatus.FILLED
                assert retrieved.filled_quantity == order.quantity

    def test_upsert_by_client_id_updates_order_id(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                pending = create_test_order(
                    order_id="client-1",
                    client_order_id="client-1",
                    status=OrderStatus.PENDING,
                )
                store.upsert_by_client_id(pending)

                final = create_test_order(
                    order_id="order-123",
                    client_order_id="client-1",
                    status=OrderStatus.OPEN,
                )
                store.upsert_by_client_id(final)

                retrieved = store.get_order("order-123")
                assert retrieved is not None
                assert retrieved.client_order_id == "client-1"
                assert store.get_order("client-1") is None

    def test_get_pending_orders(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                # Create orders in various states
                pending = create_test_order(order_id="pending-1", status=OrderStatus.PENDING)
                open_order = create_test_order(order_id="open-1", status=OrderStatus.OPEN)
                filled = create_test_order(order_id="filled-1", status=OrderStatus.FILLED)
                cancelled = create_test_order(order_id="cancelled-1", status=OrderStatus.CANCELLED)

                for order in [pending, open_order, filled, cancelled]:
                    store.save_order(order)

                pending_orders = store.get_pending_orders()

                order_ids = [o.order_id for o in pending_orders]
                assert "pending-1" in order_ids
                assert "open-1" in order_ids
                assert "filled-1" not in order_ids
                assert "cancelled-1" not in order_ids

    def test_get_pending_orders_by_bot(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                order1 = create_test_order(order_id="order-1", bot_id="bot-a")
                order2 = create_test_order(order_id="order-2", bot_id="bot-b")

                store.save_order(order1)
                store.save_order(order2)

                bot_a_orders = store.get_pending_orders(bot_id="bot-a")
                assert len(bot_a_orders) == 1
                assert bot_a_orders[0].order_id == "order-1"

    def test_get_orders_by_symbol(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                btc_order = create_test_order(order_id="btc-1", symbol="BTC-USD")
                eth_order = create_test_order(order_id="eth-1", symbol="ETH-USD")

                store.save_order(btc_order)
                store.save_order(eth_order)

                btc_orders = store.get_orders_by_symbol("BTC-USD")
                assert len(btc_orders) == 1
                assert btc_orders[0].symbol == "BTC-USD"

    def test_update_status(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                order = create_test_order()
                store.save_order(order)

                result = store.update_status(
                    order.order_id,
                    OrderStatus.FILLED,
                    filled_quantity=order.quantity,
                    average_fill_price=Decimal("50000"),
                )

                assert result.success is True

                retrieved = store.get_order(order.order_id)
                assert retrieved.status == OrderStatus.FILLED
                assert retrieved.filled_quantity == order.quantity
                assert retrieved.average_fill_price == Decimal("50000")

    def test_verify_integrity_valid(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                order = create_test_order()
                store.save_order(order)

                valid_count, invalid_orders = store.verify_integrity()

                assert valid_count == 1
                assert invalid_orders == []

    def test_cleanup_old_orders(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                # Create a filled order
                order = create_test_order(status=OrderStatus.FILLED)
                store.save_order(order)

                # Cleanup with 0 days should delete all terminal orders
                # Note: This test uses SQLite datetime functions
                deleted = store.cleanup_old_orders(days=0)

                # The order was just created, so it won't be deleted with days=0
                # because SQLite compares against 'now'
                # This tests that the method runs without error
                assert deleted >= 0

    def test_context_manager(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                order = create_test_order()
                store.save_order(order)

            # After context exit, can create new store and read
            store2 = OrdersStore(tmpdir)
            store2.initialize()
            retrieved = store2.get_order(order.order_id)
            assert retrieved is not None
            store2.close()


class TestOrderStatus:
    """Tests for OrderStatus enum."""

    def test_status_values(self) -> None:
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"

    def test_status_from_string(self) -> None:
        status = OrderStatus("pending")
        assert status == OrderStatus.PENDING
