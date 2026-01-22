"""Tests for OrdersStore persistence."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from gpt_trader.persistence.orders_store import OrdersStore, OrderStatus
from tests.unit.gpt_trader.persistence.orders_store_test_helpers import create_test_order


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
                order = create_test_order(status=OrderStatus.FILLED)
                store.save_order(order)

                deleted = store.cleanup_old_orders(days=0)

                assert deleted >= 0

    def test_context_manager(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with OrdersStore(tmpdir) as store:
                order = create_test_order()
                store.save_order(order)

            store2 = OrdersStore(tmpdir)
            store2.initialize()
            retrieved = store2.get_order(order.order_id)
            assert retrieved is not None
            store2.close()
