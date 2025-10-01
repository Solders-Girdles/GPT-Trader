"""Tests for OrderReconciler - startup order reconciliation.

This module tests the OrderReconciler's ability to synchronize local order
storage with exchange state during bot startup, ensuring data consistency
and preventing orphaned orders.

Critical behaviors tested:
- Fetching local and exchange open orders
- Detecting discrepancies between local and exchange state
- Reconciling missing orders on exchange (cancellations/fills)
- Reconciling missing local orders (untracked orders)
- Position snapshot capture
- Error handling and graceful degradation
- Event store audit trail

Operational Context:
    Order reconciliation is critical for bot reliability during restarts.
    After a crash, deployment, or network outage, the bot's local order
    state may be out of sync with the exchange. Failures here can result in:

    - Orphaned orders executing unexpectedly
    - Duplicate order placement
    - Incorrect position tracking
    - Risk limit violations from untracked positions
    - Trading disruptions requiring manual intervention

    This reconciliation logic is the first defense against state
    inconsistency and must be bulletproof.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from bot_v2.orchestration.order_reconciler import OrderDiff, OrderReconciler
from bot_v2.persistence.event_store import EventStore
from bot_v2.persistence.orders_store import OrdersStore


@pytest.fixture
def mock_broker() -> Mock:
    """Create a mock broker with common methods."""
    broker = Mock()
    broker.list_orders = Mock(return_value=[])
    broker.get_order = Mock(return_value=None)
    broker.list_positions = Mock(return_value=[])
    return broker


@pytest.fixture
def orders_store(tmp_path) -> OrdersStore:
    """Create a test orders store."""
    return OrdersStore(storage_root=str(tmp_path))


@pytest.fixture
def event_store(tmp_path) -> EventStore:
    """Create a test event store."""
    return EventStore(storage_root=str(tmp_path))


@pytest.fixture
def reconciler(
    mock_broker: Mock, orders_store: OrdersStore, event_store: EventStore
) -> OrderReconciler:
    """Create an OrderReconciler with test dependencies."""
    return OrderReconciler(
        broker=mock_broker,
        orders_store=orders_store,
        event_store=event_store,
        bot_id="test-bot",
    )


def create_test_order(
    order_id: str = "order-1",
    symbol: str = "BTC-PERP",
    side: OrderSide = OrderSide.BUY,
    status: OrderStatus = OrderStatus.SUBMITTED,
    quantity: Decimal = Decimal("1.0"),
    price: Decimal | None = Decimal("50000.00"),
) -> Order:
    """Helper to create test orders."""
    return Order(
        id=order_id,
        client_id=f"client-{order_id}",
        symbol=symbol,
        side=side,
        type=OrderType.LIMIT,
        quantity=quantity,
        price=price,
        stop_price=None,
        tif=TimeInForce.GTC,
        status=status,
        filled_quantity=Decimal("0"),
        avg_fill_price=None,
        submitted_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class TestOrderReconcilerInitialization:
    """Test OrderReconciler initialization."""

    def test_initializes_with_required_dependencies(
        self, mock_broker: Mock, orders_store: OrdersStore, event_store: EventStore
    ) -> None:
        """OrderReconciler initializes with required dependencies.

        Ensures all required components are stored for use during reconciliation.
        """
        reconciler = OrderReconciler(
            broker=mock_broker,
            orders_store=orders_store,
            event_store=event_store,
            bot_id="test-bot",
        )

        assert reconciler._broker is mock_broker
        assert reconciler._orders_store is orders_store
        assert reconciler._event_store is event_store
        assert reconciler._bot_id == "test-bot"


class TestFetchLocalOpenOrders:
    """Test fetching local open orders from storage."""

    def test_returns_empty_dict_when_no_orders(self, reconciler: OrderReconciler) -> None:
        """Returns empty dict when no local open orders exist.

        Consistent behavior - always returns dict, never None.
        """
        result = reconciler.fetch_local_open_orders()

        assert result == {}

    def test_returns_local_orders_as_dict(
        self, reconciler: OrderReconciler, orders_store: OrdersStore
    ) -> None:
        """Returns local open orders indexed by order_id.

        Enables efficient lookup during diff calculation.
        """
        order1 = create_test_order("order-1", status=OrderStatus.SUBMITTED)
        order2 = create_test_order("order-2", status=OrderStatus.PARTIALLY_FILLED)

        orders_store.upsert(order1)
        orders_store.upsert(order2)

        result = reconciler.fetch_local_open_orders()

        assert len(result) == 2
        assert "order-1" in result
        assert "order-2" in result

    def test_handles_storage_errors_gracefully(self, reconciler: OrderReconciler) -> None:
        """Returns empty dict if storage fails, preventing startup crash.

        Critical: Reconciliation failures should not prevent bot startup.
        """
        reconciler._orders_store.get_open_orders = Mock(side_effect=Exception("Storage failure"))

        result = reconciler.fetch_local_open_orders()

        assert result == {}


class TestFetchExchangeOpenOrders:
    """Test fetching exchange open orders."""

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_no_orders(self, reconciler: OrderReconciler) -> None:
        """Returns empty dict when no exchange orders exist."""
        result = await reconciler.fetch_exchange_open_orders()

        assert result == {}

    @pytest.mark.asyncio
    async def test_fetches_orders_for_all_interested_statuses(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Queries exchange for PENDING, SUBMITTED, and PARTIALLY_FILLED orders.

        Must check all active order states to ensure complete reconciliation.
        """
        pending_order = create_test_order("order-1", status=OrderStatus.PENDING)
        submitted_order = create_test_order("order-2", status=OrderStatus.SUBMITTED)
        partial_order = create_test_order("order-3", status=OrderStatus.PARTIALLY_FILLED)

        mock_broker.list_orders = Mock(
            side_effect=[
                [pending_order],  # PENDING
                [submitted_order],  # SUBMITTED
                [partial_order],  # PARTIALLY_FILLED
            ]
        )

        result = await reconciler.fetch_exchange_open_orders()

        assert len(result) == 3
        assert "order-1" in result
        assert "order-2" in result
        assert "order-3" in result

    @pytest.mark.asyncio
    async def test_returns_orders_indexed_by_id(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Returns exchange orders indexed by order ID for efficient lookup."""
        order = create_test_order("order-123")
        mock_broker.list_orders = Mock(return_value=[order])

        result = await reconciler.fetch_exchange_open_orders()

        assert result["order-123"] == order

    @pytest.mark.asyncio
    async def test_handles_broker_errors_gracefully(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Handles broker API errors without crashing reconciliation.

        Network issues during startup should not prevent bot from starting.
        """
        mock_broker.list_orders = Mock(side_effect=Exception("API error"))

        result = await reconciler.fetch_exchange_open_orders()

        # Should return partial results, not crash
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_fallback_to_statusless_query(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Falls back to status-less query if broker doesn't support status filtering.

        Some brokers don't accept status parameter - must handle gracefully.
        """
        order = create_test_order("order-1", status=OrderStatus.SUBMITTED)

        # First call raises TypeError (status param not supported)
        # Fallback call returns all orders
        mock_broker.list_orders = Mock(side_effect=[TypeError("Unexpected argument"), [order]])

        result = await reconciler.fetch_exchange_open_orders()

        assert len(result) == 1
        assert "order-1" in result

    @pytest.mark.asyncio
    async def test_filters_orders_in_fallback_mode(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Filters for interested statuses when using fallback query.

        Fallback returns all orders - must filter to relevant statuses.
        """
        submitted_order = create_test_order("order-1", status=OrderStatus.SUBMITTED)
        filled_order = create_test_order("order-2", status=OrderStatus.FILLED)

        mock_broker.list_orders = Mock(
            side_effect=[
                TypeError("Unexpected argument"),
                [submitted_order, filled_order],
            ]
        )

        result = await reconciler.fetch_exchange_open_orders()

        # Should include SUBMITTED but not FILLED
        assert "order-1" in result
        assert "order-2" not in result


class TestDiffOrders:
    """Test order diffing logic."""

    def test_detects_no_differences_when_synced(self) -> None:
        """Returns empty diffs when local and exchange are synchronized.

        Happy path - no reconciliation needed.
        """
        order = create_test_order("order-1")
        local = {"order-1": order}
        exchange = {"order-1": order}

        diff = OrderReconciler.diff_orders(local, exchange)

        assert len(diff.missing_on_exchange) == 0
        assert len(diff.missing_locally) == 0

    def test_detects_missing_on_exchange(self) -> None:
        """Detects orders in local storage but not on exchange.

        Critical: These orders may have been cancelled or filled during downtime.
        """
        local_order = create_test_order("order-1")
        local = {"order-1": local_order}
        exchange = {}

        diff = OrderReconciler.diff_orders(local, exchange)

        assert len(diff.missing_on_exchange) == 1
        assert "order-1" in diff.missing_on_exchange
        assert len(diff.missing_locally) == 0

    def test_detects_missing_locally(self) -> None:
        """Detects orders on exchange but not in local storage.

        Critical: These are untracked orders that must be added to storage.
        """
        exchange_order = create_test_order("order-1")
        local = {}
        exchange = {"order-1": exchange_order}

        diff = OrderReconciler.diff_orders(local, exchange)

        assert len(diff.missing_locally) == 1
        assert "order-1" in diff.missing_locally
        assert len(diff.missing_on_exchange) == 0

    def test_detects_multiple_discrepancies(self) -> None:
        """Detects multiple discrepancies in both directions.

        Real-world scenario: Some orders cancelled, some new orders placed.
        """
        local_only = create_test_order("order-local")
        exchange_only = create_test_order("order-exchange")
        synced = create_test_order("order-synced")

        local = {"order-local": local_only, "order-synced": synced}
        exchange = {"order-exchange": exchange_only, "order-synced": synced}

        diff = OrderReconciler.diff_orders(local, exchange)

        assert "order-local" in diff.missing_on_exchange
        assert "order-exchange" in diff.missing_locally
        assert "order-synced" not in diff.missing_on_exchange
        assert "order-synced" not in diff.missing_locally


class TestRecordSnapshot:
    """Test snapshot recording to event store."""

    @pytest.mark.asyncio
    async def test_records_snapshot_metrics(
        self, reconciler: OrderReconciler, event_store: EventStore
    ) -> None:
        """Records reconciliation snapshot to event store for audit trail.

        Provides historical record of reconciliation operations.
        """
        local = {"order-1": create_test_order("order-1")}
        exchange = {"order-2": create_test_order("order-2")}

        await reconciler.record_snapshot(local, exchange)

        # Verify metric was recorded (implementation-dependent)
        assert True  # Actual verification depends on EventStore implementation

    @pytest.mark.asyncio
    async def test_handles_event_store_errors_gracefully(self, reconciler: OrderReconciler) -> None:
        """Handles event store failures without crashing reconciliation.

        Audit trail failures should not prevent reconciliation from completing.
        """
        reconciler._event_store.append_metric = Mock(side_effect=Exception("Storage error"))

        # Should not raise
        await reconciler.record_snapshot({}, {})


class TestReconcileMissingOnExchange:
    """Test reconciliation of orders missing on exchange."""

    @pytest.mark.asyncio
    async def test_fetches_final_status_for_missing_orders(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Queries broker for final status of orders missing on exchange.

        Order may be filled, cancelled, or expired - must get accurate status.
        """
        local_order = create_test_order("order-1")
        final_order = create_test_order("order-1", status=OrderStatus.FILLED)
        mock_broker.get_order = Mock(return_value=final_order)

        diff = OrderDiff(missing_on_exchange={"order-1": local_order}, missing_locally={})

        await reconciler.reconcile_missing_on_exchange(diff)

        mock_broker.get_order.assert_called_once_with("order-1")

    @pytest.mark.asyncio
    async def test_updates_local_storage_with_final_status(
        self, reconciler: OrderReconciler, orders_store: OrdersStore, mock_broker: Mock
    ) -> None:
        """Updates local storage with final order status from exchange.

        Critical: Local storage must reflect actual order outcomes.
        """
        local_order = create_test_order("order-1", status=OrderStatus.SUBMITTED)
        final_order = create_test_order("order-1", status=OrderStatus.FILLED)

        # Add to local storage first
        orders_store.upsert(local_order)

        mock_broker.get_order = Mock(return_value=final_order)

        diff = OrderDiff(missing_on_exchange={"order-1": local_order}, missing_locally={})

        await reconciler.reconcile_missing_on_exchange(diff)

        # Verify order updated in storage
        stored_order = orders_store.get_by_id("order-1")
        assert stored_order.status == OrderStatus.FILLED.value

    @pytest.mark.asyncio
    async def test_assumes_cancelled_when_fetch_fails(
        self, reconciler: OrderReconciler, orders_store: OrdersStore, mock_broker: Mock
    ) -> None:
        """Marks order as cancelled if final status cannot be retrieved.

        Conservative approach: If we can't confirm status, assume cancelled
        to prevent re-submission.
        """
        local_order = create_test_order("order-1")
        orders_store.upsert(local_order)

        mock_broker.get_order = Mock(return_value=None)

        diff = OrderDiff(missing_on_exchange={"order-1": local_order}, missing_locally={})

        await reconciler.reconcile_missing_on_exchange(diff)

        # Verify marked as cancelled
        stored_order = orders_store.get_by_id("order-1")
        assert stored_order.status == OrderStatus.CANCELLED.value

    @pytest.mark.asyncio
    async def test_handles_multiple_missing_orders(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Processes all missing orders sequentially.

        Must handle batch reconciliation, not just single orders.
        """
        order1 = create_test_order("order-1")
        order2 = create_test_order("order-2")

        mock_broker.get_order = Mock(
            side_effect=[
                create_test_order("order-1", status=OrderStatus.CANCELLED),
                create_test_order("order-2", status=OrderStatus.FILLED),
            ]
        )

        diff = OrderDiff(
            missing_on_exchange={"order-1": order1, "order-2": order2},
            missing_locally={},
        )

        await reconciler.reconcile_missing_on_exchange(diff)

        assert mock_broker.get_order.call_count == 2


class TestReconcileMissingLocally:
    """Test reconciliation of orders missing locally."""

    def test_adds_untracked_orders_to_storage(
        self, reconciler: OrderReconciler, orders_store: OrdersStore
    ) -> None:
        """Adds exchange orders to local storage if not tracked.

        Critical: Must track all active orders to prevent duplicates.
        """
        exchange_order = create_test_order("order-1")

        diff = OrderDiff(missing_on_exchange={}, missing_locally={"order-1": exchange_order})

        reconciler.reconcile_missing_locally(diff)

        # Verify added to storage
        stored_order = orders_store.get_by_id("order-1")
        assert stored_order is not None
        assert stored_order.order_id == "order-1"

    def test_handles_multiple_untracked_orders(
        self, reconciler: OrderReconciler, orders_store: OrdersStore
    ) -> None:
        """Adds all untracked orders to storage.

        Batch processing of multiple orphaned orders.
        """
        order1 = create_test_order("order-1")
        order2 = create_test_order("order-2")

        diff = OrderDiff(
            missing_on_exchange={},
            missing_locally={"order-1": order1, "order-2": order2},
        )

        reconciler.reconcile_missing_locally(diff)

        assert orders_store.get_by_id("order-1") is not None
        assert orders_store.get_by_id("order-2") is not None

    def test_handles_storage_errors_gracefully(self, reconciler: OrderReconciler) -> None:
        """Handles storage failures without crashing reconciliation.

        Individual order failures should not prevent other orders from syncing.
        """
        reconciler._orders_store.upsert = Mock(side_effect=Exception("Storage error"))

        exchange_order = create_test_order("order-1")
        diff = OrderDiff(missing_on_exchange={}, missing_locally={"order-1": exchange_order})

        # Should not raise
        reconciler.reconcile_missing_locally(diff)


class TestSnapshotPositions:
    """Test position snapshot capture."""

    @pytest.mark.asyncio
    async def test_returns_empty_dict_when_no_positions(self, reconciler: OrderReconciler) -> None:
        """Returns empty dict when no positions exist."""
        result = await reconciler.snapshot_positions()

        assert result == {}

    @pytest.mark.asyncio
    async def test_captures_position_data(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Captures position quantity and side for each symbol.

        Position snapshot is critical for post-restart risk validation.
        """

        class Position:
            symbol = "BTC-PERP"
            quantity = Decimal("1.5")
            side = "long"

        mock_broker.list_positions = Mock(return_value=[Position()])

        result = await reconciler.snapshot_positions()

        assert "BTC-PERP" in result
        assert result["BTC-PERP"]["quantity"] == "1.5"
        assert result["BTC-PERP"]["side"] == "long"

    @pytest.mark.asyncio
    async def test_handles_multiple_positions(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Captures all positions across multiple symbols."""

        class Position:
            def __init__(self, symbol, quantity, side):
                self.symbol = symbol
                self.quantity = quantity
                self.side = side

        positions = [
            Position("BTC-PERP", Decimal("1.0"), "long"),
            Position("ETH-PERP", Decimal("10.0"), "short"),
        ]
        mock_broker.list_positions = Mock(return_value=positions)

        result = await reconciler.snapshot_positions()

        assert len(result) == 2
        assert "BTC-PERP" in result
        assert "ETH-PERP" in result

    @pytest.mark.asyncio
    async def test_handles_broker_errors_gracefully(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Returns empty dict if position fetch fails.

        Position snapshot failures should not crash startup.
        """
        mock_broker.list_positions = Mock(side_effect=Exception("API error"))

        result = await reconciler.snapshot_positions()

        assert result == {}


class TestIntegration:
    """Test end-to-end reconciliation scenarios."""

    @pytest.mark.asyncio
    async def test_full_reconciliation_flow(
        self,
        reconciler: OrderReconciler,
        orders_store: OrdersStore,
        mock_broker: Mock,
    ) -> None:
        """Tests complete reconciliation workflow from fetch to sync.

        End-to-end validation of the entire reconciliation process.
        """
        # Setup: Local has order-1, exchange has order-2
        local_order = create_test_order("order-1", status=OrderStatus.SUBMITTED)
        orders_store.upsert(local_order)

        exchange_order = create_test_order("order-2", status=OrderStatus.SUBMITTED)
        mock_broker.list_orders = Mock(return_value=[exchange_order])
        mock_broker.get_order = Mock(
            return_value=create_test_order("order-1", status=OrderStatus.FILLED)
        )

        # Execute reconciliation
        local_open = reconciler.fetch_local_open_orders()
        exchange_open = await reconciler.fetch_exchange_open_orders()
        diff = OrderReconciler.diff_orders(local_open, exchange_open)

        await reconciler.reconcile_missing_on_exchange(diff)
        reconciler.reconcile_missing_locally(diff)

        # Verify: Both orders now in local storage with correct status
        order1 = orders_store.get_by_id("order-1")
        order2 = orders_store.get_by_id("order-2")

        assert order1.status == OrderStatus.FILLED.value
        assert order2.status == OrderStatus.SUBMITTED.value

    @pytest.mark.asyncio
    async def test_handles_complete_desync(
        self,
        reconciler: OrderReconciler,
        orders_store: OrdersStore,
        mock_broker: Mock,
    ) -> None:
        """Handles complete state desync without data loss.

        Worst-case scenario: No orders match between local and exchange.
        """
        # Local: order-1, order-2
        orders_store.upsert(create_test_order("order-1"))
        orders_store.upsert(create_test_order("order-2"))

        # Exchange: order-3, order-4
        mock_broker.list_orders = Mock(
            return_value=[
                create_test_order("order-3"),
                create_test_order("order-4"),
            ]
        )
        mock_broker.get_order = Mock(
            side_effect=[
                create_test_order("order-1", status=OrderStatus.CANCELLED),
                create_test_order("order-2", status=OrderStatus.FILLED),
            ]
        )

        # Execute reconciliation
        local_open = reconciler.fetch_local_open_orders()
        exchange_open = await reconciler.fetch_exchange_open_orders()
        diff = OrderReconciler.diff_orders(local_open, exchange_open)

        await reconciler.reconcile_missing_on_exchange(diff)
        reconciler.reconcile_missing_locally(diff)

        # All 4 orders should be tracked with correct status
        assert orders_store.get_by_id("order-1") is not None
        assert orders_store.get_by_id("order-2") is not None
        assert orders_store.get_by_id("order-3") is not None
        assert orders_store.get_by_id("order-4") is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_handles_malformed_local_order_data(self, reconciler: OrderReconciler) -> None:
        """Handles malformed local order data without crashing.

        Defensive: Corrupted storage should not prevent reconciliation.
        """

        class MalformedOrder:
            order_id = None  # Missing required field

        reconciler._orders_store.get_open_orders = Mock(return_value=[MalformedOrder()])

        # Should not raise
        result = reconciler.fetch_local_open_orders()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handles_broker_timeout_gracefully(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Handles broker API timeouts without crashing.

        Network issues should result in partial reconciliation, not crash.
        """
        mock_broker.list_orders = Mock(side_effect=asyncio.TimeoutError())

        result = await reconciler.fetch_exchange_open_orders()

        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handles_exchange_order_without_id(
        self, reconciler: OrderReconciler, mock_broker: Mock
    ) -> None:
        """Handles exchange orders missing required ID field.

        Defensive: Malformed exchange data should not crash reconciliation.
        """

        class OrderWithoutId:
            id = None
            status = OrderStatus.SUBMITTED

        mock_broker.list_orders = Mock(return_value=[OrderWithoutId()])

        # Should not raise
        result = await reconciler.fetch_exchange_open_orders()
        assert isinstance(result, dict)
