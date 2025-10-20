"""
Tests covering order reconciliation utility methods.
"""

from unittest.mock import AsyncMock, Mock

import pytest


class TestExecutionCoordinatorReconciliation:
    """Validate reconciliation cycles and failure handling."""

    @pytest.mark.asyncio
    async def test_reconciliation_cycle_success(
        self, execution_coordinator, execution_context, fake_order
    ):
        mock_reconciler = Mock()
        mock_reconciler.fetch_local_open_orders = Mock(return_value={"order-123": fake_order})
        mock_reconciler.fetch_exchange_open_orders = AsyncMock(
            return_value={"order-123": fake_order}
        )
        mock_reconciler.diff_orders = Mock(
            return_value=Mock(missing_on_exchange={"order-123": fake_order}, missing_locally={})
        )
        mock_reconciler.reconcile_missing_on_exchange = AsyncMock()
        mock_reconciler.reconcile_missing_locally = Mock()
        mock_reconciler.record_snapshot = AsyncMock()

        execution_context = execution_context.with_updates(orders_store=Mock())
        execution_coordinator.update_context(execution_context)

        await execution_coordinator._run_order_reconciliation_cycle(mock_reconciler)

        mock_reconciler.fetch_local_open_orders.assert_called_once()
        mock_reconciler.fetch_exchange_open_orders.assert_awaited_once()
        mock_reconciler.diff_orders.assert_called_once()
        mock_reconciler.reconcile_missing_on_exchange.assert_awaited_once()
        mock_reconciler.reconcile_missing_locally.assert_called_once()
        mock_reconciler.record_snapshot.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reconciliation_handles_orders_store_failure(
        self, execution_coordinator, execution_context, fake_order
    ):
        mock_reconciler = Mock()
        mock_reconciler.fetch_local_open_orders = Mock(return_value={})
        mock_reconciler.fetch_exchange_open_orders = AsyncMock(
            return_value={"order-123": fake_order}
        )
        mock_reconciler.diff_orders = Mock(
            return_value=Mock(missing_on_exchange={}, missing_locally={"order-123": fake_order})
        )
        mock_reconciler.reconcile_missing_on_exchange = AsyncMock()
        mock_reconciler.reconcile_missing_locally = Mock()
        mock_reconciler.record_snapshot = AsyncMock()

        orders_store = Mock()
        orders_store.upsert = Mock(side_effect=Exception("Upsert failed"))
        execution_context = execution_context.with_updates(orders_store=orders_store)
        execution_coordinator.update_context(execution_context)

        await execution_coordinator._run_order_reconciliation_cycle(mock_reconciler)

        orders_store.upsert.assert_called_once_with(fake_order)
