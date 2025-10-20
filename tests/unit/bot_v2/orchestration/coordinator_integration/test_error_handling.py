"""
Error-handling integration scenarios across coordinators.
"""

from decimal import Decimal
from unittest.mock import Mock

import pytest


class TestErrorHandlingIntegration:
    """Verify coordinators recover from cross-cutting failures."""

    @pytest.mark.asyncio
    async def test_execution_failure_triggers_telemetry(
        self, coordinators, integration_context, decision_payload
    ):
        execution_coord = coordinators["execution"]

        exec_engine = Mock()
        exec_engine.place_order = Mock(side_effect=Exception("execution_failed"))
        integration_context.runtime_state.exec_engine = exec_engine

        decision, product = decision_payload

        await execution_coord.execute_decision(
            "BTC-PERP", decision, Decimal("50000"), product, None
        )

        assert integration_context.runtime_state.order_stats["failed"] == 1

    @pytest.mark.asyncio
    async def test_reconciliation_failures_enable_reduce_only(
        self, coordinators, integration_context
    ):
        runtime_coord = coordinators["runtime"]

        broker = Mock()
        orders_store = Mock()
        event_store = Mock()
        runtime_coord.update_context(
            runtime_coord.context.with_updates(
                broker=broker,
                orders_store=orders_store,
                event_store=event_store,
                registry=runtime_coord.context.registry.with_updates(
                    broker=broker, event_store=event_store, orders_store=orders_store
                ),
            )
        )

        reconciler = Mock()
        reconciler.fetch_local_open_orders = Mock(side_effect=Exception("reconcile_failed"))

        with pytest.MonkeyPatch().context() as m:
            m.setattr(
                "bot_v2.orchestration.coordinators.runtime.OrderReconciler",
                lambda **kwargs: reconciler,
            )

            await runtime_coord.reconcile_state_on_startup()

        integration_context.config_controller.set_reduce_only_mode.assert_called_with(
            True,
            reason="startup_reconcile_failed",
            risk_manager=runtime_coord.context.risk_manager,
        )
