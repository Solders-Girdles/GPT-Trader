"""
Streaming and order lock integration tests.
"""

from decimal import Decimal


class TestStreamingIntegration:
    """Ensure streaming updates propagate."""

    def test_streaming_mark_updates_reach_strategy_coordinator(
        self, coordinators, integration_context
    ):
        telemetry_coord = coordinators["telemetry"]
        strategy_coord = coordinators["strategy"]

        updated_context = integration_context.with_updates(strategy_coordinator=strategy_coord)
        telemetry_coord.update_context(updated_context)
        strategy_coord.update_context(updated_context)

        telemetry_coord._update_mark_and_metrics(updated_context, "BTC-PERP", Decimal("50000"))

        runtime_state = updated_context.runtime_state
        assert "BTC-PERP" in runtime_state.mark_windows
        assert runtime_state.mark_windows["BTC-PERP"][-1] == Decimal("50000")


class TestOrderLockIntegration:
    """Confirm order locks are shared across coordinators."""

    def test_order_lock_coordination_between_coordinators(self, coordinators, integration_context):
        strategy_coord = coordinators["strategy"]
        execution_coord = coordinators["execution"]

        shared_context = integration_context.with_updates(execution_coordinator=execution_coord)
        strategy_coord.update_context(shared_context)
        execution_coord.update_context(shared_context)

        strategy_lock = strategy_coord.ensure_order_lock()
        execution_lock = execution_coord.ensure_order_lock()

        assert strategy_lock is execution_lock
        assert shared_context.runtime_state.order_lock is strategy_lock
