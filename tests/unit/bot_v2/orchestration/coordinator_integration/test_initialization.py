"""
Initialization integration tests across coordinators.
"""

import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest


class TestCoordinatorLifecycleIntegration:
    """Verify each coordinator initializes its dependencies."""

    def test_runtime_initialization_sets_up_all_coordinators(
        self, coordinators, integration_context
    ):
        runtime_coord = coordinators["runtime"]

        # Use patching to avoid modifying strict Pydantic config
        with unittest.mock.patch("bot_v2.orchestration.engines.runtime.broker_management.discover_derivatives_eligibility") as mock_discover:
            mock_discover.return_value = Mock(eligibility=True)

            result = runtime_coord.initialize(integration_context)

        assert result.broker is not None
        assert result.risk_manager is not None

    def test_strategy_coordinator_initialization_with_orchestrator(
        self, coordinators, integration_context
    ):
        strategy_coord = coordinators["strategy"]
        orchestrator = Mock()
        mutable_context = integration_context.with_updates(strategy_orchestrator=orchestrator)
        strategy_coord.update_context(mutable_context)

        result = strategy_coord.initialize(mutable_context)

        assert result == mutable_context

    @pytest.mark.asyncio
    async def test_execution_coordinator_initialization_with_engine(
        self, coordinators, integration_context
    ):
        execution_coord = coordinators["execution"]

        # Ensure context has an engine mock if new architecture expects one passed in,
        # OR check that it creates one.
        # In simplified architecture, if broker & risk are present, engine is created.

        integration_context.broker = Mock()
        integration_context.risk_manager = Mock()

        result = await execution_coord.initialize(integration_context)

        # The new execution coordinator stores engine in its service, not necessarily in result.runtime_state
        # But it might not expose it easily.
        # We can check if internal service is initialized.
        assert execution_coord._order_placement.execution_engine is not None

    def test_telemetry_coordinator_initialization_with_services(
        self, coordinators, integration_context
    ):
        telemetry_coord = coordinators["telemetry"]
        integration_context.broker.__class__ = Mock()
        from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage

        integration_context.broker.__class__ = CoinbaseBrokerage

        result = telemetry_coord.initialize(integration_context)

        extras = result.registry.extras
        assert "account_telemetry" in extras
        assert "market_monitor" in extras


class TestTradingCycleIntegration:
    """Exercise a banner trading cycle end-to-end."""

    @pytest.mark.asyncio
    async def test_complete_trading_cycle_execution(self, prepared_trading_cycle):
        strategy_coord = prepared_trading_cycle["strategy_coord"]
        processor = prepared_trading_cycle["processor"]

        # Mock internal backfill to avoid iteration on Mock error
        strategy_coord._backfill_history = AsyncMock()

        await strategy_coord.run_cycle()

        processor.process_symbol.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_to_execution_hand_off(
        self, coordinators, integration_context, scenario_builder
    ):
        strategy_coord = coordinators["strategy"]
        execution_coord = coordinators["execution"]

        exec_engine = Mock()
        exec_engine.place_order = Mock(return_value=Mock(order_id="order-123", symbol="BTC-PERP", side="BUY", quantity=10))
        integration_context.runtime_state.exec_engine = exec_engine

        # Inject engine into execution coordinator service
        if hasattr(execution_coord, "_order_placement"):
            execution_coord._order_placement.execution_engine = exec_engine

        decision = scenario_builder.create_decision()
        product = scenario_builder.create_product()

        # Ensure execution coordinator is wired in strategy coord
        # (TradingEngine.execute_decision calls context.execution_coordinator.execute_decision)
        strategy_coord.update_context(strategy_coord.context.with_updates(
            execution_coordinator=execution_coord
        ))

        await strategy_coord.execute_decision("BTC-PERP", decision, Decimal("50000"), product, None)

        exec_engine.place_order.assert_called_once()
