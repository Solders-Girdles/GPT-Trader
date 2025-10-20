"""
Initialization integration tests across coordinators.
"""

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest


class TestCoordinatorLifecycleIntegration:
    """Verify each coordinator initializes its dependencies."""

    def test_runtime_initialization_sets_up_all_coordinators(
        self, coordinators, integration_context
    ):
        runtime_coord = coordinators["runtime"]

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

    def test_execution_coordinator_initialization_with_engine(
        self, coordinators, integration_context
    ):
        execution_coord = coordinators["execution"]

        result = execution_coord.initialize(integration_context)

        assert result.runtime_state.exec_engine is not None

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

        await strategy_coord.run_cycle()

        processor.process_symbol.assert_called_once()

    @pytest.mark.asyncio
    async def test_strategy_to_execution_hand_off(
        self, coordinators, integration_context, scenario_builder
    ):
        strategy_coord = coordinators["strategy"]

        exec_engine = Mock()
        exec_engine.place_order = AsyncMock(return_value=Mock(id="order-123"))
        integration_context.runtime_state.exec_engine = exec_engine

        decision = scenario_builder.create_decision()
        product = scenario_builder.create_product()

        await strategy_coord.execute_decision("BTC-PERP", decision, Decimal("50000"), product, None)

        exec_engine.place_order.assert_called_once()
