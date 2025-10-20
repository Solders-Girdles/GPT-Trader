"""
Shared fixtures for coordinator integration tests.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, Mock

import pytest
from tests.unit.bot_v2.orchestration.helpers import ScenarioBuilder

from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.coordinators.runtime import RuntimeCoordinator
from bot_v2.orchestration.coordinators.strategy import StrategyCoordinator
from bot_v2.orchestration.coordinators.telemetry import TelemetryCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.fixture
def integration_context():
    """Integration test context with all coordinators."""
    config = BotConfig(
        profile=Profile.DEV, symbols=["BTC-PERP"], dry_run=False
    )  # Use DEV to avoid broker validation
    runtime_state = PerpsBotRuntimeState(["BTC-PERP"])

    broker = Mock()
    risk_manager = Mock()
    orders_store = Mock()
    event_store = Mock()

    registry = ServiceRegistry(config=config)

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="integration-test-bot",
        runtime_state=runtime_state,
        config_controller=Mock(),
        strategy_orchestrator=Mock(),
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def coordinators(integration_context):
    """All coordinators initialized for integration testing."""
    runtime_coord = RuntimeCoordinator(integration_context)
    strategy_coord = StrategyCoordinator(integration_context)
    execution_coord = ExecutionCoordinator(integration_context)
    telemetry_coord = TelemetryCoordinator(integration_context)

    mutable_context = integration_context.with_updates(
        execution_coordinator=execution_coord,
        strategy_coordinator=strategy_coord,
    )

    runtime_coord.update_context(mutable_context)
    strategy_coord.update_context(mutable_context)
    execution_coord.update_context(mutable_context)
    telemetry_coord.update_context(mutable_context)

    return {
        "runtime": runtime_coord,
        "strategy": strategy_coord,
        "execution": execution_coord,
        "telemetry": telemetry_coord,
        "context": mutable_context,
    }


@pytest.fixture
def scenario_builder():
    """Expose ScenarioBuilder helpers for readability."""
    return ScenarioBuilder


@pytest.fixture
def prepared_trading_cycle(integration_context, coordinators):
    """Pre-configure strategy coordinator for trading cycle tests."""
    strategy_coord = coordinators["strategy"]

    integration_context.broker.list_balances = AsyncMock(return_value=[])
    integration_context.broker.list_positions = AsyncMock(return_value=[])
    integration_context.broker.get_account_info = AsyncMock(return_value=None)

    system_monitor = Mock()
    system_monitor.log_status = AsyncMock()
    session_guard = Mock()
    session_guard.should_trade = Mock(return_value=True)

    mutable_context = integration_context.with_updates(
        system_monitor=system_monitor, session_guard=session_guard
    )

    strategy_coord.update_context(mutable_context)

    processor = Mock()
    processor.process_symbol = AsyncMock()
    strategy_coord.set_symbol_processor(processor)

    exec_engine = Mock()
    integration_context.runtime_state.exec_engine = exec_engine

    return {
        "strategy_coord": strategy_coord,
        "processor": processor,
        "exec_engine": exec_engine,
        "context": mutable_context,
        "system_monitor": system_monitor,
        "session_guard": session_guard,
    }


@pytest.fixture
def decision_payload(scenario_builder):
    """Convenience helper for Action.BUY decisions."""
    decision = scenario_builder.create_decision(action=Action.BUY, quantity=Decimal("0.1"))
    product = scenario_builder.create_product()
    return decision, product
