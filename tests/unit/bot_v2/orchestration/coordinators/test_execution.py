from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.execution import ExecutionCoordinator
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


def _make_context(*, dry_run: bool = False, advanced: bool = False) -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD, dry_run=dry_run)
    runtime_state = PerpsBotRuntimeState([])
    broker = Mock()
    risk_manager = Mock()
    risk_manager.config = Mock()
    risk_manager.config.enable_dynamic_position_sizing = advanced
    risk_manager.config.enable_market_impact_guard = False
    context = CoordinatorContext(
        config=config,
        registry=ServiceRegistry(config=config, broker=broker, risk_manager=risk_manager),
        event_store=Mock(),
        orders_store=Mock(),
        broker=broker,
        risk_manager=risk_manager,
        symbols=("BTC-PERP",),
        bot_id="perps_bot",
        runtime_state=runtime_state,
    )
    return context


def test_initialize_creates_execution_engine() -> None:
    context = _make_context()
    coordinator = ExecutionCoordinator(context)

    updated = coordinator.initialize(context)

    assert updated.runtime_state.exec_engine is not None
    assert updated.registry.extras["execution_engine"] is updated.runtime_state.exec_engine


def test_initialize_advanced_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    context = _make_context(advanced=True)
    coordinator = ExecutionCoordinator(context)

    updated = coordinator.initialize(context)

    engine = updated.runtime_state.exec_engine
    assert isinstance(engine, AdvancedExecutionEngine)


@pytest.mark.asyncio
async def test_start_background_tasks_skips_in_dry_run() -> None:
    context = _make_context(dry_run=True)
    coordinator = ExecutionCoordinator(context)

    tasks = await coordinator.start_background_tasks()

    assert tasks == []


@pytest.mark.asyncio
async def test_order_reconciliation_cycle_updates_store() -> None:
    context = _make_context()
    coordinator = ExecutionCoordinator(context)
    coordinator.update_context(coordinator.initialize(context))

    reconciler = Mock()
    reconciler.fetch_local_open_orders = Mock(return_value={})
    reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
    reconciler.diff_orders = Mock(return_value=Mock(missing_on_exchange=[], missing_locally=[]))
    reconciler.reconcile_missing_on_exchange = AsyncMock()
    reconciler.reconcile_missing_locally = Mock()
    reconciler.record_snapshot = AsyncMock()

    await coordinator._run_order_reconciliation_cycle(reconciler)

    reconciler.fetch_local_open_orders.assert_called_once()
    reconciler.fetch_exchange_open_orders.assert_awaited_once()
    reconciler.record_snapshot.assert_awaited_once()
