from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.strategy import StrategyCoordinator
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.orchestration.configuration.validation import ConfigValidationResult


shutdown_flag = SimpleNamespace(value=True)


class DummyQuote:
    def __init__(self, last: Decimal, ts: object | None = None) -> None:
        self.last = last
        self.ts = ts


def _make_context(
    *,
    symbols: tuple[str, ...] = ("BTC-PERP",),
    broker: object | None = None,
    risk_manager: object | None = None,
    session_guard: object | None = None,
    configuration_guardian: object | None = None,
    system_monitor: object | None = None,
    execution_coordinator: object | None = None,
    set_reduce_only_mode: object | None = None,
    shutdown_hook: object | None = None,
    set_running_flag: object | None = None,
    strategy_orchestrator: object | None = None,
) -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD)
    runtime_state = PerpsBotRuntimeState(list(symbols))

    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=Mock(),
        orders_store=Mock(),
    )

    return CoordinatorContext(
        config=config,
        registry=registry,
        broker=broker,
        risk_manager=risk_manager,
        event_store=registry.event_store,
        orders_store=registry.orders_store,
        symbols=symbols,
        bot_id="perps_bot",
        runtime_state=runtime_state,
        strategy_orchestrator=strategy_orchestrator
        or Mock(process_symbol=Mock(return_value=None), requires_context=False),
        execution_coordinator=execution_coordinator,
        session_guard=session_guard,
        configuration_guardian=configuration_guardian,
        system_monitor=system_monitor,
        set_reduce_only_mode=set_reduce_only_mode,
        shutdown_hook=shutdown_hook,
        set_running_flag=set_running_flag or (lambda value: setattr(shutdown_flag, "value", value)),
    )


@pytest.mark.asyncio
async def test_run_cycle_skips_when_session_guard_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = Mock()
    broker.get_quote = Mock(return_value=DummyQuote(Decimal("50000")))
    risk_manager = Mock()
    risk_manager.last_mark_update = {}
    session_guard = Mock()
    session_guard.should_trade.return_value = False
    system_monitor = AsyncMock()
    configuration_guardian = Mock()
    configuration_guardian.pre_cycle_check.return_value = ConfigValidationResult(is_valid=True)

    context = _make_context(
        broker=broker,
        risk_manager=risk_manager,
        session_guard=session_guard,
        configuration_guardian=configuration_guardian,
        system_monitor=system_monitor,
        execution_coordinator=Mock(),
    )

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    coordinator = StrategyCoordinator(context)
    await coordinator.run_cycle()

    system_monitor.log_status.assert_awaited_once()
    session_guard.should_trade.assert_called_once()


@pytest.mark.asyncio
async def test_update_marks_updates_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    broker = Mock()
    broker.get_quote = Mock(return_value=DummyQuote(Decimal("27000")))
    risk_manager = Mock()
    risk_manager.last_mark_update = {}

    context = _make_context(broker=broker, risk_manager=risk_manager)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    coordinator = StrategyCoordinator(context)
    await coordinator.update_marks()

    runtime_state = context.runtime_state
    assert runtime_state is not None
    assert runtime_state.mark_windows["BTC-PERP"]


@pytest.mark.asyncio
async def test_validate_configuration_handles_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    guardian = Mock()
    guardian.pre_cycle_check.return_value = ConfigValidationResult(
        is_valid=False,
        errors=["critical configuration emergency_shutdown"],
    )
    shutdown_called = AsyncMock()
    reduce_only = Mock()

    context = _make_context(
        configuration_guardian=guardian,
        set_reduce_only_mode=reduce_only,
        shutdown_hook=shutdown_called,
    )

    coordinator = StrategyCoordinator(context)
    result = await coordinator._validate_configuration_and_handle_drift(
        {"balances": [], "positions": [], "position_map": {}, "account_equity": None}
    )

    assert result is False
    shutdown_called.assert_awaited_once()
    reduce_only.assert_not_called()


@pytest.mark.asyncio
async def test_process_symbol_invokes_strategy_orchestrator() -> None:
    orchestrator = Mock()
    orchestrator.process_symbol = AsyncMock()
    orchestrator.requires_context = True

    context = _make_context(
        strategy_orchestrator=orchestrator,
        execution_coordinator=Mock(),
    )

    coordinator = StrategyCoordinator(context)
    await coordinator.process_symbol("BTC-PERP", balances=[], position_map={})

    orchestrator.process_symbol.assert_awaited_once_with("BTC-PERP", [], {})
