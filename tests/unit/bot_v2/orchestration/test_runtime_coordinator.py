"""
RuntimeCoordinator tests for the coordinator implementation.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from bot_v2.features.live_trade.risk import RiskRuntimeState
from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.runtime import (
    BrokerBootstrapArtifacts,
    BrokerBootstrapError,
    RuntimeCoordinator,
)
from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState
from bot_v2.orchestration.service_registry import ServiceRegistry


@pytest.fixture
def base_context() -> CoordinatorContext:
    config = BotConfig(profile=Profile.PROD, symbols=["BTC-PERP"], dry_run=False)
    runtime_state = PerpsBotRuntimeState(config.symbols)
    runtime_state.product_map = {}

    broker = None
    risk_manager = Mock()
    risk_manager.set_state_listener = Mock()
    orders_store = Mock()
    event_store = Mock()

    registry = ServiceRegistry(
        config=config,
        broker=broker,
        risk_manager=risk_manager,
        event_store=event_store,
        orders_store=orders_store,
    )

    controller = Mock()
    controller.reduce_only_mode = False
    controller.is_reduce_only_mode.return_value = False
    controller.set_reduce_only_mode.return_value = True
    controller.apply_risk_update.return_value = True
    controller.sync_with_risk_manager = Mock()

    context = CoordinatorContext(
        config=config,
        registry=registry,
        event_store=event_store,
        orders_store=orders_store,
        broker=broker,
        risk_manager=risk_manager,
        symbols=tuple(config.symbols),
        bot_id="perps_bot",
        runtime_state=runtime_state,
        config_controller=controller,
        set_running_flag=lambda _: None,
    )
    return context


@pytest.fixture
def coordinator(base_context: CoordinatorContext) -> RuntimeCoordinator:
    return RuntimeCoordinator(base_context)


def test_init_broker_uses_deterministic_for_dev(
    base_context: CoordinatorContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_context.config
    config.profile = Profile.DEV
    config.mock_broker = False
    config.perps_paper_trading = False
    base_context = base_context.with_updates(config=config, registry=base_context.registry)

    stub_broker = object()
    monkeypatch.setattr(
        "bot_v2.orchestration.coordinators.runtime.DeterministicBroker",
        lambda: stub_broker,
    )

    coordinator = RuntimeCoordinator(base_context)
    updated = coordinator._init_broker(base_context)

    assert updated.broker is stub_broker
    assert updated.registry.broker is stub_broker


def test_init_broker_raises_on_connection_failure(
    base_context: CoordinatorContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = base_context.config
    config.profile = Profile.PROD
    config.mock_broker = False
    config.perps_paper_trading = False
    config.perps_force_mock = False
    config.derivatives_enabled = True

    real_broker = Mock()
    real_broker.connect.return_value = False

    monkeypatch.setattr(
        "bot_v2.orchestration.coordinators.runtime.create_brokerage",
        lambda *args, **kwargs: (real_broker, base_context.event_store, Mock(), Mock()),
    )
    monkeypatch.setattr(
        RuntimeCoordinator,
        "_validate_broker_environment",
        lambda self, context, settings=None: None,
    )

    coordinator = RuntimeCoordinator(base_context)
    with pytest.raises(BrokerBootstrapError):
        coordinator._init_broker(base_context)


def test_apply_broker_bootstrap_populates_registry(base_context: CoordinatorContext) -> None:
    coordinator = RuntimeCoordinator(base_context)
    artifacts = BrokerBootstrapArtifacts(
        broker=Mock(),
        registry_updates={"market_data_service": Mock()},
        event_store=Mock(),
        products=[SimpleNamespace(symbol="BTC-PERP")],
    )

    updated = coordinator._apply_broker_bootstrap(base_context, artifacts)

    assert updated.registry.market_data_service is artifacts.registry_updates["market_data_service"]
    assert updated.event_store is artifacts.event_store
    assert "BTC-PERP" in (updated.product_cache or {})


def test_validate_broker_environment_checks_settings(
    base_context: CoordinatorContext,
) -> None:
    coordinator = RuntimeCoordinator(base_context)
    settings = SimpleNamespace(
        broker_hint="other",
        coinbase_sandbox_enabled=False,
        raw_env={
            "COINBASE_API_MODE": "advanced",
            "COINBASE_API_KEY": "abc",
            "COINBASE_API_SECRET": "xyz",
        },
        coinbase_api_mode="advanced",
    )

    with pytest.raises(RuntimeError):
        coordinator._validate_broker_environment(base_context, settings)


def test_set_reduce_only_mode_updates_controller(coordinator: RuntimeCoordinator) -> None:
    coordinator.context.config_controller.set_reduce_only_mode.return_value = True

    coordinator.set_reduce_only_mode(True, "test")

    coordinator.context.config_controller.set_reduce_only_mode.assert_called_with(
        True, reason="test", risk_manager=coordinator.context.risk_manager
    )


def test_is_reduce_only_mode_uses_controller(coordinator: RuntimeCoordinator) -> None:
    coordinator.context.config_controller.is_reduce_only_mode.return_value = True

    assert coordinator.is_reduce_only_mode() is True


@pytest.mark.asyncio
async def test_reconcile_state_on_startup_handles_missing_broker(
    coordinator: RuntimeCoordinator,
) -> None:
    context = coordinator.context.with_updates(broker=None, registry=coordinator.context.registry)
    coordinator.update_context(context)

    await coordinator.reconcile_state_on_startup()

    # Should simply return without raising


@pytest.mark.asyncio
async def test_reconcile_state_on_startup_runs_reconciler(
    coordinator: RuntimeCoordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = Mock()
    orders_store = Mock()
    event_store = Mock()
    coordinator.update_context(
        coordinator.context.with_updates(
            broker=broker,
            orders_store=orders_store,
            event_store=event_store,
            registry=coordinator.context.registry.with_updates(
                broker=broker, event_store=event_store, orders_store=orders_store
            ),
        )
    )

    reconciler = Mock()
    reconciler.fetch_local_open_orders.return_value = {}
    reconciler.fetch_exchange_open_orders = AsyncMock(return_value={})
    reconciler.diff_orders.return_value = Mock(missing_on_exchange=[], missing_locally=[])
    reconciler.reconcile_missing_on_exchange = AsyncMock()
    reconciler.reconcile_missing_locally = Mock()
    reconciler.record_snapshot = AsyncMock()
    reconciler.snapshot_positions = AsyncMock(return_value={})

    monkeypatch.setattr(
        "bot_v2.orchestration.coordinators.runtime.OrderReconciler",
        lambda **kwargs: reconciler,
    )

    await coordinator.reconcile_state_on_startup()

    reconciler.fetch_exchange_open_orders.assert_called_once()


def test_on_risk_state_change_emits_metrics(
    coordinator: RuntimeCoordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = RiskRuntimeState(reduce_only_mode=True, last_reduce_only_reason="risk")
    emit_metric = Mock()
    monkeypatch.setattr("bot_v2.orchestration.coordinators.runtime.emit_metric", emit_metric)

    coordinator.update_context(coordinator.context.with_updates(event_store=Mock()))
    coordinator.on_risk_state_change(state)

    emit_metric.assert_called_once()
