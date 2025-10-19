"""
Enhanced RuntimeCoordinator tests for the coordinator implementation.

Tests broker bootstrapping, risk manager initialization, state transitions,
reconciliation flows, bootstrap failures, and runtime safety toggles.
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

BOT_ID = "coinbase_trader"


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
        bot_id=BOT_ID,
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


class TestRuntimeCoordinatorStateTransitions:
    """Test runtime state transitions and safety toggles."""

    def test_set_reduce_only_mode_calls_controller(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test set_reduce_only_mode calls config controller."""
        base_context.config_controller.set_reduce_only_mode = Mock(return_value=True)
        coordinator.update_context(base_context)

        coordinator.set_reduce_only_mode(True, "test_reason")

        base_context.config_controller.set_reduce_only_mode.assert_called_with(
            True, reason="test_reason", risk_manager=base_context.risk_manager
        )

    def test_set_reduce_only_mode_emits_metric_on_success(
        coordinator: RuntimeCoordinator,
        base_context: CoordinatorContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test set_reduce_only_mode emits metric when successful."""
        base_context.config_controller.set_reduce_only_mode = Mock(return_value=True)
        emit_metric = Mock()
        monkeypatch.setattr("bot_v2.orchestration.coordinators.runtime.emit_metric", emit_metric)
        coordinator.update_context(base_context)

        coordinator.set_reduce_only_mode(True, "test_reason")

        emit_metric.assert_called_once()

    def test_is_reduce_only_mode_delegates_to_controller(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test is_reduce_only_mode delegates to config controller."""
        base_context.config_controller.is_reduce_only_mode = Mock(return_value=True)
        coordinator.update_context(base_context)

        result = coordinator.is_reduce_only_mode()

        assert result is True
        base_context.config_controller.is_reduce_only_mode.assert_called_once_with(
            base_context.risk_manager
        )

    def test_on_risk_state_change_handles_reduce_only_toggle(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test on_risk_state_change handles reduce-only mode changes."""
        base_context.config_controller.apply_risk_update = Mock(return_value=True)
        coordinator.update_context(base_context)

        state = RiskRuntimeState(reduce_only_mode=True, last_reduce_only_reason="risk_trigger")

        coordinator.on_risk_state_change(state)

        base_context.config_controller.apply_risk_update.assert_called_once_with(True)

    def test_on_risk_state_change_emits_metric_on_toggle(
        coordinator: RuntimeCoordinator,
        base_context: CoordinatorContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test on_risk_state_change emits metric when reduce-only toggled."""
        base_context.config_controller.apply_risk_update = Mock(return_value=True)
        emit_metric = Mock()
        monkeypatch.setattr("bot_v2.orchestration.coordinators.runtime.emit_metric", emit_metric)
        coordinator.update_context(base_context)

        state = RiskRuntimeState(reduce_only_mode=True, last_reduce_only_reason="risk_trigger")

        coordinator.on_risk_state_change(state)

        emit_metric.assert_called_once()


class TestRuntimeCoordinatorBootstrapFailures:
    """Test bootstrap failure handling and error recovery."""

    def test_init_broker_handles_missing_dependencies(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _init_broker handles missing broker or risk manager."""
        base_context.broker = None
        base_context.risk_manager = None
        coordinator.update_context(base_context)

        result = coordinator._init_broker(base_context)

        # Should return context unchanged when dependencies missing
        assert result == base_context

    def test_init_risk_manager_handles_missing_config(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _init_risk_manager handles missing risk config gracefully."""
        base_context.registry.risk_manager = None
        coordinator.update_context(base_context)

        # Mock risk config loading to fail
        coordinator._risk_config_cls = Mock(side_effect=Exception("config_load_failed"))

        result = coordinator._init_risk_manager(base_context)

        # Should still create risk manager with defaults
        assert result.risk_manager is not None

    def test_validate_broker_environment_blocks_invalid_broker_hint(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _validate_broker_environment blocks invalid broker hints."""
        settings = SimpleNamespace(broker_hint="invalid_broker")
        base_context.config.derivatives_enabled = True

        with pytest.raises(RuntimeError, match="BROKER must be set to 'coinbase'"):
            coordinator._validate_broker_environment(base_context, settings)

    def test_validate_broker_environment_blocks_sandbox_for_live_trading(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _validate_broker_environment blocks sandbox for live trading."""
        settings = SimpleNamespace(broker_hint="coinbase", coinbase_sandbox_enabled=True)
        base_context.config.derivatives_enabled = True

        with pytest.raises(RuntimeError, match="COINBASE_SANDBOX=1 is not supported"):
            coordinator._validate_broker_environment(base_context, settings)

    def test_validate_broker_environment_requires_cdp_for_derivatives(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _validate_broker_environment requires CDP credentials for derivatives."""
        settings = SimpleNamespace(
            broker_hint="coinbase",
            coinbase_sandbox_enabled=False,
            raw_env={},
            coinbase_api_mode="advanced",
        )
        base_context.config.derivatives_enabled = True

        with pytest.raises(RuntimeError, match="Missing CDP JWT credentials"):
            coordinator._validate_broker_environment(base_context, settings)

    def test_validate_broker_environment_blocks_perps_without_derivatives_enabled(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _validate_broker_environment blocks perps symbols when derivatives disabled."""
        settings = SimpleNamespace(
            broker_hint="coinbase",
            coinbase_sandbox_enabled=False,
            raw_env={"COINBASE_API_KEY": "key", "COINBASE_API_SECRET": "secret"},
            coinbase_api_mode="advanced",
        )
        base_context.config.derivatives_enabled = False
        base_context.symbols = ("BTC-PERP",)

        with pytest.raises(RuntimeError, match="COINBASE_ENABLE_DERIVATIVES is not enabled"):
            coordinator._validate_broker_environment(base_context, settings)


class TestRuntimeCoordinatorReconciliation:
    """Test state reconciliation and startup flows."""

    @pytest.mark.asyncio
    async def test_reconcile_state_on_startup_skips_dry_run(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test reconcile_state_on_startup skips in dry run mode."""
        base_context.config.dry_run = True
        coordinator.update_context(base_context)

        await coordinator.reconcile_state_on_startup()

        # Should return without attempting reconciliation

    @pytest.mark.asyncio
    async def test_reconcile_state_on_startup_skips_when_skip_flag_set(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test reconcile_state_on_startup skips when skip flag is set."""
        base_context.config.perps_skip_startup_reconcile = True
        coordinator.update_context(base_context)

        await coordinator.reconcile_state_on_startup()

        # Should return without attempting reconciliation

    @pytest.mark.asyncio
    async def test_reconcile_state_on_startup_handles_reconciler_errors(
        coordinator: RuntimeCoordinator,
        base_context: CoordinatorContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test reconcile_state_on_startup handles reconciler errors gracefully."""
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

        # Mock reconciler to fail
        reconciler = Mock()
        reconciler.fetch_local_open_orders = Mock(side_effect=Exception("reconcile_failed"))
        monkeypatch.setattr(
            "bot_v2.orchestration.coordinators.runtime.OrderReconciler",
            lambda **kwargs: reconciler,
        )

        await coordinator.reconcile_state_on_startup()

        # Should have enabled reduce-only mode due to failure
        base_context.config_controller.set_reduce_only_mode.assert_called_with(
            True, reason="startup_reconcile_failed"
        )

    @pytest.mark.asyncio
    async def test_reconcile_state_on_startup_emits_error_event_on_failure(
        coordinator: RuntimeCoordinator,
        base_context: CoordinatorContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test reconcile_state_on_startup emits error event on failure."""
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

        # Mock reconciler to fail
        reconciler = Mock()
        reconciler.fetch_local_open_orders = Mock(side_effect=Exception("reconcile_failed"))
        monkeypatch.setattr(
            "bot_v2.orchestration.coordinators.runtime.OrderReconciler",
            lambda **kwargs: reconciler,
        )

        await coordinator.reconcile_state_on_startup()

        # Should have appended error to event store
        event_store.append_error.assert_called_once()


class TestRuntimeCoordinatorInitialization:
    """Test initialization and bootstrap flows."""

    def test_initialize_calls_broker_and_risk_init(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test initialize calls both broker and risk manager initialization."""
        # Mock the init methods
        coordinator._init_broker = Mock(return_value=base_context)
        coordinator._init_risk_manager = Mock(return_value=base_context)

        result = coordinator.initialize(base_context)

        coordinator._init_broker.assert_called_once()
        coordinator._init_risk_manager.assert_called_once()
        assert result == base_context

    def test_bootstrap_delegates_to_initialize(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test bootstrap delegates to initialize (legacy compatibility)."""
        coordinator.initialize = Mock(return_value=base_context)

        coordinator.bootstrap()

        coordinator.initialize.assert_called_once_with(base_context)

    def test_update_context_updates_internals(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test update_context updates internal references."""
        new_context = base_context.with_updates(
            config_controller=Mock(),
            strategy_orchestrator=Mock(),
            execution_coordinator=Mock(),
            product_cache={"test": "cache"},
        )

        coordinator.update_context(new_context)

        assert coordinator._config_controller == new_context.config_controller
        assert coordinator._strategy_orchestrator == new_context.strategy_orchestrator
        assert coordinator._execution_coordinator == new_context.execution_coordinator
        assert coordinator._product_cache == new_context.product_cache


class TestRuntimeCoordinatorBrokerBootstrap:
    """Test broker bootstrap artifacts and context updates."""

    def test_apply_broker_bootstrap_updates_context(
        coordinator: RuntimeCoordinator, base_context: CoordinatorContext
    ) -> None:
        """Test _apply_broker_bootstrap updates context with artifacts."""
        artifacts = BrokerBootstrapArtifacts(
            broker=Mock(),
            registry_updates={"test_service": Mock()},
            event_store=Mock(),
            products=[SimpleNamespace(symbol="BTC-PERP")],
        )

        result = coordinator._apply_broker_bootstrap(base_context, artifacts)

        assert result.broker == artifacts.broker
        assert result.event_store == artifacts.event_store
        assert "BTC-PERP" in result.product_cache

    def test_hydrate_product_cache_handles_empty_products(coordinator: RuntimeCoordinator) -> None:
        """Test _hydrate_product_cache handles empty product list."""
        coordinator._hydrate_product_cache([])

        # Should not modify cache when no products
        assert coordinator._product_cache is None

    def test_hydrate_product_cache_creates_cache_when_needed(
        coordinator: RuntimeCoordinator,
    ) -> None:
        """Test _hydrate_product_cache creates cache dict when needed."""
        products = [SimpleNamespace(symbol="BTC-PERP"), SimpleNamespace(symbol="ETH-PERP")]

        coordinator._hydrate_product_cache(products)

        assert isinstance(coordinator._product_cache, dict)
        assert "BTC-PERP" in coordinator._product_cache
        assert "ETH-PERP" in coordinator._product_cache


class TestRuntimeCoordinatorProperties:
    """Test property accessors and factory methods."""

    def test_deterministic_broker_cls_returns_correct_type(coordinator: RuntimeCoordinator) -> None:
        """Test _deterministic_broker_cls returns DeterministicBroker type."""
        from bot_v2.orchestration.deterministic_broker import DeterministicBroker

        cls = coordinator._deterministic_broker_cls
        assert cls == DeterministicBroker

    def test_risk_config_cls_returns_correct_type(coordinator: RuntimeCoordinator) -> None:
        """Test _risk_config_cls returns RiskConfig type."""
        from bot_v2.config.live_trade_config import RiskConfig

        cls = coordinator._risk_config_cls
        assert cls == RiskConfig

    def test_risk_manager_cls_returns_correct_type(coordinator: RuntimeCoordinator) -> None:
        """Test _risk_manager_cls returns LiveRiskManager type."""
        from bot_v2.features.live_trade.risk import LiveRiskManager

        cls = coordinator._risk_manager_cls
        assert cls == LiveRiskManager

    def test_order_reconciler_cls_returns_correct_type(coordinator: RuntimeCoordinator) -> None:
        """Test _order_reconciler_cls returns OrderReconciler type."""
        from bot_v2.orchestration.order_reconciler import OrderReconciler

        cls = coordinator._order_reconciler_cls
        assert cls == OrderReconciler
