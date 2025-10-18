"""Legacy runtime coordinator facade wrapping the coordinator package."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskRuntimeState
from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.orchestration.coordinator_facades import (
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
)
from bot_v2.orchestration.coordinators.base import CoordinatorContext
from bot_v2.orchestration.coordinators.runtime import (
    BrokerBootstrapArtifacts,
    BrokerBootstrapError,
)
from bot_v2.orchestration.coordinators.runtime import (
    RuntimeCoordinator as _RuntimeCoordinator,
)
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
    from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.persistence.event_store import EventStore

logger = get_logger(__name__, component="runtime_coordinator_facade")


class RuntimeCoordinator(
    BaseCoordinatorFacade,
    ContextPreservingCoordinator,
    _RuntimeCoordinator,
):
    """Compatibility layer for existing PerpsBot runtime coordination."""

    def __init__(self, bot: PerpsBot) -> None:
        context = self._setup_facade(bot)
        super().__init__(
            context,
            config_controller=getattr(bot, "config_controller", None),
            strategy_orchestrator=getattr(bot, "strategy_orchestrator", None),
            execution_coordinator=getattr(bot, "execution_coordinator", None),
            product_cache=getattr(bot, "_product_map", None),
        )

    # ------------------------------------------------------------------
    @ContextPreservingCoordinator.context_action(sync_after=True)
    def bootstrap(self) -> None:
        super().bootstrap()

    def _context_with_bot_config(self, ctx: CoordinatorContext) -> CoordinatorContext:
        bot_config = getattr(self._bot, "config", None)
        if bot_config is not None and ctx.config is not bot_config:
            symbols = getattr(bot_config, "symbols", None)
            symbol_tuple: tuple[str, ...] | None = None
            if symbols is not None:
                try:
                    symbol_tuple = tuple(symbols)
                except TypeError:
                    symbol_tuple = None
            overrides: dict[str, Any] = {"config": bot_config}
            if symbol_tuple is not None:
                overrides["symbols"] = symbol_tuple
            ctx = ctx.with_updates(**overrides)
        return ctx

    def _init_broker(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self._refresh_context_from_bot()
        ctx = self._context_with_bot_config(ctx)

        bot_registry = getattr(self._bot, "registry", None)
        external_broker = getattr(bot_registry, "broker", None)
        if external_broker is not None:
            ctx = ctx.with_updates(broker=external_broker)
            self.update_context(ctx)
            return ctx

        ctx = ctx.with_updates(
            broker=None,
            registry=ctx.registry.with_updates(broker=None),
        )
        updated = super()._init_broker(ctx)
        self.update_context(updated)
        return updated

    def _init_risk_manager(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self._refresh_context_from_bot()
        ctx = self._context_with_bot_config(ctx)
        bot_registry = getattr(self._bot, "registry", None)
        external_risk = getattr(bot_registry, "risk_manager", None)
        if external_risk is not None:
            external_risk.set_state_listener(self.on_risk_state_change)
            controller = getattr(self, "_config_controller", None)
            if controller is not None:
                controller.sync_with_risk_manager(external_risk)
                try:
                    reduce_only = bool(controller.reduce_only_mode)
                except Exception:
                    reduce_only = False
                external_risk.set_reduce_only_mode(reduce_only, reason="config_init")
            ctx = ctx.with_updates(risk_manager=external_risk)
            self.update_context(ctx)
            return ctx

        ctx = ctx.with_updates(
            risk_manager=None,
            registry=ctx.registry.with_updates(risk_manager=None),
        )
        updated = super()._init_risk_manager(ctx)
        self.update_context(updated)
        return updated

    @ContextPreservingCoordinator.context_action()
    def _validate_broker_environment(self) -> None:  # type: ignore[override]
        ctx = self._context_with_bot_config(self.context)
        super()._validate_broker_environment(ctx)

    @ContextPreservingCoordinator.context_action(sync_after=True)
    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:  # type: ignore[override]
        super().set_reduce_only_mode(enabled, reason)

    @ContextPreservingCoordinator.context_action()
    def is_reduce_only_mode(self) -> bool:  # type: ignore[override]
        return bool(super().is_reduce_only_mode())

    @ContextPreservingCoordinator.context_action(sync_after=True)
    def on_risk_state_change(self, state: RiskRuntimeState) -> None:  # type: ignore[override]
        super().on_risk_state_change(state)

    @ContextPreservingCoordinator.context_action(sync_after=True)
    async def reconcile_state_on_startup(self) -> None:  # type: ignore[override]
        context = self._context_with_bot_config(self.context)
        if context.broker is None:
            fallback_broker: Any | None = None
            try:
                fallback_broker = getattr(self._bot, "broker")
            except Exception:  # pragma: no cover - defensive guard
                fallback_broker = None
            if fallback_broker is not None:
                updated = context.with_updates(broker=cast("IBrokerage", fallback_broker))
                self.update_context(updated)
        await super().reconcile_state_on_startup()

    # ------------------------------------------------------------------
    def _sync_bot(self, context: CoordinatorContext) -> None:
        bot = self._bot
        bot.registry = context.registry
        if context.event_store is not None:
            bot.event_store = context.event_store
        if context.orders_store is not None:
            bot.orders_store = context.orders_store
        broker_candidate = context.broker or context.registry.broker
        if broker_candidate is not None:
            bot.broker = cast("IBrokerage", broker_candidate)
        risk_candidate = context.risk_manager or context.registry.risk_manager
        if risk_candidate is not None:
            bot.risk_manager = cast("LiveRiskManager", risk_candidate)
        if context.product_cache is not None:
            if hasattr(bot, "_state") and hasattr(bot._state, "product_map"):
                bot._state.product_map = context.product_cache
            runtime_state = getattr(bot, "runtime_state", None)
            if runtime_state is not None and hasattr(runtime_state, "product_map"):
                runtime_state.product_map = context.product_cache

    # ------------------------------------------------------------------
    @property
    def _deterministic_broker_cls(self) -> type[DeterministicBroker]:  # type: ignore[override]
        return cast(type[DeterministicBroker], DeterministicBroker)

    @property
    def _create_brokerage(
        self,
    ) -> Callable[
        ..., tuple[IBrokerage, EventStore, MarketDataService, ProductCatalog]
    ]:  # type: ignore[override]
        return cast(
            Callable[
                ...,
                tuple["IBrokerage", "EventStore", "MarketDataService", "ProductCatalog"],
            ],
            create_brokerage,
        )

    @property
    def _risk_config_cls(self) -> type[RiskConfig]:  # type: ignore[override]
        return cast(type[RiskConfig], RiskConfig)

    @property
    def _risk_manager_cls(self) -> type[LiveRiskManager]:  # type: ignore[override]
        return cast(type[LiveRiskManager], LiveRiskManager)

    @property
    def _order_reconciler_cls(self) -> type[OrderReconciler]:  # type: ignore[override]
        return cast(type[OrderReconciler], OrderReconciler)


__all__ = [
    "BrokerBootstrapArtifacts",
    "BrokerBootstrapError",
    "RuntimeCoordinator",
    "RuntimeSettings",
    "load_runtime_settings",
    "emit_metric",
]
