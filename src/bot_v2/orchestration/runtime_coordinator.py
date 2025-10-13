"""Legacy runtime coordinator facade wrapping the coordinator package."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskRuntimeState
from bot_v2.orchestration.broker_factory import create_brokerage
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
from bot_v2.utilities.telemetry import emit_metric

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
    from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.orchestration.perps_bot import PerpsBot
    from bot_v2.persistence.event_store import EventStore
    from bot_v2.persistence.orders_store import OrdersStore

logger = logging.getLogger(__name__)


class RuntimeCoordinator(_RuntimeCoordinator):
    """Compatibility layer for existing PerpsBot runtime coordination."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot
        context = self._build_context(bot)
        super().__init__(
            context,
            config_controller=getattr(bot, "config_controller", None),
            strategy_orchestrator=getattr(bot, "strategy_orchestrator", None),
            execution_coordinator=getattr(bot, "execution_coordinator", None),
            product_cache=getattr(bot, "_product_map", None),
        )

    # ------------------------------------------------------------------
    def bootstrap(self) -> None:
        self._refresh_context_from_bot()
        super().bootstrap()
        self._sync_bot(self.context)

    def _init_broker(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self._refresh_context_from_bot()
        updated = super()._init_broker(ctx)
        self.update_context(updated)
        return updated

    def _init_risk_manager(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self._refresh_context_from_bot()
        updated = super()._init_risk_manager(ctx)
        self.update_context(updated)
        return updated

    def _validate_broker_environment(self) -> None:  # type: ignore[override]
        ctx = self._refresh_context_from_bot()
        super()._validate_broker_environment(ctx)

    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:  # type: ignore[override]
        self._refresh_context_from_bot()
        super().set_reduce_only_mode(enabled, reason)
        self._sync_bot(self.context)

    def is_reduce_only_mode(self) -> bool:  # type: ignore[override]
        self._refresh_context_from_bot()
        return bool(super().is_reduce_only_mode())

    def on_risk_state_change(self, state: RiskRuntimeState) -> None:  # type: ignore[override]
        self._refresh_context_from_bot()
        super().on_risk_state_change(state)
        self._sync_bot(self.context)

    async def reconcile_state_on_startup(self) -> None:  # type: ignore[override]
        context = self._refresh_context_from_bot()
        if context.broker is None:
            fallback_broker = None
            try:
                fallback_broker = getattr(self._bot, "broker")
            except Exception:
                fallback_broker = None
            if fallback_broker is not None:
                context = context.with_updates(broker=cast("IBrokerage", fallback_broker))
                super().update_context(context)
        await super().reconcile_state_on_startup()
        self._sync_bot(self.context)

    # ------------------------------------------------------------------
    def update_context(self, context: CoordinatorContext) -> None:  # type: ignore[override]
        super().update_context(context)
        self._sync_bot(context)

    # ------------------------------------------------------------------
    def _refresh_context_from_bot(self) -> CoordinatorContext:
        updated = self._build_context(self._bot)
        super().update_context(updated)
        return self.context

    def _build_context(self, bot: PerpsBot) -> CoordinatorContext:
        symbols = tuple(getattr(bot.config, "symbols", None) or ())
        product_cache = getattr(bot._state, "product_map", None) if hasattr(bot, "_state") else None
        registry = bot.registry

        def _safe_attr(obj: object, name: str) -> object | None:
            try:
                return cast(object, getattr(obj, name))
            except Exception:
                return None

        broker_attr = (
            getattr(registry, "broker", None)
            or bot.__dict__.get("broker")
            or _safe_attr(bot, "broker")
        )
        risk_attr = (
            getattr(registry, "risk_manager", None)
            or bot.__dict__.get("risk_manager")
            or _safe_attr(bot, "risk_manager")
        )
        broker_value = cast("IBrokerage | None", broker_attr)
        risk_value = cast("LiveRiskManager | None", risk_attr)
        event_store_value = cast("EventStore | None", getattr(bot, "event_store", None))
        orders_store_value = cast("OrdersStore | None", getattr(bot, "orders_store", None))
        return CoordinatorContext(
            config=bot.config,
            registry=registry,
            event_store=event_store_value,
            orders_store=orders_store_value,
            broker=broker_value,
            risk_manager=risk_value,
            symbols=symbols,
            bot_id=getattr(bot, "bot_id", "perps_bot"),
            runtime_state=getattr(bot, "runtime_state", None),
            config_controller=getattr(bot, "config_controller", None),
            strategy_orchestrator=getattr(bot, "strategy_orchestrator", None),
            execution_coordinator=getattr(bot, "execution_coordinator", None),
            product_cache=product_cache,
        )

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
