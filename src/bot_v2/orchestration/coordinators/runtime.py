"""Runtime coordinator responsible for broker and risk bootstrapping."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from bot_v2.config.live_trade_config import RiskConfig
from bot_v2.features.live_trade.risk import LiveRiskManager, RiskRuntimeState
from bot_v2.orchestration.broker_factory import create_brokerage
from bot_v2.orchestration.configuration import DEFAULT_SPOT_RISK_PATH, Profile
from bot_v2.orchestration.deterministic_broker import DeterministicBroker
from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.telemetry import emit_metric

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.brokerages.coinbase.market_data_service import MarketDataService
    from bot_v2.features.brokerages.coinbase.utilities import ProductCatalog
    from bot_v2.features.brokerages.core.interfaces import IBrokerage
    from bot_v2.persistence.event_store import EventStore

from .base import BaseCoordinator, CoordinatorContext

logger = get_logger(__name__, component="runtime_coordinator")


class BrokerBootstrapError(RuntimeError):
    """Raised when broker initialization fails."""


@dataclass
class BrokerBootstrapArtifacts:
    broker: object
    registry_updates: dict[str, Any]
    event_store: object | None = None
    products: Sequence[object] = ()


class RuntimeCoordinator(BaseCoordinator):
    """Handles broker and risk bootstrapping plus runtime safety toggles."""

    def __init__(
        self,
        context: CoordinatorContext,
        *,
        config_controller: Any | None = None,
        strategy_orchestrator: Any | None = None,
        execution_coordinator: Any | None = None,
        product_cache: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(context)
        self._config_controller = config_controller or context.config_controller
        self._strategy_orchestrator = strategy_orchestrator or context.strategy_orchestrator
        self._execution_coordinator = execution_coordinator or context.execution_coordinator
        self._product_cache = product_cache or context.product_cache

    @property
    def name(self) -> str:
        return "runtime"

    def update_context(self, context: CoordinatorContext) -> None:
        super().update_context(context)
        if context.config_controller is not None:
            self._config_controller = context.config_controller
        if context.strategy_orchestrator is not None:
            self._strategy_orchestrator = context.strategy_orchestrator
        if context.execution_coordinator is not None:
            self._execution_coordinator = context.execution_coordinator
        if context.product_cache is not None:
            self._product_cache = context.product_cache

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self.context

        broker_ctx = self._init_broker(ctx)
        if isinstance(broker_ctx, CoordinatorContext):
            ctx = broker_ctx
        self.update_context(ctx)
        ctx = self.context

        risk_ctx = self._init_risk_manager(ctx)
        if isinstance(risk_ctx, CoordinatorContext):
            ctx = risk_ctx
        self.update_context(ctx)
        ctx = self.context

        return self.context

    def bootstrap(self) -> None:
        """Compatibility wrapper to align with legacy API."""

        self.initialize(self.context)

    def _init_broker(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self.context
        registry = ctx.registry

        if registry.broker is not None:
            broker = registry.broker
            ctx = ctx.with_updates(broker=broker)
            return ctx

        try:
            if self._should_use_mock_broker(ctx):
                artifacts = self._build_mock_broker()
                updated_ctx = ctx
            else:
                artifacts, updated_ctx = self._build_real_broker(ctx)
        except Exception as exc:  # pragma: no cover - fatal boot failure
            logger.error(
                "Failed to initialize broker",
                error=str(exc),
                operation="broker_bootstrap",
                stage="init",
                exc_info=True,
            )
            raise BrokerBootstrapError("Broker initialization failed") from exc

        return self._apply_broker_bootstrap(updated_ctx, artifacts)

    def _should_use_mock_broker(self, context: CoordinatorContext) -> bool:
        config = context.config
        paper_env = bool(getattr(config, "perps_paper_trading", False))
        force_mock = bool(getattr(config, "perps_force_mock", False))
        is_dev = config.profile == Profile.DEV
        return paper_env or force_mock or is_dev or bool(getattr(config, "mock_broker", False))

    def _build_mock_broker(self) -> BrokerBootstrapArtifacts:
        broker = self._deterministic_broker_cls()
        logger.info(
            "Using deterministic broker (REST-first marks)",
            operation="broker_bootstrap",
            stage="mock",
        )
        return BrokerBootstrapArtifacts(broker=broker, registry_updates={"broker": broker})

    def _build_real_broker(
        self, context: CoordinatorContext
    ) -> tuple[BrokerBootstrapArtifacts, CoordinatorContext]:
        settings, ctx = self._resolve_settings(context)
        self._validate_broker_environment(ctx, settings)

        registry = ctx.registry
        broker, event_store, market_data, product_catalog = self._create_brokerage(
            registry,
            event_store=ctx.event_store,
            market_data=registry.market_data_service,
            product_catalog=registry.product_catalog,
            settings=settings,
        )
        if not broker.connect():
            raise RuntimeError("Failed to connect to broker")
        products = broker.list_products()
        logger.info(
            "Connected to broker",
            products=len(products),
            operation="broker_bootstrap",
            stage="connect",
        )

        artifacts = BrokerBootstrapArtifacts(
            broker=broker,
            event_store=event_store,
            registry_updates={
                "broker": broker,
                "event_store": event_store,
                "market_data_service": market_data,
                "product_catalog": product_catalog,
            },
            products=products,
        )
        return artifacts, ctx

    def _resolve_settings(
        self, context: CoordinatorContext
    ) -> tuple[RuntimeSettings, CoordinatorContext]:
        registry = context.registry
        settings = registry.runtime_settings
        if isinstance(settings, RuntimeSettings):
            return settings, context

        resolved = load_runtime_settings()
        registry = registry.with_updates(runtime_settings=resolved)
        context = context.with_updates(registry=registry)
        return resolved, context

    def _apply_broker_bootstrap(
        self, context: CoordinatorContext, artifacts: BrokerBootstrapArtifacts
    ) -> CoordinatorContext:
        registry = context.registry.with_updates(**artifacts.registry_updates)
        event_store = (
            artifacts.event_store if artifacts.event_store is not None else context.event_store
        )

        ctx = context.with_updates(
            broker=artifacts.broker,
            registry=registry,
            event_store=event_store,
        )

        self._hydrate_product_cache(artifacts.products)
        if self._product_cache is not None:
            ctx = ctx.with_updates(product_cache=self._product_cache)

        return ctx

    def _hydrate_product_cache(self, products: Sequence[object]) -> None:
        if not products:
            return
        if not isinstance(self._product_cache, dict):
            self._product_cache = {}
        for product in products:
            symbol = getattr(product, "symbol", None)
            if symbol:
                self._product_cache[symbol] = product

    def _validate_broker_environment(
        self, context: CoordinatorContext, settings: RuntimeSettings | None = None
    ) -> None:
        config = context.config
        if settings is None:
            settings, updated = self._resolve_settings(context)
            context = updated

        if self._should_use_mock_broker(context):
            logger.info(
                "Paper/mock mode enabled — skipping production env checks",
                operation="broker_env_validate",
                stage="mock",
            )
            return

        broker_hint = settings.broker_hint or ""
        if broker_hint != "coinbase":
            raise RuntimeError("BROKER must be set to 'coinbase' for perps trading")

        if settings.coinbase_sandbox_enabled:
            raise RuntimeError(
                "COINBASE_SANDBOX=1 is not supported for live trading. "
                "Remove it or enable PERPS_PAPER=1."
            )

        derivatives_enabled = bool(getattr(config, "derivatives_enabled", False))
        symbols = context.symbols or ()
        if not derivatives_enabled:
            for sym in symbols:
                if sym.upper().endswith("-PERP"):
                    raise RuntimeError(
                        f"Symbol {sym} is perpetual but COINBASE_ENABLE_DERIVATIVES is not enabled."
                    )

        raw_env = settings.raw_env
        api_mode = settings.coinbase_api_mode
        if api_mode != "advanced":
            raise RuntimeError(
                "Perpetuals require Advanced Trade API in production. "
                "Set COINBASE_API_MODE=advanced and unset COINBASE_SANDBOX, "
                "or set PERPS_PAPER=1 for mock mode."
            )

        cdp_key = raw_env.get("COINBASE_PROD_CDP_API_KEY") or raw_env.get("COINBASE_CDP_API_KEY")
        cdp_priv = raw_env.get("COINBASE_PROD_CDP_PRIVATE_KEY") or raw_env.get(
            "COINBASE_CDP_PRIVATE_KEY"
        )
        if derivatives_enabled and not (cdp_key and cdp_priv):
            raise RuntimeError(
                "Missing CDP JWT credentials. Set COINBASE_PROD_CDP_API_KEY and "
                "COINBASE_PROD_CDP_PRIVATE_KEY, or enable PERPS_PAPER=1 for mock trading."
            )

        api_key_present = any(
            raw_env.get(env) for env in ("COINBASE_API_KEY", "COINBASE_PROD_API_KEY")
        )
        api_secret_present = any(
            raw_env.get(env) for env in ("COINBASE_API_SECRET", "COINBASE_PROD_API_SECRET")
        )

        if not (api_key_present and api_secret_present):
            raise RuntimeError(
                "Spot trading requires Coinbase production API key/secret. "
                "Set COINBASE_API_KEY/SECRET (or PROD variants)."
            )

    def _init_risk_manager(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self.context
        controller = self._config_controller

        if ctx.registry.risk_manager is not None:
            risk_manager = ctx.registry.risk_manager
            risk_manager.set_state_listener(self.on_risk_state_change)
            if controller is not None:
                controller.sync_with_risk_manager(risk_manager)
                risk_manager.set_reduce_only_mode(controller.reduce_only_mode, reason="config_init")
            return ctx.with_updates(risk_manager=risk_manager)

        settings, ctx = self._resolve_settings(ctx)
        env_risk_config_path = settings.risk_config_path
        resolved_risk_path = str(env_risk_config_path) if env_risk_config_path else None

        config_profile = ctx.config.profile
        if not resolved_risk_path and config_profile in {Profile.SPOT, Profile.DEV, Profile.DEMO}:
            if DEFAULT_SPOT_RISK_PATH.exists():
                resolved_risk_path = str(DEFAULT_SPOT_RISK_PATH)
                logger.info(
                    "Loading spot risk profile",
                    path=resolved_risk_path,
                    operation="risk_config",
                    stage="load",
                )

        path_obj: Path | None = Path(resolved_risk_path) if resolved_risk_path else None
        try:
            if path_obj and path_obj.exists():
                risk_config = self._risk_config_cls.from_json(str(path_obj))
            else:
                risk_config = self._risk_config_cls.from_env()
        except Exception:
            logger.exception(
                "Failed to load risk config; using defaults",
                operation="risk_config",
                stage="load",
            )
            risk_config = self._risk_config_cls()

        try:
            if getattr(ctx.config, "max_leverage", None):
                risk_config.max_leverage = int(ctx.config.max_leverage)
        except Exception as exc:
            logger.warning(
                "Failed to apply max leverage override",
                error=str(exc),
                exc_info=True,
                operation="risk_config",
                stage="override",
            )
        try:
            risk_config.reduce_only_mode = bool(ctx.config.reduce_only_mode)
        except Exception as exc:
            logger.warning(
                "Failed to sync reduce-only override into risk config",
                error=str(exc),
                exc_info=True,
                operation="risk_config",
                stage="override",
            )

        risk_manager = self._risk_manager_cls(config=risk_config, event_store=ctx.event_store)
        broker = ctx.broker or ctx.registry.broker
        try:
            if broker is not None and hasattr(broker, "get_position_risk"):
                risk_manager.set_risk_info_provider(broker.get_position_risk)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(
                "Failed to set broker risk info provider",
                error=str(exc),
                exc_info=True,
                operation="risk_config",
                stage="risk_provider",
            )

        risk_manager.set_state_listener(self.on_risk_state_change)
        if controller is not None:
            controller.sync_with_risk_manager(risk_manager)
            risk_manager.set_reduce_only_mode(controller.reduce_only_mode, reason="config_init")

        registry = ctx.registry.with_updates(risk_manager=risk_manager)
        return ctx.with_updates(risk_manager=risk_manager, registry=registry)

    def set_reduce_only_mode(self, enabled: bool, reason: str) -> None:
        controller = self._config_controller
        risk_manager = self.context.risk_manager
        if controller is None:
            logger.debug(
                "No config controller available to toggle reduce-only mode",
                operation="reduce_only_toggle",
                stage="controller_missing",
            )
            return
        if not controller.set_reduce_only_mode(enabled, reason=reason, risk_manager=risk_manager):
            return
        logger.warning(
            "Reduce-only mode %s (%s)",
            "enabled" if enabled else "disabled",
            reason,
            operation="reduce_only_toggle",
            stage="set",
            enabled=bool(enabled),
            reason=reason,
        )
        self._emit_reduce_only_metric(enabled, reason)

    def is_reduce_only_mode(self) -> bool:
        controller = self._config_controller
        if controller is None:
            return False
        return bool(controller.is_reduce_only_mode(self.context.risk_manager))

    def on_risk_state_change(self, state: RiskRuntimeState) -> None:
        controller = self._config_controller
        if controller is None:
            return
        reduce_only = bool(state.reduce_only_mode)
        if not controller.apply_risk_update(reduce_only):
            return
        reason = state.last_reduce_only_reason or "unspecified"
        logger.warning(
            "Risk manager toggled reduce-only mode",
            enabled=reduce_only,
            reason=reason,
            operation="reduce_only_toggle",
            stage="risk_update",
        )
        self._emit_reduce_only_metric(reduce_only, reason)

    def _emit_reduce_only_metric(self, enabled: bool, reason: str) -> None:
        event_store = self.context.event_store
        if event_store is None:
            return
        emit_metric(
            event_store,
            self.context.bot_id,
            {
                "event_type": "reduce_only_mode_changed",
                "enabled": enabled,
                "reason": reason,
            },
            logger=logger,
        )

    async def reconcile_state_on_startup(self) -> None:
        ctx = self.context
        config = ctx.config
        if config.dry_run or getattr(config, "perps_skip_startup_reconcile", False):
            logger.info(
                "Skipping startup reconciliation",
                reason="dry_run" if config.dry_run else "perps_skip_startup_reconcile",
                operation="startup_reconcile",
                stage="skip",
            )
            return

        broker = ctx.broker or ctx.registry.broker
        if broker is None:
            logger.info(
                "No broker available for reconciliation; skipping",
                operation="startup_reconcile",
                stage="skip",
            )
            return

        orders_store = ctx.orders_store
        event_store = ctx.event_store
        if orders_store is None or event_store is None:
            logger.warning(
                "Skipping reconciliation: missing orders/event store",
                operation="startup_reconcile",
                stage="skip",
            )
            return

        logger.info(
            "Reconciling state with exchange",
            operation="startup_reconcile",
            stage="begin",
        )
        try:
            reconciler = self._order_reconciler_cls(
                broker=broker,
                orders_store=orders_store,
                event_store=event_store,
                bot_id=ctx.bot_id,
            )

            local_open = reconciler.fetch_local_open_orders()
            exchange_open = await reconciler.fetch_exchange_open_orders()

            logger.info(
                "Reconciliation snapshot",
                local_open=len(local_open),
                exchange_open=len(exchange_open),
                operation="startup_reconcile",
                stage="snapshot",
            )
            await reconciler.record_snapshot(local_open, exchange_open)

            diff = reconciler.diff_orders(local_open, exchange_open)
            await reconciler.reconcile_missing_on_exchange(diff)
            reconciler.reconcile_missing_locally(diff)

            try:
                snapshot = await reconciler.snapshot_positions()
                if snapshot:
                    runtime_state = ctx.runtime_state
                    if runtime_state is not None:
                        runtime_state.last_positions = snapshot
            except Exception as exc:
                logger.debug(
                    "Failed to snapshot initial positions",
                    error=str(exc),
                    exc_info=True,
                    operation="startup_reconcile",
                    stage="positions",
                )

            logger.info(
                "State reconciliation complete",
                operation="startup_reconcile",
                stage="complete",
            )
        except Exception as exc:
            logger.error(
                "Failed to reconcile state on startup",
                error=str(exc),
                exc_info=True,
                operation="startup_reconcile",
                stage="error",
            )
            try:
                if ctx.event_store is not None:
                    ctx.event_store.append_error(
                        bot_id=ctx.bot_id,
                        message="startup_reconcile_failed",
                        context={"error": str(exc)},
                    )
            except Exception:
                logger.exception(
                    "Failed to persist startup reconciliation error",
                    operation="startup_reconcile",
                    stage="error_persist",
                )
            self.set_reduce_only_mode(True, reason="startup_reconcile_failed")

    @property
    def _deterministic_broker_cls(self) -> type[DeterministicBroker]:
        return cast(type[DeterministicBroker], DeterministicBroker)

    @property
    def _create_brokerage(
        self,
    ) -> Callable[..., tuple[IBrokerage, EventStore, MarketDataService, ProductCatalog]]:
        return cast(
            Callable[..., tuple["IBrokerage", "EventStore", "MarketDataService", "ProductCatalog"]],
            create_brokerage,
        )

    @property
    def _risk_config_cls(self) -> type[RiskConfig]:
        return cast(type[RiskConfig], RiskConfig)

    @property
    def _risk_manager_cls(self) -> type[LiveRiskManager]:
        return cast(type[LiveRiskManager], LiveRiskManager)

    @property
    def _order_reconciler_cls(self) -> type[OrderReconciler]:
        return cast(type[OrderReconciler], OrderReconciler)


__all__ = [
    "BrokerBootstrapArtifacts",
    "BrokerBootstrapError",
    "RuntimeCoordinator",
]
