"""Core initialization and infrastructure helpers for the execution coordinator."""

from __future__ import annotations

import inspect
from decimal import Decimal
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from bot_v2.config.types import Profile
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.orchestration.configuration.core import BotConfig
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.orchestration.service_registry import ServiceRegistry
from bot_v2.utilities import utc_now
from bot_v2.utilities.config import load_slippage_multipliers

from .logging_utils import logger

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from bot_v2.features.live_trade.liquidity_service import LiquidityService
    from bot_v2.orchestration.coordinators.base import CoordinatorContext, HealthStatus


class ExecutionCoordinatorCoreMixin:
    """Provide initialization, configuration, and event recording utilities."""

    def __init__(self, context: CoordinatorContext | None = None, **legacy_kwargs: Any) -> None:
        if context is None:
            context = self._build_context_from_legacy(**legacy_kwargs)
        super().__init__(context)
        self._order_reconciler = None
        self._config_controller = context.config_controller
        self._connection_down = False
        self._resubmission_pending = False
        self._last_exec_engine: Any | None = None
        self._settings = getattr(self.context.registry, "runtime_settings", None)
        if self._settings is None:
            self._settings = legacy_kwargs.get("settings")
        self.broker = getattr(self.context, "broker", None)

    @property
    def name(self) -> str:
        return "execution"

    @property
    def settings(self) -> Any | None:
        """Runtime settings accessor for legacy integrations."""
        return self._settings

    def update_settings(self, settings: Any) -> None:
        """Update runtime settings snapshot for integration compatibility."""
        self._settings = settings

    async def initialize_from_events(self) -> None:
        """Legacy compatibility hook for event log replay."""
        return None

    def _event_store(self) -> Any | None:
        return getattr(self.context, "event_store", None)

    def _record_event(self, event_type: str, data: dict[str, Any]) -> None:
        store = self._event_store()
        if store is None:
            return
        payload = dict(data)
        try:
            payload.setdefault("bot_id", getattr(self.context, "bot_id", None))
        except Exception:
            payload.setdefault("bot_id", None)
        if hasattr(store, "store_event"):
            try:
                store.store_event(event_type, payload)
                return
            except Exception:
                pass
        if hasattr(store, "append_error"):
            bot_id = str(payload.get("bot_id") or getattr(self.context, "bot_id", "execution"))
            try:
                store.append_error(bot_id, message=event_type, context=payload)
            except Exception:
                pass

    def _record_metric(
        self, metric_name: str, value: float, tags: dict[str, Any] | None = None
    ) -> None:
        store = self._event_store()
        if store is None or not hasattr(store, "store_metric"):
            return
        metric_tags = {k: str(v) for k, v in (tags or {}).items()}
        try:
            store.store_metric(metric_name, float(value), metric_tags)
        except Exception:
            pass

    @staticmethod
    async def _await_if_needed(value: Any) -> Any:
        """Await coroutine-like values returned by integration doubles."""
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _classify_broker_error(exc: Exception) -> str:
        message = str(exc).lower()
        if isinstance(exc, TimeoutError) or "timeout" in message:
            return "network_timeout"
        if "rate" in message and "limit" in message:
            return "api_error"
        if "auth" in message or "credential" in message:
            return "security_error"
        if "maintenance" in message:
            return "broker_maintenance"
        if "connection" in message:
            return "broker_connection_error"
        return "broker_error"

    @staticmethod
    def _build_context_from_legacy(**kwargs: Any) -> CoordinatorContext:
        from bot_v2.orchestration.coordinators.base import CoordinatorContext

        settings = kwargs.get("settings")
        broker = kwargs.get("broker")
        event_store = kwargs.get("event_store")
        risk_manager = kwargs.get("risk_manager")
        config = kwargs.get("config")
        if config is None:
            config = BotConfig(profile=Profile.DEMO)
        registry = ServiceRegistry(
            config=config,
            event_store=event_store,
            risk_manager=risk_manager,
            broker=broker,
            runtime_settings=settings,
        )
        runtime_state = kwargs.get("runtime_state")
        if runtime_state is None:
            runtime_state = SimpleNamespace()
        return CoordinatorContext(
            config=config,
            registry=registry,
            event_store=event_store,
            broker=broker,
            risk_manager=risk_manager,
            runtime_state=runtime_state,
        )

    def _record_broker_error(
        self, exc: Exception, *, symbol: str | None = None, order_id: str | None = None
    ) -> None:
        event_type = self._classify_broker_error(exc)
        payload: dict[str, Any] = {"error": str(exc)}
        if symbol:
            payload["symbol"] = symbol
        if order_id:
            payload["order_id"] = order_id

        event_types = ["error", "broker_error"]
        if event_type not in event_types:
            event_types.append(event_type)
        event_types.append("system_error")

        for et in event_types:
            self._record_event(et, payload)

        if event_type == "network_timeout":
            self._record_event("cleanup_initiated", payload)
        elif event_type == "api_error":
            self._record_event("retry_attempt", payload)
        elif event_type == "broker_connection_error":
            self._connection_down = True
        else:
            self._resubmission_pending = True

        tags: dict[str, Any] = {"type": event_type}
        if symbol:
            tags["symbol"] = symbol
        self._record_metric("error_count", 1.0, tags)
        self._record_metric("broker_error_rate", 1.0, {"symbol": symbol or "unknown"})

    def update_context(self, context: CoordinatorContext) -> None:
        previous = self.context
        super().update_context(context)
        self._config_controller = context.config_controller
        if (
            previous.broker is not context.broker
            or previous.orders_store is not context.orders_store
            or previous.event_store is not context.event_store
        ):
            self._order_reconciler = None

    def initialize(self, context: CoordinatorContext | None = None) -> CoordinatorContext:
        ctx = context or self.context
        broker = ctx.broker
        risk_manager = ctx.risk_manager
        runtime_state = ctx.runtime_state

        if broker is None or risk_manager is None or runtime_state is None:
            logger.warning(
                "Execution initialization skipped: missing broker or risk manager",
                operation="execution_init",
                stage="dependencies",
            )
            return ctx

        slippage_multipliers = load_slippage_multipliers()
        live_slippage = (
            {symbol: float(mult) for symbol, mult in slippage_multipliers.items()}
            if slippage_multipliers
            else None
        )

        risk_config = getattr(risk_manager, "config", None)
        use_advanced = self._should_use_advanced(risk_config)
        if use_advanced:
            try:
                impact_estimator = self._build_impact_estimator(ctx)
                risk_manager.set_impact_estimator(impact_estimator)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning(
                    "Failed to initialize LiquidityService impact estimator",
                    exc_info=True,
                    operation="execution_init",
                    stage="impact_estimator",
                    error=str(exc),
                )

        registry = ctx.registry
        runtime_settings = registry.runtime_settings
        if not isinstance(runtime_settings, RuntimeSettings):
            runtime_settings = load_runtime_settings()
            registry = registry.with_updates(runtime_settings=runtime_settings)
            ctx = ctx.with_updates(registry=registry)

        if use_advanced:
            runtime_state.exec_engine = AdvancedExecutionEngine(
                broker=broker,
                risk_manager=risk_manager,
                slippage_multipliers=slippage_multipliers,
            )
            logger.info(
                "Initialized AdvancedExecutionEngine with dynamic sizing integration",
                operation="execution_init",
                stage="engine",
                engine="advanced",
            )
        else:
            runtime_state.exec_engine = LiveExecutionEngine(
                broker=broker,
                risk_manager=risk_manager,
                event_store=ctx.event_store,
                bot_id=ctx.bot_id,
                slippage_multipliers=live_slippage,
                enable_preview=getattr(ctx.config, "enable_order_preview", False),
                settings=runtime_settings,
            )
            logger.info(
                "Initialized LiveExecutionEngine with risk integration",
                operation="execution_init",
                stage="engine",
                engine="live",
            )

        extras = dict(ctx.registry.extras)
        extras["execution_engine"] = runtime_state.exec_engine
        registry = registry.with_updates(extras=extras)
        self._order_reconciler = None
        return ctx.with_updates(registry=registry)

    def health_check(self) -> HealthStatus:
        runtime_state = self.context.runtime_state
        exec_engine = getattr(runtime_state, "exec_engine", None) if runtime_state else None
        order_stats = getattr(runtime_state, "order_stats", {}) if runtime_state else {}

        from bot_v2.orchestration.coordinators.base import HealthStatus

        return HealthStatus(
            healthy=exec_engine is not None,
            component=self.name,
            details={
                "has_execution_engine": exec_engine is not None,
                "order_stats": dict(order_stats),
                "background_tasks": len(self._background_tasks),
            },
        )

    @staticmethod
    def _should_use_advanced(risk_config: Any) -> bool:
        if risk_config is None:
            return False
        if getattr(risk_config, "enable_dynamic_position_sizing", False):
            return True
        if getattr(risk_config, "enable_market_impact_guard", False):
            return True
        return False

    def _build_impact_estimator(self, context: CoordinatorContext) -> Any:
        from bot_v2.features.live_trade.liquidity_service import LiquidityService

        broker = context.broker
        liquidity_service: LiquidityService = LiquidityService()

        def _impact_estimator(req: Any) -> Any:
            quote = None
            if broker is not None:
                try:
                    quote = broker.get_quote(req.symbol)
                except Exception:
                    quote = None

            bids: list[tuple[Decimal, Decimal]]
            asks: list[tuple[Decimal, Decimal]]

            seeded_orderbooks = getattr(broker, "order_books", None) if broker else None
            if seeded_orderbooks and req.symbol in seeded_orderbooks:
                seeded = seeded_orderbooks[req.symbol]
                bids = [(Decimal(str(p)), Decimal(str(s))) for p, s in seeded[0]]
                asks = [(Decimal(str(p)), Decimal(str(s))) for p, s in seeded[1]]
            else:
                mid = None
                if quote is not None and getattr(quote, "last", None) is not None:
                    mid = Decimal(str(quote.last))
                elif (
                    quote is not None
                    and getattr(quote, "bid", None) is not None
                    and getattr(quote, "ask", None) is not None
                ):
                    mid = (Decimal(str(quote.bid)) + Decimal(str(quote.ask))) / Decimal("2")
                if mid is None:
                    mid = Decimal("100")

                tick = None
                if (
                    quote is not None
                    and getattr(quote, "ask", None) is not None
                    and getattr(quote, "bid", None) is not None
                ):
                    spread = Decimal(str(quote.ask)) - Decimal(str(quote.bid))
                    if spread > 0:
                        tick = spread / Decimal("2")
                if tick is None or tick == 0:
                    tick = mid * Decimal("0.0005")

                depth_size = max(Decimal("1000"), abs(Decimal(str(req.quantity))) * Decimal("20"))
                bids = [(mid - tick * Decimal(i + 1), depth_size) for i in range(5)]
                asks = [(mid + tick * Decimal(i + 1), depth_size) for i in range(5)]

            liquidity_service.analyze_order_book(
                req.symbol,
                bids=bids,
                asks=asks,
                timestamp=utc_now(),
            )
            return liquidity_service.estimate_market_impact(
                symbol=req.symbol,
                side=req.side,
                quantity=Decimal(str(req.quantity)),
                book_data=(bids, asks),
            )

        return _impact_estimator

    def _update_balance(self, exec_engine: Any) -> None:
        broker = self.context.broker
        if exec_engine is None and broker is None:
            return
        if hasattr(exec_engine, "state_collector"):
            exec_engine.state_collector.collect_account_state()
        elif broker is not None and hasattr(broker, "list_balances"):
            broker.list_balances()


__all__ = ["ExecutionCoordinatorCoreMixin"]
