"""Execution coordinator responsible for order placement and reconciliation."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from bot_v2.errors import ExecutionError, ValidationError
from bot_v2.features.brokerages.core.interfaces import (
    Order,
    OrderSide,
    OrderType,
    Product,
    TimeInForce,
)
from bot_v2.features.live_trade.advanced_execution import AdvancedExecutionEngine
from bot_v2.features.live_trade.risk import ValidationError as RiskValidationError
from bot_v2.features.live_trade.strategies.perps_baseline import Action
from bot_v2.orchestration.live_execution import LiveExecutionEngine
from bot_v2.orchestration.order_reconciler import OrderReconciler
from bot_v2.orchestration.runtime_settings import RuntimeSettings, load_runtime_settings
from bot_v2.utilities import utc_now
from bot_v2.utilities.async_utils import run_in_thread
from bot_v2.utilities.config import load_slippage_multipliers
from bot_v2.utilities.logging_patterns import get_logger
from bot_v2.utilities.quantities import quantity_from

from .base import BaseCoordinator, CoordinatorContext, HealthStatus

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.features.live_trade.liquidity_service import LiquidityService
    from bot_v2.orchestration.perps_bot_state import PerpsBotRuntimeState

logger = get_logger(__name__, component="execution_coordinator")


class ExecutionCoordinator(BaseCoordinator):
    """Coordinates execution engine interactions and background reconciliation."""

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self._order_reconciler: OrderReconciler | None = None
        self._config_controller = context.config_controller

    @property
    def name(self) -> str:
        return "execution"

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

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        ctx = self.context
        if ctx.config.dry_run or ctx.runtime_state is None:
            return []

        tasks: list[asyncio.Task[Any]] = []
        guards_task = asyncio.create_task(self._run_runtime_guards_loop())
        reconciliation_task = asyncio.create_task(self._run_order_reconciliation_loop())

        for task in (guards_task, reconciliation_task):
            self._register_background_task(task)
            tasks.append(task)

        return tasks

    def ensure_order_lock(self) -> asyncio.Lock:
        runtime_state_obj = self.context.runtime_state
        if runtime_state_obj is None:
            raise RuntimeError("Runtime state is unavailable; cannot create order lock")

        runtime_state = cast("PerpsBotRuntimeState", runtime_state_obj)

        if runtime_state.order_lock is None:
            try:
                runtime_state.order_lock = asyncio.Lock()
            except RuntimeError as exc:
                logger.error(
                    "Unable to initialize async order lock: %s",
                    exc,
                    operation="order_lock",
                    stage="initialize",
                )
                raise
        return cast(asyncio.Lock, runtime_state.order_lock)

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        ctx = self.context
        runtime_state_obj = ctx.runtime_state
        if runtime_state_obj is None:
            logger.debug(
                "Runtime state missing; skipping decision execution",
                symbol=symbol,
                operation="execution_decision",
                stage="runtime_state",
            )
            return
        runtime_state = cast("PerpsBotRuntimeState", runtime_state_obj)
        try:
            assert product is not None, "Missing product metadata"
            assert mark is not None and mark > 0, f"Invalid mark: {mark}"
            if position_state is not None and "quantity" not in position_state:
                raise AssertionError("Position state missing quantity")

            if ctx.config.dry_run:
                logger.info(
                    "DRY RUN: Would execute %s for %s",
                    decision.action.value,
                    symbol,
                    operation="execution_decision",
                    stage="dry_run",
                    symbol=symbol,
                    action=decision.action.value,
                )
                return

            position_quantity_raw = quantity_from(position_state)
            position_quantity = (
                position_quantity_raw
                if isinstance(position_quantity_raw, Decimal)
                else Decimal("0")
            )

            if decision.action == Action.CLOSE:
                if not position_state or position_quantity == 0:
                    logger.warning(
                        "No position to close for %s",
                        symbol,
                        operation="execution_decision",
                        stage="close",
                    )
                    return
                order_quantity = abs(position_quantity)
            elif getattr(decision, "target_notional", None):
                order_quantity = Decimal(str(decision.target_notional)) / mark
            elif getattr(decision, "quantity", None) is not None:
                order_quantity = Decimal(str(decision.quantity))
            else:
                logger.warning(
                    "No quantity or notional in decision for %s",
                    symbol,
                    operation="execution_decision",
                    stage="quantity",
                )
                return

            side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
            if decision.action == Action.CLOSE:
                side = (
                    OrderSide.SELL
                    if position_state and position_state.get("side") == "long"
                    else OrderSide.BUY
                )

            reduce_only_global = False
            if self._config_controller is not None:
                try:
                    reduce_only_global = bool(
                        self._config_controller.is_reduce_only_mode(ctx.risk_manager)
                    )
                except Exception:
                    reduce_only_global = False

            reduce_only = (
                getattr(decision, "reduce_only", False)
                or reduce_only_global
                or decision.action == Action.CLOSE
            )

            order_type = getattr(decision, "order_type", OrderType.MARKET)
            limit_price = getattr(decision, "limit_price", None)
            stop_price = getattr(decision, "stop_trigger", None)
            tif = getattr(decision, "time_in_force", None)
            try:
                if isinstance(tif, str):
                    tif = TimeInForce[tif.upper()]
                elif tif is None and isinstance(ctx.config.time_in_force, str):
                    tif = TimeInForce[ctx.config.time_in_force.upper()]
            except Exception:
                tif = None

            if isinstance(order_type, OrderType):
                normalised_order_type = order_type
            else:
                normalised_order_type = (
                    OrderType[order_type.upper()]
                    if isinstance(order_type, str)
                    else OrderType.MARKET
                )

            exec_engine = runtime_state.exec_engine
            if exec_engine is None:
                logger.warning(
                    "Execution engine not initialized; cannot place order for %s",
                    symbol,
                    operation="execution_decision",
                    stage="engine_missing",
                    symbol=symbol,
                )
                return

            place_kwargs: dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "quantity": order_quantity,
                "order_type": normalised_order_type,
                "reduce_only": reduce_only,
                "leverage": getattr(decision, "leverage", None),
            }

            if isinstance(exec_engine, AdvancedExecutionEngine):
                place_kwargs.update(
                    {
                        "limit_price": limit_price,
                        "stop_price": stop_price,
                        "time_in_force": tif or TimeInForce.GTC,
                    }
                )
            else:
                place_kwargs.update(
                    {
                        "product": product,
                        "price": limit_price,
                        "stop_price": stop_price,
                        "tif": tif or None,
                    }
                )

            order = await self.place_order(exec_engine, **place_kwargs)
            if order:
                logger.info(
                    "Order placed successfully",
                    order_id=str(order.id),
                    symbol=symbol,
                    operation="execution_decision",
                    stage="placed",
                )
            else:
                logger.warning(
                    "Order rejected or failed",
                    symbol=symbol,
                    operation="execution_decision",
                    stage="placed",
                )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Error executing decision",
                symbol=symbol,
                error=str(exc),
                exc_info=True,
                operation="execution_decision",
                stage="exception",
            )

    async def place_order(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        lock = self.ensure_order_lock()
        try:
            async with lock:
                return await self.place_order_inner(exec_engine, **kwargs)
        except (ValidationError, RiskValidationError, ExecutionError) as exc:
            logger.warning(
                "Order validation/execution failed",
                error=str(exc),
                operation="execution_order",
                stage="validation",
            )
            self._increment_order_stat("failed")
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Failed to place order",
                error=str(exc),
                exc_info=True,
                operation="execution_order",
                stage="submit_exception",
            )
            self._increment_order_stat("failed")
            return None

    async def place_order_inner(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        self._increment_order_stat("attempted")
        orders_store = self.context.orders_store
        broker = self.context.broker
        runtime_state_obj = self.context.runtime_state

        if runtime_state_obj is None:
            logger.debug(
                "Runtime state missing; cannot record order",
                operation="execution_order",
                stage="runtime_state",
            )
            return None

        def _place() -> Any:
            return exec_engine.place_order(**kwargs)

        result = await run_in_thread(_place)

        if isinstance(exec_engine, AdvancedExecutionEngine):
            order = cast(Order | None, result)
        else:
            order = None
            if result and broker is not None:
                order = await run_in_thread(broker.get_order, result)
                order = cast(Order | None, order)

        if order:
            if orders_store is not None:
                orders_store.upsert(order)
            self._increment_order_stat("successful")
            order_quantity_raw = quantity_from(order)
            order_quantity = (
                order_quantity_raw if isinstance(order_quantity_raw, Decimal) else Decimal("0")
            )
            logger.info(
                "Order recorded",
                order_id=str(order.id),
                side=order.side.value,
                quantity=float(order_quantity),
                symbol=order.symbol,
                operation="execution_order",
                stage="record",
            )
            return order

        self._increment_order_stat("failed")
        logger.warning(
            "Order attempt failed (no order returned)",
            operation="execution_order",
            stage="record",
        )
        return None

    def _get_order_reconciler(self) -> OrderReconciler:
        if self._order_reconciler is None:
            ctx = self.context
            broker = ctx.broker
            if broker is None or ctx.orders_store is None or ctx.event_store is None:
                raise RuntimeError(
                    "Cannot create OrderReconciler without broker, stores, and event log"
                )
            self._order_reconciler = OrderReconciler(
                broker=broker,
                orders_store=ctx.orders_store,
                event_store=ctx.event_store,
                bot_id=ctx.bot_id,
            )
        return self._order_reconciler

    def reset_order_reconciler(self) -> None:
        self._order_reconciler = None

    async def run_runtime_guards(self) -> None:
        await self._run_runtime_guards_loop()

    async def _run_runtime_guards_loop(self) -> None:
        try:
            while True:
                exec_engine = getattr(self.context.runtime_state, "exec_engine", None)
                if exec_engine is not None:
                    try:
                        await run_in_thread(exec_engine.run_runtime_guards)
                    except Exception as exc:  # pragma: no cover - defensive logging
                        logger.error(
                            "Error in runtime guards",
                            error=str(exc),
                            exc_info=True,
                            operation="runtime_guard_loop",
                            stage="run",
                        )
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise

    async def run_order_reconciliation(self, interval_seconds: int = 45) -> None:
        await self._run_order_reconciliation_loop(interval_seconds=interval_seconds)

    async def _run_order_reconciliation_loop(self, interval_seconds: int = 45) -> None:
        try:
            while True:
                try:
                    reconciler = self._get_order_reconciler()
                    await self._run_order_reconciliation_cycle(reconciler)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Order reconciliation error",
                        error=str(exc),
                        exc_info=True,
                        operation="order_reconcile_loop",
                        stage="run",
                    )
                await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            raise

    async def _run_order_reconciliation_cycle(self, reconciler: OrderReconciler) -> None:
        orders_store = self.context.orders_store

        local_open = reconciler.fetch_local_open_orders()
        exchange_open = await reconciler.fetch_exchange_open_orders()

        if len(local_open) != len(exchange_open):
            logger.info(
                "Order count mismatch",
                local=len(local_open),
                exchange=len(exchange_open),
                operation="order_reconcile",
                stage="diff",
            )

        diff = reconciler.diff_orders(local_open, exchange_open)
        if diff.missing_on_exchange or diff.missing_locally:
            await reconciler.reconcile_missing_on_exchange(diff)
            reconciler.reconcile_missing_locally(diff)

        if orders_store is not None:
            for order in exchange_open.values():
                try:
                    orders_store.upsert(order)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.debug(
                        "Failed to upsert exchange order during reconciliation",
                        order_id=order.id,
                        error=str(exc),
                        exc_info=True,
                        operation="order_reconcile",
                        stage="upsert",
                    )

        await reconciler.record_snapshot(local_open, exchange_open)

    def health_check(self) -> HealthStatus:
        runtime_state = self.context.runtime_state
        exec_engine = getattr(runtime_state, "exec_engine", None) if runtime_state else None
        order_stats = getattr(runtime_state, "order_stats", {}) if runtime_state else {}

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

    def _increment_order_stat(self, key: str) -> None:
        runtime_state = self.context.runtime_state
        if runtime_state is None:
            return
        stats = getattr(runtime_state, "order_stats", None)
        if isinstance(stats, dict):
            stats[key] = stats.get(key, 0) + 1


__all__ = ["ExecutionCoordinator"]
