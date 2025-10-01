from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

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
from bot_v2.utilities.quantities import quantity_from

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


class ExecutionCoordinator:
    """Coordinates execution engine interactions and order placement."""

    def __init__(self, bot: PerpsBot) -> None:
        self._bot = bot
        self._order_reconciler: OrderReconciler | None = None

    def init_execution(self) -> None:
        import os

        bot = self._bot

        slippage_env = os.environ.get("SLIPPAGE_MULTIPLIERS", "")
        slippage_map: dict[str, float] = {}
        if slippage_env:
            try:
                parts = [p for p in slippage_env.split(",") if ":" in p]
                for part in parts:
                    k, v = part.split(":", 1)
                    slippage_map[k.strip()] = float(v)
            except Exception as exc:
                logger.warning(
                    "Invalid SLIPPAGE_MULTIPLIERS entry '%s': %s", slippage_env, exc, exc_info=True
                )

        risk_config = getattr(bot.risk_manager, "config", None)
        use_advanced = False
        if risk_config is not None:
            if getattr(risk_config, "enable_dynamic_position_sizing", False):
                use_advanced = True
            if getattr(risk_config, "enable_market_impact_guard", False):
                use_advanced = True

        if use_advanced:
            try:
                from bot_v2.features.live_trade.liquidity_service import LiquidityService

                liquidity_service = LiquidityService()

                def _impact_estimator(req: Any) -> Any:
                    try:
                        quote = bot.broker.get_quote(req.symbol)
                    except Exception:
                        quote = None

                    bids: list[tuple[Decimal, Decimal]]
                    asks: list[tuple[Decimal, Decimal]]

                    # Allow deterministic brokers to seed a custom order book
                    seeded_orderbooks = getattr(bot.broker, "order_books", None)
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

                        depth_size = max(
                            Decimal("1000"), abs(Decimal(str(req.quantity))) * Decimal("20")
                        )
                        bids = [(mid - tick * Decimal(i + 1), depth_size) for i in range(5)]
                        asks = [(mid + tick * Decimal(i + 1), depth_size) for i in range(5)]

                    liquidity_service.analyze_order_book(
                        req.symbol,
                        bids=bids,
                        asks=asks,
                        timestamp=datetime.utcnow(),
                    )
                    return liquidity_service.estimate_market_impact(
                        symbol=req.symbol,
                        side=req.side,
                        quantity=Decimal(str(req.quantity)),
                        book_data=(bids, asks),
                    )

                bot.risk_manager.set_impact_estimator(_impact_estimator)
            except Exception as exc:
                logger.warning(
                    "Failed to initialize LiquidityService impact estimator: %s",
                    exc,
                    exc_info=True,
                )
        use_advanced = False
        if risk_config is not None:
            if getattr(risk_config, "enable_dynamic_position_sizing", False):
                use_advanced = True
            if getattr(risk_config, "enable_market_impact_guard", False):
                use_advanced = True

        if use_advanced:
            bot.exec_engine = AdvancedExecutionEngine(
                broker=bot.broker,
                risk_manager=bot.risk_manager,
            )
            logger.info("Initialized AdvancedExecutionEngine with dynamic sizing integration")
        else:
            bot.exec_engine = LiveExecutionEngine(
                broker=bot.broker,
                risk_manager=bot.risk_manager,
                event_store=bot.event_store,
                bot_id="perps_bot",
                slippage_multipliers=slippage_map or None,
                enable_preview=bot.config.enable_order_preview,
            )
            logger.info("Initialized LiveExecutionEngine with risk integration")
        extras = dict(bot.registry.extras)
        extras["execution_engine"] = bot.exec_engine
        bot.registry = bot.registry.with_updates(extras=extras)

    async def execute_decision(
        self,
        symbol: str,
        decision: Any,
        mark: Decimal,
        product: Product,
        position_state: dict[str, Any] | None,
    ) -> None:
        bot = self._bot
        try:
            assert product is not None, "Missing product metadata"
            assert mark is not None and mark > 0, f"Invalid mark: {mark}"
            assert (
                position_state is None or "quantity" in position_state
            ), "Position state missing quantity"
            if bot.config.dry_run:
                logger.info(f"DRY RUN: Would execute {decision.action.value} for {symbol}")
                return

            position_quantity = quantity_from(position_state)
            if position_quantity is None:
                position_quantity = Decimal("0")

            if decision.action == Action.CLOSE:
                if not position_state or position_quantity == 0:
                    logger.warning(f"No position to close for {symbol}")
                    return
                order_quantity = abs(position_quantity)
            elif getattr(decision, "target_notional", None):
                order_quantity = Decimal(str(decision.target_notional)) / mark
            elif getattr(decision, "quantity", None) is not None:
                order_quantity = Decimal(str(decision.quantity))
            else:
                logger.warning(f"No quantity or notional in decision for {symbol}")
                return

            side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
            if decision.action == Action.CLOSE:
                side = (
                    OrderSide.SELL
                    if position_state and position_state.get("side") == "long"
                    else OrderSide.BUY
                )

            reduce_only_global = bot.is_reduce_only_mode()
            reduce_only = (
                decision.reduce_only or reduce_only_global or decision.action == Action.CLOSE
            )

            order_type = getattr(decision, "order_type", OrderType.MARKET)
            limit_price = getattr(decision, "limit_price", None)
            stop_price = getattr(decision, "stop_trigger", None)
            tif = getattr(decision, "time_in_force", None)
            try:
                if isinstance(tif, str):
                    tif = TimeInForce[tif.upper()]
                elif tif is None and isinstance(bot.config.time_in_force, str):
                    tif = TimeInForce[bot.config.time_in_force.upper()]
            except Exception:
                tif = None

            normalised_order_type = (
                order_type
                if isinstance(order_type, OrderType)
                else (
                    OrderType[order_type.upper()]
                    if isinstance(order_type, str)
                    else OrderType.MARKET
                )
            )

            exec_engine = bot.exec_engine
            place_kwargs: dict[str, Any] = {
                "symbol": symbol,
                "side": side,
                "quantity": order_quantity,
                "order_type": normalised_order_type,
                "reduce_only": reduce_only,
                "leverage": decision.leverage,
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

            order = await self._place_order(exec_engine, **place_kwargs)

            if order:
                logger.info(f"Order placed successfully: {order.id}")
            else:
                logger.warning(f"Order rejected or failed for {symbol}")
        except Exception as e:
            logger.error(f"Error executing decision for {symbol}: {e}")

    def _ensure_order_lock(self) -> asyncio.Lock:
        bot = self._bot
        if bot._order_lock is None:
            try:
                bot._order_lock = asyncio.Lock()
            except RuntimeError as exc:
                logger.error("Unable to initialize async order lock: %s", exc)
                raise
        return bot._order_lock

    async def _place_order(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        try:
            lock = self._ensure_order_lock()
            async with lock:
                return await self._place_order_inner(exec_engine, **kwargs)
        except (ValidationError, RiskValidationError, ExecutionError) as e:
            logger.warning(f"Order validation/execution failed: {e}")
            self._bot.order_stats["failed"] += 1
            raise
        except Exception as e:
            logger.error(f"Failed to place order: {e}", exc_info=True)
            self._bot.order_stats["failed"] += 1
            return None

    async def _place_order_inner(self, exec_engine: Any, **kwargs: Any) -> Order | None:
        bot = self._bot
        bot.order_stats["attempted"] += 1

        def _place() -> Any:
            return exec_engine.place_order(**kwargs)

        result = await asyncio.to_thread(_place)

        if isinstance(exec_engine, AdvancedExecutionEngine):
            order = result
        else:
            order = None
            if result:
                order = await asyncio.to_thread(bot.broker.get_order, result)

        if order:
            bot.orders_store.upsert(order)
            bot.order_stats["successful"] += 1
            order_quantity = quantity_from(order)
            # Normalize side to handle both enum and string
            side_str = getattr(order.side, "value", order.side)
            logger.info(f"Order recorded: {order.id} {side_str} {order_quantity} {order.symbol}")
            return order

        bot.order_stats["failed"] += 1
        logger.warning("Order attempt failed (no order returned)")
        return None

    def _get_order_reconciler(self) -> OrderReconciler:
        if self._order_reconciler is None:
            bot = self._bot
            self._order_reconciler = OrderReconciler(
                broker=bot.broker,
                orders_store=bot.orders_store,
                event_store=bot.event_store,
                bot_id=bot.bot_id,
            )
        return self._order_reconciler

    def reset_order_reconciler(self) -> None:
        self._order_reconciler = None

    async def run_runtime_guards(self) -> None:
        await self._run_runtime_guards_loop()

    async def _run_runtime_guards_loop(self) -> None:
        bot = self._bot
        while bot.running:
            try:
                await asyncio.to_thread(bot.exec_engine.run_runtime_guards)
            except Exception as e:
                logger.error(f"Error in runtime guards: {e}", exc_info=True)
            await asyncio.sleep(60)

    async def run_order_reconciliation(self, interval_seconds: int = 45) -> None:
        await self._run_order_reconciliation_loop(interval_seconds=interval_seconds)

    async def _run_order_reconciliation_loop(self, interval_seconds: int = 45) -> None:
        bot = self._bot
        while bot.running:
            try:
                reconciler = self._get_order_reconciler()
                await self._run_order_reconciliation_cycle(reconciler)
            except Exception as e:
                logger.debug(f"Order reconciliation error: {e}", exc_info=True)
            await asyncio.sleep(interval_seconds)

    async def _run_order_reconciliation_cycle(self, reconciler: OrderReconciler) -> None:
        bot = self._bot
        local_open = reconciler.fetch_local_open_orders()
        exchange_open = await reconciler.fetch_exchange_open_orders()

        if len(local_open) != len(exchange_open):
            logger.info(
                "Order count mismatch: local=%s exchange=%s",
                len(local_open),
                len(exchange_open),
            )

        diff = reconciler.diff_orders(local_open, exchange_open)
        if diff.missing_on_exchange or diff.missing_locally:
            await reconciler.reconcile_missing_on_exchange(diff)
            reconciler.reconcile_missing_locally(diff)

        for order in exchange_open.values():
            try:
                bot.orders_store.upsert(order)
            except Exception as exc:
                logger.debug(
                    "Failed to upsert exchange order %s during reconciliation: %s",
                    order.id,
                    exc,
                    exc_info=True,
                )

        await reconciler.record_snapshot(local_open, exchange_open)
