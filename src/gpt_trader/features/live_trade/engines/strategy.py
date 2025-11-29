"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.

State Recovery:
On startup, reads `price_tick` events from EventStore to restore price history.
During operation, persists price ticks to EventStore for crash recovery.
"""

import asyncio
import logging
import time
from collections import defaultdict
from decimal import Decimal
from typing import Any

from gpt_trader.core import OrderSide, OrderType, Position, Product
from gpt_trader.features.live_trade.engines.base import (
    BaseEngine,
    CoordinatorContext,
    HealthStatus,
)
from gpt_trader.features.live_trade.factory import create_strategy
from gpt_trader.features.live_trade.risk.manager import ValidationError
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    Action,
    Decision,
)
from gpt_trader.monitoring.alert_types import AlertSeverity
from gpt_trader.monitoring.heartbeat import HeartbeatService
from gpt_trader.monitoring.status_reporter import StatusReporter

logger = logging.getLogger(__name__)

# Event type for price ticks
EVENT_PRICE_TICK = "price_tick"


class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.

    Supports state recovery via EventStore persistence.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self.running = False
        # Create strategy via factory (supports baseline and mean_reversion)
        self.strategy = create_strategy(self.context.config)
        self.price_history: dict[str, list[Decimal]] = defaultdict(list)
        self._current_positions: dict[str, Position] = {}
        self._rehydrated = False

        # Initialize heartbeat service
        self._heartbeat = HeartbeatService(
            event_store=context.event_store,
            ping_url=getattr(context.config, "heartbeat_url", None),
            interval_seconds=getattr(context.config, "heartbeat_interval", 60),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "heartbeat_enabled", True),
        )

        # Initialize status reporter
        self._status_reporter = StatusReporter(
            status_file=getattr(context.config, "status_file", "status.json"),
            update_interval=getattr(context.config, "status_interval", 10),
            bot_id=context.bot_id,
            enabled=getattr(context.config, "status_enabled", True),
        )
        self._status_reporter.set_heartbeat_service(self._heartbeat)

        # Pruning configuration
        self._prune_interval_seconds = 3600  # 1 hour
        self._prune_max_rows = 1_000_000  # Keep 1M events max

    async def _notify(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send notification if service is available."""
        if self.context.notification_service is None:
            return
        try:
            await self.context.notification_service.notify(
                title=title,
                message=message,
                severity=severity,
                source="TradingEngine",
                context=context,
            )
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    @property
    def name(self) -> str:
        return "strategy"

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Start the main trading loop and heartbeat service.

        Before starting, attempts to rehydrate state from EventStore.
        """
        # Rehydrate state from EventStore before starting
        if not self._rehydrated:
            self._rehydrate_from_events()
            self._rehydrated = True

        self.running = True

        tasks: list[asyncio.Task[Any]] = []

        # Start main trading loop
        trading_task = asyncio.create_task(self._run_loop())
        self._register_background_task(trading_task)
        tasks.append(trading_task)

        # Start heartbeat service
        heartbeat_task = await self._heartbeat.start()
        if heartbeat_task:
            self._register_background_task(heartbeat_task)
            tasks.append(heartbeat_task)

        # Start status reporter
        status_task = await self._status_reporter.start()
        if status_task:
            self._register_background_task(status_task)
            tasks.append(status_task)

        # Start database pruning task
        prune_task = asyncio.create_task(self._prune_loop())
        self._register_background_task(prune_task)
        tasks.append(prune_task)

        return tasks

    def _rehydrate_from_events(self) -> int:
        """Restore price history from persisted events.

        Returns:
            Number of price ticks restored
        """
        if self.context.event_store is None:
            logger.debug("No event store configured - skipping rehydration")
            return 0

        events = self.context.event_store.get_recent(count=1000)
        restored = 0

        for event in events:
            if event.get("type") != EVENT_PRICE_TICK:
                continue

            data = event.get("data", {})
            symbol = data.get("symbol")
            price_str = data.get("price")

            if not symbol or not price_str:
                continue

            # Only restore prices for symbols we're trading
            if symbol not in self.context.config.symbols:
                continue

            try:
                price = Decimal(str(price_str))
                self.price_history[symbol].append(price)
                # Keep history bounded
                if len(self.price_history[symbol]) > 20:
                    self.price_history[symbol].pop(0)
                restored += 1
            except Exception as e:
                logger.warning(f"Failed to parse price from event: {e}")

        if restored > 0:
            logger.info(f"Rehydrated {restored} price ticks from EventStore")
            for symbol, prices in self.price_history.items():
                logger.info(f"  {symbol}: {len(prices)} prices")

        # Also call strategy rehydration (for future stateful strategies)
        if hasattr(self.strategy, "rehydrate"):
            self.strategy.rehydrate(events)

        return restored

    async def _run_loop(self) -> None:
        logger.info("Starting strategy loop...")
        while self.running:
            try:
                await self._cycle()
                # Record successful cycle
                self._status_reporter.record_cycle()
            except Exception as e:
                logger.error(f"Error in strategy cycle: {e}", exc_info=True)
                # Record error in status reporter
                self._status_reporter.record_error(str(e))
                await self._notify(
                    title="Strategy Cycle Error",
                    message=f"Error during trading cycle: {e}",
                    severity=AlertSeverity.ERROR,
                    context={"error": str(e)},
                )

            await asyncio.sleep(self.context.config.interval)

    async def _prune_loop(self) -> None:
        """Periodically prune the event store to prevent unbounded growth."""
        logger.info(
            f"Starting database prune task (interval={self._prune_interval_seconds}s, "
            f"max_rows={self._prune_max_rows})"
        )
        while self.running:
            await asyncio.sleep(self._prune_interval_seconds)

            if self.context.event_store is None:
                continue

            try:
                # Check if the event store supports pruning
                if hasattr(self.context.event_store, "prune"):
                    pruned = self.context.event_store.prune(max_rows=self._prune_max_rows)
                    if pruned > 0:
                        logger.info(f"Pruned {pruned} old events from database")
            except Exception as e:
                logger.error(f"Database pruning failed: {e}")

    async def _cycle(self) -> None:
        """One trading cycle."""
        assert self.context.broker is not None, "Broker not initialized"

        # 1. Fetch positions first (needed for equity calculation)
        positions = await self._fetch_positions()
        self._current_positions = positions

        # Update status reporter with positions
        self._status_reporter.update_positions(self._positions_to_risk_format(positions))

        # 2. Calculate total equity including unrealized PnL
        equity = await self._fetch_total_equity(positions)
        if equity is None:
            logger.warning("Failed to fetch equity, skipping cycle")
            return

        # 2. Process Symbols
        for symbol in self.context.config.symbols:
            # Offload blocking network call
            try:
                ticker = await asyncio.to_thread(self.context.broker.get_ticker, symbol)
            except Exception as e:
                logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                continue

            price = Decimal(str(ticker.get("price", 0)))
            logger.info(f"{symbol} price: {price}")

            # Update status reporter with price
            self._status_reporter.update_price(symbol, price)

            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > 20:
                self.price_history[symbol].pop(0)

            # Persist price tick for crash recovery
            self._record_price_tick(symbol, price)

            position_state = self._build_position_state(symbol, positions)

            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=price,
                position_state=position_state,
                recent_marks=self.price_history[symbol],
                equity=equity,
                product=None,  # Future: fetch from broker
            )

            logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

            if decision.action in (Action.BUY, Action.SELL):
                logger.info(f"EXECUTING {decision.action} for {symbol}")
                try:
                    await self._validate_and_place_order(
                        symbol=symbol,
                        decision=decision,
                        price=price,
                        equity=equity,
                    )
                except ValidationError as e:
                    logger.warning(f"Risk validation failed for {symbol}: {e}")
                    await self._notify(
                        title="Risk Validation Failed",
                        message=f"Order blocked by risk manager: {e}",
                        severity=AlertSeverity.WARNING,
                        context={
                            "symbol": symbol,
                            "action": decision.action.value,
                            "reason": str(e),
                        },
                    )
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")
                    await self._notify(
                        title="Order Placement Failed",
                        message=f"Failed to execute {decision.action} for {symbol}: {e}",
                        severity=AlertSeverity.ERROR,
                        context={
                            "symbol": symbol,
                            "action": decision.action.value,
                            "error": str(e),
                        },
                    )

            elif decision.action == Action.CLOSE and position_state:
                # Handle CLOSE action separately if needed, or integrate into place_order
                # For now, logging it as per original logic, or we can implement close logic here
                logger.info(f"CLOSE signal for {symbol} - not fully implemented yet")

    async def _fetch_total_equity(self, positions: dict[str, Position]) -> Decimal | None:
        """Fetch total equity = collateral + unrealized PnL."""
        assert self.context.broker is not None
        try:
            balances = await asyncio.to_thread(self.context.broker.list_balances)
            collateral = Decimal("0")
            for balance in balances:
                if balance.asset in ("USD", "USDC"):
                    collateral += balance.available

            # Add unrealized PnL from open positions
            unrealized_pnl = sum(
                (p.unrealized_pnl for p in positions.values()),
                Decimal("0"),
            )
            return collateral + unrealized_pnl
        except Exception as e:
            logger.error(f"Failed to fetch balances: {e}")
            return None

    async def _fetch_positions(self) -> dict[str, Position]:
        """Fetch current positions as a lookup dict."""
        assert self.context.broker is not None
        try:
            positions_list = await asyncio.to_thread(self.context.broker.list_positions)
            return {p.symbol: p for p in positions_list}
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return {}

    def _build_position_state(
        self, symbol: str, positions: dict[str, Position]
    ) -> dict[str, Any] | None:
        """Build position state dict for strategy.decide()."""
        if symbol not in positions:
            return None
        pos = positions[symbol]
        return {
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "side": pos.side,
            # Add other fields if needed by strategy
        }

    def _record_price_tick(self, symbol: str, price: Decimal) -> None:
        """Persist price tick to EventStore for crash recovery."""
        if self.context.event_store is None:
            return

        self.context.event_store.store(
            {
                "type": EVENT_PRICE_TICK,
                "data": {
                    "symbol": symbol,
                    "price": str(price),
                    "timestamp": time.time(),
                    "bot_id": self.context.bot_id,
                },
            }
        )

    def _positions_to_risk_format(
        self, positions: dict[str, Position]
    ) -> dict[str, dict[str, Any]]:
        """Convert Position objects to dict format expected by risk manager."""
        return {
            symbol: {
                "quantity": pos.quantity,
                "mark": pos.mark_price,
            }
            for symbol, pos in positions.items()
        }

    def _calculate_order_quantity(
        self,
        symbol: str,
        price: Decimal,
        equity: Decimal,
        product: Product | None,
    ) -> Decimal:
        """Calculate order size based on equity and position_fraction."""
        # 1. Determine fraction
        fraction = Decimal("0.1")  # Default
        if hasattr(self.strategy, "config") and self.strategy.config.position_fraction:
            fraction = Decimal(str(self.strategy.config.position_fraction))
        elif hasattr(self.context.config, "perps_position_fraction"):
            fraction = self.context.config.perps_position_fraction

        # 2. Calculate raw quantity
        if price == 0:
            return Decimal("0")

        target_notional = equity * fraction
        quantity = target_notional / price

        # 3. Apply constraints
        if product and product.min_size:
            if quantity < product.min_size:
                logger.warning(f"Quantity {quantity} below min size {product.min_size}")
                return Decimal("0")

            # Round to step size if needed (simplified)
            # quantity = (quantity // product.step_size) * product.step_size

        return quantity

    async def _validate_and_place_order(
        self,
        symbol: str,
        decision: Decision,
        price: Decimal,
        equity: Decimal,
    ) -> None:
        """Validate order with risk manager before execution.

        Raises:
            ValidationError: If risk validation fails.
        """
        side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL

        # Dynamic position sizing
        quantity = self._calculate_order_quantity(symbol, price, equity, product=None)

        if quantity <= 0:
            logger.warning(f"Calculated quantity is {quantity}, skipping order")
            return

        # Run pre-trade validation if risk manager is available
        if self.context.risk_manager is not None:
            self.context.risk_manager.pre_trade_validate(
                symbol=symbol,
                side=side.value,
                quantity=quantity,
                price=price,
                product=None,
                equity=equity,
                current_positions=self._positions_to_risk_format(self._current_positions),
            )
            logger.info(f"Risk validation passed for {symbol} {side.value}")
        else:
            logger.warning("No risk manager configured - skipping validation")

        # Place order only after validation passes
        assert self.context.broker is not None, "Broker not initialized"
        await asyncio.to_thread(
            self.context.broker.place_order,
            symbol,
            side,
            OrderType.MARKET,
            quantity,
        )

        # Notify on successful order placement
        await self._notify(
            title="Order Executed",
            message=f"{side.value} {quantity} {symbol} at ~{price}",
            severity=AlertSeverity.INFO,
            context={
                "symbol": symbol,
                "side": side.value,
                "quantity": str(quantity),
                "price": str(price),
            },
        )

    async def shutdown(self) -> None:
        self.running = False
        await self._status_reporter.stop()
        await self._heartbeat.stop()
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)
