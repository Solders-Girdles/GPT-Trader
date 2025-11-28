"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.
"""

import asyncio
import logging
from collections import defaultdict
from decimal import Decimal
from typing import Any

from gpt_trader.features.brokerages.core.interfaces import (
    OrderSide,
    OrderType,
    Position,
    Product,
)
from gpt_trader.features.live_trade.engines.base import (
    BaseEngine,
    CoordinatorContext,
    HealthStatus,
)
from gpt_trader.features.live_trade.risk.manager import ValidationError
from gpt_trader.features.live_trade.strategies.perps_baseline import (
    Action,
    BaselinePerpsStrategy,
    Decision,
)

logger = logging.getLogger(__name__)


class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self.running = False
        # Pass strategy config directly (BotConfig.strategy is a PerpsStrategyConfig)
        self.strategy = BaselinePerpsStrategy(config=self.context.config.strategy)
        self.price_history: dict[str, list[Decimal]] = defaultdict(list)
        self._current_positions: dict[str, Position] = {}

    @property
    def name(self) -> str:
        return "strategy"

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Start the main trading loop."""
        self.running = True
        task = asyncio.create_task(self._run_loop())
        self._register_background_task(task)
        return [task]

    async def _run_loop(self) -> None:
        logger.info("Starting strategy loop...")
        while self.running:
            try:
                await self._cycle()
            except Exception as e:
                logger.error(f"Error in strategy cycle: {e}", exc_info=True)

            await asyncio.sleep(self.context.config.interval)

    async def _cycle(self) -> None:
        """One trading cycle."""
        assert self.context.broker is not None, "Broker not initialized"

        # 1. Fetch positions first (needed for equity calculation)
        positions = await self._fetch_positions()
        self._current_positions = positions

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

            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > 20:
                self.price_history[symbol].pop(0)

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
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")

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

    async def shutdown(self) -> None:
        self.running = False
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)
