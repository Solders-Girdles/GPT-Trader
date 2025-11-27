"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.
"""

import asyncio
import logging
from collections import defaultdict
from decimal import Decimal
from typing import Any

from gpt_trader.features.live_trade.engines.base import BaseEngine, CoordinatorContext, HealthStatus
from gpt_trader.features.live_trade.strategies.perps_baseline import Action, BaselinePerpsStrategy

logger = logging.getLogger(__name__)


class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.
    """

    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self.running = False
        self.strategy = BaselinePerpsStrategy(config=self.context.config)
        self.price_history: dict[str, list[Decimal]] = defaultdict(list)

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
        # 1. Fetch Data
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

            decision = self.strategy.decide(
                symbol=symbol,
                current_mark=price,
                position_state=None,
                recent_marks=self.price_history[symbol],
                equity=Decimal("1000"),
                product=None,
            )

            logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

            if decision.action in (Action.BUY, Action.SELL):
                logger.info(f"EXECUTING {decision.action} for {symbol}")
                try:
                    from gpt_trader.features.brokerages.core.interfaces import OrderSide, OrderType

                    side = OrderSide.BUY if decision.action == Action.BUY else OrderSide.SELL
                    # Offload blocking network call
                    await asyncio.to_thread(
                        self.context.broker.place_order,
                        symbol,
                        side,
                        OrderType.MARKET,
                        Decimal("0.001"),  # Dummy size for test
                    )
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")

    async def shutdown(self) -> None:
        self.running = False
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)
