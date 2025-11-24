"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.
"""
import asyncio
import logging
from typing import Any
from decimal import Decimal
from collections import defaultdict

from gpt_trader.features.live_trade.strategies.perps_baseline import BaselinePerpsStrategy, Action

from gpt_trader.features.live_trade.engines.base import BaseEngine, CoordinatorContext, HealthStatus


logger = logging.getLogger(__name__)

class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.
    """
    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self.running = False
        self.strategy = BaselinePerpsStrategy(config=self.context.config)
        self.price_history = defaultdict(list)

    @property
    def name(self) -> str:
        return "strategy"

    async def start_background_tasks(self) -> list[asyncio.Task[Any]]:
        """Start the main trading loop."""
        self.running = True
        task = asyncio.create_task(self._run_loop())
        self._register_background_task(task)
        return [task]

    async def _run_loop(self):
        logger.info("Starting strategy loop...")
        while self.running:
            try:
                await self._cycle()
            except Exception as e:
                logger.error(f"Error in strategy cycle: {e}", exc_info=True)

            await asyncio.sleep(self.context.config.interval)

    async def _cycle(self):
        """One trading cycle."""
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
                product=None
            )
            
            logger.info(f"Strategy Decision for {symbol}: {decision.action} ({decision.reason})")

            if decision.action in (Action.BUY, Action.SELL):
                logger.info(f"EXECUTING {decision.action} for {symbol}")
                try:
                    order_payload = {
                        "product_id": symbol,
                        "side": decision.action.value.upper(),
                        "order_configuration": {
                            "market_market_ioc": {
                                "quote_size": "10" # Dummy size for test
                            }
                        }
                    }
                    # Offload blocking network call
                    await asyncio.to_thread(self.context.broker.place_order, order_payload)
                except Exception as e:
                    logger.error(f"Order placement failed: {e}")

    async def shutdown(self) -> None:
        self.running = False
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)
