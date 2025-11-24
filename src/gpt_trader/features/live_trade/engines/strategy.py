"""
Simplified Strategy Engine.
Replaces the 608-line enterprise coordinator with a simple loop.
"""
import asyncio
import logging
from typing import Any
from decimal import Decimal

from gpt_trader.features.live_trade.engines.base import BaseEngine, CoordinatorContext, HealthStatus
from gpt_trader.orchestration.configuration import config
from gpt_trader.features.brokerages.coinbase.client import CoinbaseClient

logger = logging.getLogger(__name__)

class TradingEngine(BaseEngine):
    """
    Simple trading loop that fetches data and executes strategy.
    """
    def __init__(self, context: CoordinatorContext) -> None:
        super().__init__(context)
        self.running = False
        self.client = CoinbaseClient(
            base_url="https://api.coinbase.com",
            auth=None, # Will auto-load from env in CoinbaseClient if implemented,
                       # but our CoinbaseClient uses mixins. We need to check how we instantiated it.
                       # Ah, I need to make sure CoinbaseClient loads auth.
            api_mode="advanced"
        )
        # Manually load auth for the client
        from gpt_trader.features.brokerages.coinbase.client import SimpleAuth
        import os
        key_name = os.getenv("COINBASE_API_KEY_NAME")
        key_secret = os.getenv("COINBASE_PRIVATE_KEY")
        if key_name and key_secret:
            self.client.auth = SimpleAuth(key_name, key_secret)

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

            await asyncio.sleep(config.interval)

    async def _cycle(self):
        """One trading cycle."""
        # 1. Fetch Data
        for symbol in self.context.config.symbols:
            ticker = self.client.get_ticker(symbol)
            price = Decimal(str(ticker.get("price", 0)))
            logger.info(f"{symbol} price: {price}")

            # 2. Calculate Signal (Simple MA Crossover Placeholder)
            # In a real scenario, we'd have a strategy class here.
            # For "Kill the Hydra", we keep it inline or simple.

            # 3. Execute
            # if signal: self.client.create_order(...)
            pass

    async def shutdown(self) -> None:
        self.running = False
        await super().shutdown()

    def health_check(self) -> HealthStatus:
        return HealthStatus(healthy=self.running, component=self.name)
