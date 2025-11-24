"""
Simplified TradingBot.
Acts as the main entry point runner.
"""
import asyncio
import logging
from typing import Any, TYPE_CHECKING

from gpt_trader.orchestration.configuration import BotConfig
from gpt_trader.features.live_trade.engines.strategy import TradingEngine
from gpt_trader.features.live_trade.engines.base import CoordinatorContext

if TYPE_CHECKING:
    from gpt_trader.app.container import ApplicationContainer

logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(
        self, 
        config: BotConfig,
        container: "ApplicationContainer | None" = None,
        registry: Any = None,
        event_store: Any = None,
        orders_store: Any = None,
    ):
        self.config = config
        self.container = container
        self.running = False

        # Setup context (simplified)
        self.context = CoordinatorContext(
            config=config,
            registry=registry,
            broker=registry.broker if registry else None,
            symbols=tuple(config.symbols)
        )

        self.engine = TradingEngine(self.context)

    async def run(self, single_cycle: bool = False):
        self.running = True
        logger.info(f"TradingBot starting with symbols: {self.config.symbols}")

        tasks = await self.engine.start_background_tasks()

        try:
            if single_cycle:
                # For testing/dev-fast
                await asyncio.sleep(self.config.interval + 0.1)
                await self.engine.shutdown()
            else:
                # Keep alive until tasks complete (which is never, unless error or shutdown)
                await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Bot stopped.")
        finally:
            await self.engine.shutdown()
            self.running = False

    async def stop(self):
        self.running = False
        await self.engine.shutdown()

__all__ = ["TradingBot"]
