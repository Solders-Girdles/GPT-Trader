"""
Simplified PerpsBot.
Acts as the main entry point runner.
"""
import asyncio
import logging
from typing import Any

from bot_v2.orchestration.configuration import BotConfig
from bot_v2.orchestration.engines.strategy import TradingEngine
from bot_v2.orchestration.engines.base import CoordinatorContext

logger = logging.getLogger(__name__)

class PerpsBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.running = False

        # Setup context (simplified)
        self.context = CoordinatorContext(
            config=config,
            registry=None, # type: ignore
            symbols=tuple(config.symbols)
        )

        self.engine = TradingEngine(self.context)

    async def run(self, single_cycle: bool = False):
        self.running = True
        logger.info(f"PerpsBot starting with symbols: {self.config.symbols}")

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

__all__ = ["PerpsBot"]
