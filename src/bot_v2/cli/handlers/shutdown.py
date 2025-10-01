"""
Shutdown handler for the Perps Trading Bot CLI.

Provides consistent shutdown behavior across all CLI commands,
with proper signal handling and logging.
"""

import asyncio
import logging
import signal
from types import FrameType
from typing import Optional

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """
    Manages graceful shutdown of the bot with signal handling.

    Provides consistent shutdown behavior across all CLI commands,
    ensuring proper cleanup and logging.
    """

    def __init__(self, bot) -> None:
        """
        Initialize the shutdown handler.

        Args:
            bot: PerpsBot instance to manage shutdown for
        """
        self.bot = bot
        self._shutdown_requested = False

    def register_signals(self) -> None:
        """Register SIGINT and SIGTERM handlers."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.debug("Registered signal handlers for SIGINT and SIGTERM")

    def _signal_handler(self, sig: int, frame: Optional[FrameType]) -> None:
        """
        Handle shutdown signals.

        Args:
            sig: Signal number received
            frame: Current stack frame
        """
        if self._shutdown_requested:
            logger.warning("Shutdown already in progress, ignoring signal")
            return

        self._shutdown_requested = True
        signal_name = signal.Signals(sig).name
        logger.info("Signal %s received, initiating shutdown...", signal_name)
        self.bot.running = False

    async def shutdown(self) -> None:
        """
        Perform graceful shutdown of the bot.

        Logs shutdown initiation and ensures proper cleanup.
        """
        if self._shutdown_requested:
            logger.debug("Shutdown already initiated")
            return

        self._shutdown_requested = True
        logger.info("Initiating graceful shutdown...")

        try:
            await self.bot.shutdown()
            logger.info("Shutdown completed successfully")
        except Exception as e:
            logger.error("Error during shutdown: %s", e, exc_info=True)
            raise

    def shutdown_sync(self) -> None:
        """
        Perform synchronous shutdown of the bot.

        Convenience method for commands that don't run in async context.
        """
        logger.info("Performing synchronous shutdown...")
        try:
            asyncio.run(self.shutdown())
        except Exception as e:
            logger.error("Error during synchronous shutdown: %s", e, exc_info=True)
            raise


def ensure_shutdown(bot) -> None:
    """
    Ensure bot is shut down cleanly, synchronously.

    Convenience function for simple command handlers that just need
    to ensure cleanup before exiting.

    Args:
        bot: PerpsBot instance to shut down
    """
    logger.debug("Ensuring bot shutdown...")
    try:
        asyncio.run(bot.shutdown())
        logger.debug("Bot shutdown complete")
    except Exception as e:
        logger.error("Error ensuring shutdown: %s", e, exc_info=True)
        raise
