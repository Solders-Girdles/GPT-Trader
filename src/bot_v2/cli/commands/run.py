"""
Main bot execution command for the Perps Trading Bot CLI.

Provides continuous trading execution with signal handling and graceful shutdown.
"""

import logging

from bot_v2.cli.commands.lifecycle_controller import LifecycleController
from bot_v2.cli.commands.signal_manager import SignalManager
from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


def handle_run_bot(bot: PerpsBot, dev_fast: bool = False) -> int:
    """
    Run the main trading bot with signal handling.

    Args:
        bot: Initialized PerpsBot instance
        dev_fast: If True, run single cycle and exit (for smoke tests)

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Starting bot execution (dev_fast=%s)", dev_fast)

    # Setup signal handling
    signal_manager = SignalManager()
    signal_manager.setup_signals(bot)

    # Setup lifecycle controller
    lifecycle = LifecycleController()

    try:
        # Run the bot (async via lifecycle controller)
        lifecycle.execute(bot, single_cycle=dev_fast)

        if dev_fast:
            logger.info("Single cycle completed successfully")
        else:
            logger.info("Bot execution completed")

        return 0

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutdown complete")
        return 0

    except Exception as e:
        logger.error("Bot execution failed: %s", e, exc_info=True)
        return 1
