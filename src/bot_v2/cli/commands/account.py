"""
Account management commands for the Perps Trading Bot CLI.

Provides account snapshot and telemetry collection functionality.
"""

import json
import logging

from bot_v2.cli.handlers.shutdown import ensure_shutdown
from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


def handle_account_snapshot(bot: PerpsBot) -> int:
    """
    Print account telemetry snapshot and exit.

    Args:
        bot: Initialized PerpsBot instance

    Returns:
        Exit code (0 for success, non-zero for failure)

    Raises:
        RuntimeError: If account snapshot telemetry is not available
    """
    logger.info("Collecting account snapshot...")

    telemetry = getattr(bot, "account_telemetry", None)
    if telemetry is None or not telemetry.supports_snapshots():
        logger.error("Account snapshot telemetry is not available for this broker")
        ensure_shutdown(bot)
        raise RuntimeError("Account snapshot telemetry is not available for this broker")

    try:
        snapshot = telemetry.collect_snapshot()
        logger.info("Account snapshot collected successfully")

        # Print snapshot as formatted JSON
        output = json.dumps(snapshot, indent=2, default=str)
        print(output)

        return 0
    except Exception as e:
        logger.error("Failed to collect account snapshot: %s", e, exc_info=True)
        raise
    finally:
        ensure_shutdown(bot)
