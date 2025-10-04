"""
Account management commands for the Perps Trading Bot CLI.

Provides account snapshot and telemetry collection functionality.
"""

import logging

from bot_v2.cli.commands.account_snapshot_service import AccountSnapshotService
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
    service = AccountSnapshotService()
    try:
        return service.collect_and_print(bot)
    finally:
        ensure_shutdown(bot)
