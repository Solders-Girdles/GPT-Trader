"""Account snapshot collection and formatting service.

Provides AccountSnapshotService to collect account telemetry snapshots
and format them as JSON output.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class AccountSnapshotService:
    """Collects and formats account telemetry snapshots."""

    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        """
        Initialize account snapshot service.

        Args:
            printer: Function to print output (default: print)
        """
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def collect_and_print(self, bot: Any) -> int:
        """
        Collect account snapshot and print as formatted JSON.

        Args:
            bot: PerpsBot instance with account_telemetry

        Returns:
            Exit code (0 for success, non-zero for failure)

        Raises:
            RuntimeError: If account snapshot telemetry is not available
        """
        self._logger.info("Collecting account snapshot...")

        # Validate telemetry availability
        telemetry = getattr(bot, "account_telemetry", None)
        if telemetry is None or not telemetry.supports_snapshots():
            self._logger.error("Account snapshot telemetry is not available for this broker")
            raise RuntimeError("Account snapshot telemetry is not available for this broker")

        try:
            # Collect snapshot
            snapshot = telemetry.collect_snapshot()
            self._logger.info("Account snapshot collected successfully")

            # Format and print as JSON
            output = json.dumps(snapshot, indent=2, default=str)
            self._printer(output)

            return 0
        except Exception as e:
            self._logger.error("Failed to collect account snapshot: %s", e, exc_info=True)
            raise
