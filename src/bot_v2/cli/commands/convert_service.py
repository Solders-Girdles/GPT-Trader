"""Asset conversion service for CLI commands.

Provides ConvertService to execute asset conversions and format results as JSON.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from bot_v2.cli.commands.convert_request_parser import ConvertRequest

logger = logging.getLogger(__name__)


class ConvertService:
    """Executes asset conversions and formats results."""

    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        """
        Initialize conversion service.

        Args:
            printer: Function to print output (default: print)
        """
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def execute_conversion(self, bot: Any, request: ConvertRequest) -> int:
        """
        Execute asset conversion and print JSON result.

        Args:
            bot: PerpsBot instance with account_manager
            request: Parsed conversion request

        Returns:
            Exit code (0 for success, non-zero for failure)

        Raises:
            Exception: If conversion fails
        """
        self._logger.info(
            "Converting %s from %s to %s",
            request.amount,
            request.from_asset,
            request.to_asset,
        )

        try:
            # Build conversion payload
            payload = {
                "from": request.from_asset,
                "to": request.to_asset,
                "amount": request.amount,
            }

            # Execute conversion
            result = bot.account_manager.convert(payload, commit=True)

            self._logger.info("Conversion completed successfully")

            # Print result as formatted JSON
            output = json.dumps(result, indent=2, default=str)
            self._printer(output)

            return 0
        except Exception as e:
            self._logger.error("Conversion failed: %s", e, exc_info=True)
            raise
