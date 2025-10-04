"""Fund movement service for CLI commands.

Provides MoveFundsService to execute fund movements and format results as JSON.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from bot_v2.cli.commands.move_funds_request_parser import MoveFundsRequest

logger = logging.getLogger(__name__)


class MoveFundsService:
    """Executes fund movements and formats results."""

    def __init__(self, printer: Callable[[str], None] | None = None) -> None:
        """
        Initialize fund movement service.

        Args:
            printer: Function to print output (default: print)
        """
        self._printer = printer or print
        self._logger = logging.getLogger(__name__)

    def execute_fund_movement(self, bot: Any, request: MoveFundsRequest) -> int:
        """
        Execute fund movement and print JSON result.

        Args:
            bot: PerpsBot instance with account_manager
            request: Parsed fund movement request

        Returns:
            Exit code (0 for success, non-zero for failure)

        Raises:
            Exception: If fund movement fails
        """
        self._logger.info(
            "Moving %s from portfolio %s to portfolio %s",
            request.amount,
            request.from_portfolio,
            request.to_portfolio,
        )

        try:
            # Build fund movement payload
            payload = {
                "from_portfolio": request.from_portfolio,
                "to_portfolio": request.to_portfolio,
                "amount": request.amount,
            }

            # Execute fund movement (no commit flag)
            result = bot.account_manager.move_funds(payload)

            self._logger.info("Fund movement completed successfully")

            # Print result as formatted JSON
            output = json.dumps(result, indent=2, default=str)
            self._printer(output)

            return 0
        except Exception as e:
            self._logger.error("Fund movement failed: %s", e, exc_info=True)
            raise
