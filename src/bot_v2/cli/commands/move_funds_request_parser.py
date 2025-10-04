"""Move funds request parsing for CLI commands.

Provides MoveFundsRequest dataclass and MoveFundsRequestParser to parse
FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT format fund movement arguments.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MoveFundsRequest:
    """Parsed fund movement request."""

    from_portfolio: str
    to_portfolio: str
    amount: str


class MoveFundsRequestParser:
    """Parses fund movement request strings."""

    @staticmethod
    def parse(move_arg: str, parser: argparse.ArgumentParser) -> MoveFundsRequest:
        """
        Parse FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT format into MoveFundsRequest.

        Args:
            move_arg: Fund movement argument in format "FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT"
            parser: ArgumentParser for error reporting

        Returns:
            MoveFundsRequest with parsed values

        Raises:
            SystemExit: If format is invalid (via parser.error)
        """
        logger.info("Parsing move-funds argument: %s", move_arg)

        try:
            from_uuid, to_uuid, amount = (part.strip() for part in move_arg.split(":", 2))
        except ValueError:
            logger.error("Invalid move-funds argument format: %s", move_arg)
            parser.error("--move-funds requires format FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT")

        logger.info(
            "Parsed fund movement: %s from portfolio %s to portfolio %s", amount, from_uuid, to_uuid
        )

        return MoveFundsRequest(from_portfolio=from_uuid, to_portfolio=to_uuid, amount=amount)
