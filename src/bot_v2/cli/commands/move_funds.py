"""
Fund movement command for the Perps Trading Bot CLI.

Provides portfolio-to-portfolio fund transfer functionality.
"""

import argparse
import logging

from bot_v2.cli.commands.move_funds_request_parser import MoveFundsRequestParser
from bot_v2.cli.commands.move_funds_service import MoveFundsService
from bot_v2.cli.handlers.shutdown import ensure_shutdown
from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


def handle_move_funds(move_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    """
    Handle fund movement command.

    Args:
        move_arg: Move argument in format "FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT"
        bot: Initialized PerpsBot instance
        parser: ArgumentParser for error reporting

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    request = MoveFundsRequestParser.parse(move_arg, parser)
    service = MoveFundsService()
    try:
        return service.execute_fund_movement(bot, request)
    finally:
        ensure_shutdown(bot)
