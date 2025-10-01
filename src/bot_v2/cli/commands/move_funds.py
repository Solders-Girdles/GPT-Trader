"""
Fund movement command for the Perps Trading Bot CLI.

Provides portfolio-to-portfolio fund transfer functionality.
"""

import argparse
import json
import logging

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
    logger.info("Processing move-funds command with arg=%s", move_arg)

    # Parse FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT format
    try:
        from_uuid, to_uuid, amount = (part.strip() for part in move_arg.split(":", 2))
    except ValueError:
        logger.error("Invalid move-funds argument format: %s", move_arg)
        parser.error("--move-funds requires format FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT")

    logger.info(
        "Moving %s from portfolio %s to portfolio %s",
        amount,
        from_uuid,
        to_uuid,
    )

    try:
        # Build fund movement payload
        payload = {"from_portfolio": from_uuid, "to_portfolio": to_uuid, "amount": amount}

        # Execute fund movement
        result = bot.account_manager.move_funds(payload)

        logger.info("Fund movement completed successfully")

        # Print result as formatted JSON
        output = json.dumps(result, indent=2, default=str)
        print(output)

        return 0
    except Exception as e:
        logger.error("Fund movement failed: %s", e, exc_info=True)
        raise
    finally:
        ensure_shutdown(bot)
