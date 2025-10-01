"""
Asset conversion command for the Perps Trading Bot CLI.

Provides currency/asset conversion functionality.
"""

import argparse
import json
import logging

from bot_v2.cli.handlers.shutdown import ensure_shutdown
from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


def handle_convert(convert_arg: str, bot: PerpsBot, parser: argparse.ArgumentParser) -> int:
    """
    Handle asset conversion command.

    Args:
        convert_arg: Conversion argument in format "FROM:TO:AMOUNT"
        bot: Initialized PerpsBot instance
        parser: ArgumentParser for error reporting

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Processing convert command with arg=%s", convert_arg)

    # Parse FROM:TO:AMOUNT format
    try:
        from_asset, to_asset, amount = (part.strip() for part in convert_arg.split(":", 2))
    except ValueError:
        logger.error("Invalid convert argument format: %s", convert_arg)
        parser.error("--convert requires format FROM:TO:AMOUNT")

    logger.info(
        "Converting %s from %s to %s",
        amount,
        from_asset,
        to_asset,
    )

    try:
        # Build conversion payload
        payload = {"from": from_asset, "to": to_asset, "amount": amount}

        # Execute conversion
        result = bot.account_manager.convert(payload, commit=True)

        logger.info("Conversion completed successfully")

        # Print result as formatted JSON
        output = json.dumps(result, indent=2, default=str)
        print(output)

        return 0
    except Exception as e:
        logger.error("Conversion failed: %s", e, exc_info=True)
        raise
    finally:
        ensure_shutdown(bot)
