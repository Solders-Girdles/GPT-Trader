"""
Command Line Interface for the Perps Trading Bot.

Refactored into modular subcommands and handlers for maintainability.
"""

import logging
import sys

from dotenv import load_dotenv

from bot_v2.cli.commands.account import handle_account_snapshot
from bot_v2.cli.commands.convert import handle_convert
from bot_v2.cli.commands.move_funds import handle_move_funds
from bot_v2.cli.commands.orders import handle_order_tooling
from bot_v2.cli.commands.run import handle_run_bot
from bot_v2.cli.parser import (
    build_bot_config_from_args,
    order_tooling_requested,
    parse_and_validate_args,
    setup_argument_parser,
)
from bot_v2.logging import configure_logging
from bot_v2.orchestration.bootstrap import build_bot

# Preserve host-provided secrets; only fill gaps from .env
load_dotenv()

# Configure logging (rotating files + console)
configure_logging()
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Main entry point for the CLI.

    Delegates to appropriate command handlers based on arguments.
    """
    logger.info("Starting Perps Trading Bot CLI...")

    # Parse and validate arguments
    parser = setup_argument_parser()
    args = parse_and_validate_args(parser)

    # Build bot configuration and initialize
    logger.debug("Building bot configuration from arguments...")
    config = build_bot_config_from_args(args)
    bot, _registry = build_bot(config)

    logger.info("Bot initialized successfully, dispatching to command handler...")

    # Dispatch to appropriate command handler
    if args.account_snapshot:
        return handle_account_snapshot(bot)

    if order_tooling_requested(args):
        return handle_order_tooling(args, bot, parser)

    if args.convert:
        return handle_convert(args.convert, bot, parser)

    if args.move_funds:
        return handle_move_funds(args.move_funds, bot, parser)

    # Default: run the main bot
    return handle_run_bot(bot, args.dev_fast)


if __name__ == "__main__":
    sys.exit(main())
