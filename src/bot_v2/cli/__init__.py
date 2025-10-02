"""Public CLI interface for the Perps trading bot.

Historically the project exposed a ``bot_v2.cli`` module with a ``main``
entry-point plus a couple of helpers that unit tests monkeypatch. During the
recent CLI refactor the implementation was moved into a package but the
re-exports were accidentally dropped, breaking those tests. This module keeps
the ergonomics stable by providing the expected surface while delegating the
actual work to the refactored command/handler modules.
"""

from __future__ import annotations

import logging

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
from bot_v2.orchestration.configuration import BotConfig
from dotenv import load_dotenv

__all__ = [
    "BotConfig",
    "build_bot",
    "main",
    "build_bot_config_from_args",
    "parse_and_validate_args",
    "order_tooling_requested",
]

# Preserve host-provided secrets; only fill gaps from .env
load_dotenv()

# Configure logging (rotating files + console)
configure_logging()
logger = logging.getLogger(__name__)


def main() -> int:
    """Entry-point used by the CLI executable and unit tests."""
    logger.info("Starting Perps Trading Bot CLI...")

    # Parse and validate arguments
    parser = setup_argument_parser()
    args = parse_and_validate_args(parser)

    # Build bot configuration and initialize
    logger.debug("Building bot configuration from arguments...")
    # Look up dependencies from module namespace to allow test monkeypatching
    import sys

    this_module = sys.modules[__name__]
    ConfigClass = getattr(this_module, "BotConfig", BotConfig)
    build_bot_fn = getattr(this_module, "build_bot", build_bot)
    config = build_bot_config_from_args(args, config_cls=ConfigClass)
    bot, _registry = build_bot_fn(config)

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
