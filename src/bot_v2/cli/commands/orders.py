"""
Order tooling commands for the Perps Trading Bot CLI.

Provides order preview, edit preview, and edit application functionality.
"""

import argparse
import logging

from bot_v2.cli.commands.edit_preview_service import EditPreviewService
from bot_v2.cli.commands.order_args import OrderArgumentsParser
from bot_v2.cli.commands.order_preview_service import OrderPreviewService
from bot_v2.cli.handlers.shutdown import ensure_shutdown
from bot_v2.orchestration.perps_bot import PerpsBot

logger = logging.getLogger(__name__)


def handle_order_tooling(
    args: argparse.Namespace, bot: PerpsBot, parser: argparse.ArgumentParser
) -> int:
    """
    Handle order tooling commands (preview, edit-preview, apply-edit).

    Args:
        args: Parsed CLI arguments
        bot: Initialized PerpsBot instance
        parser: ArgumentParser for error reporting

    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    logger.info("Processing order tooling command...")

    try:
        if args.preview_order:
            return _handle_preview_order(args, bot, parser)

        if args.edit_order_preview:
            return _handle_edit_order_preview(args, bot, parser)

        if args.apply_order_edit:
            return _handle_apply_order_edit(args, bot, parser)

        parser.error("Order tooling command provided but no action executed")
    except Exception as e:
        logger.error("Order tooling command failed: %s", e, exc_info=True)
        raise
    finally:
        ensure_shutdown(bot)


def _handle_preview_order(
    args: argparse.Namespace, bot: PerpsBot, parser: argparse.ArgumentParser
) -> int:
    """
    Handle order preview command.

    Args:
        args: Parsed CLI arguments
        bot: PerpsBot instance
        parser: ArgumentParser for error reporting
        symbol: Trading symbol

    Returns:
        Exit code (0 for success)
    """
    parsed = OrderArgumentsParser.parse_preview(args, parser)
    service = OrderPreviewService()
    return service.preview(bot, parsed)


def _handle_edit_order_preview(
    args: argparse.Namespace, bot: PerpsBot, parser: argparse.ArgumentParser
) -> int:
    """
    Handle order edit preview command.

    Args:
        args: Parsed CLI arguments
        bot: PerpsBot instance
        parser: ArgumentParser for error reporting
        symbol: Trading symbol

    Returns:
        Exit code (0 for success)
    """
    parsed = OrderArgumentsParser.parse_edit_preview(args, parser)
    service = EditPreviewService()
    return service.edit_preview(bot, parsed)


def _handle_apply_order_edit(
    args: argparse.Namespace, bot: PerpsBot, parser: argparse.ArgumentParser
) -> int:
    """
    Handle apply order edit command.

    Args:
        args: Parsed CLI arguments
        bot: PerpsBot instance
        parser: ArgumentParser for error reporting

    Returns:
        Exit code (0 for success)
    """
    parsed = OrderArgumentsParser.parse_apply_edit(args, parser)
    service = EditPreviewService()
    return service.apply_edit(bot, parsed)
