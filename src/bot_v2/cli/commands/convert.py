"""
Asset conversion command for the Perps Trading Bot CLI.

Provides currency/asset conversion functionality.
"""

import argparse
import logging

from bot_v2.cli.commands.convert_request_parser import ConvertRequestParser
from bot_v2.cli.commands.convert_service import ConvertService
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
    request = ConvertRequestParser.parse(convert_arg, parser)
    service = ConvertService()
    try:
        return service.execute_conversion(bot, request)
    finally:
        ensure_shutdown(bot)
