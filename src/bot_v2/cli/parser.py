"""
Argument parsing for the Perps Trading Bot CLI.

Provides modular argument parsers for different command modes,
with consistent validation and logging.
"""

import argparse
import logging

from bot_v2.cli.argument_groups import ArgumentGroupRegistrar
from bot_v2.cli.argument_validator import ArgumentValidator
from bot_v2.cli.bot_config_builder import BotConfigBuilder

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with all CLI arguments.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Perpetuals Trading Bot")
    ArgumentGroupRegistrar.register_all(parser)
    return parser


def parse_and_validate_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Parse and validate CLI arguments.

    Args:
        parser: Configured ArgumentParser instance

    Returns:
        Validated argument namespace

    Raises:
        SystemExit: On argument validation errors
    """
    args = parser.parse_args()
    validator = ArgumentValidator()
    return validator.validate(args, parser)


def build_bot_config_from_args(
    args: argparse.Namespace,
    *,
    config_cls=None,
):
    """
    Build BotConfig from parsed CLI arguments.

    Args:
        args: Parsed argument namespace
        config_cls: Optional BotConfig factory class

    Returns:
        BotConfig instance configured from arguments
    """
    builder = BotConfigBuilder(config_factory=config_cls)
    return builder.build(args)


def order_tooling_requested(args: argparse.Namespace) -> bool:
    """
    Check if any order tooling command was requested.

    Args:
        args: Parsed argument namespace

    Returns:
        True if order tooling command was requested
    """
    return any([args.preview_order, args.edit_order_preview, args.apply_order_edit])
