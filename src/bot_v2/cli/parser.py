"""
Argument parsing for the Perps Trading Bot CLI.

Provides modular argument parsers for different command modes,
with consistent validation and logging.
"""

import argparse
import logging
import os
from decimal import Decimal

logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Create the main argument parser with all CLI arguments.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description="Perpetuals Trading Bot")

    # Core bot configuration
    _add_bot_config_args(parser)

    # Command-specific arguments
    _add_account_args(parser)
    _add_convert_args(parser)
    _add_move_funds_args(parser)
    _add_order_tooling_args(parser)
    _add_dev_args(parser)

    return parser


def _add_bot_config_args(parser: argparse.ArgumentParser) -> None:
    """Add core bot configuration arguments."""
    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        choices=["dev", "demo", "prod", "canary", "spot"],
        help="Configuration profile",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without placing real orders",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade (e.g., BTC-PERP ETH-PERP)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        help="Update interval in seconds",
    )
    parser.add_argument(
        "--leverage",
        dest="target_leverage",
        type=int,
        help="Target leverage",
    )
    parser.add_argument(
        "--reduce-only",
        dest="reduce_only_mode",
        action="store_true",
        help="Enable reduce-only mode",
    )
    parser.add_argument(
        "--tif",
        dest="time_in_force",
        type=str,
        choices=["GTC", "IOC", "FOK"],
        help="Time in force policy (GTC/IOC/FOK)",
    )
    parser.add_argument(
        "--enable-preview",
        dest="enable_order_preview",
        action="store_true",
        help="Enable order preview before placement",
    )
    parser.add_argument(
        "--account-interval",
        dest="account_telemetry_interval",
        type=int,
        help="Account telemetry interval in seconds",
    )


def _add_account_args(parser: argparse.ArgumentParser) -> None:
    """Add account management arguments."""
    parser.add_argument(
        "--account-snapshot",
        action="store_true",
        help="Print account telemetry snapshot and exit",
    )


def _add_convert_args(parser: argparse.ArgumentParser) -> None:
    """Add conversion command arguments."""
    parser.add_argument(
        "--convert",
        metavar="FROM:TO:AMOUNT",
        help="Perform a convert trade and exit",
    )


def _add_move_funds_args(parser: argparse.ArgumentParser) -> None:
    """Add move funds command arguments."""
    parser.add_argument(
        "--move-funds",
        metavar="FROM:TO:AMOUNT",
        help="Move funds between portfolios and exit",
    )


def _add_order_tooling_args(parser: argparse.ArgumentParser) -> None:
    """Add order tooling command arguments."""
    parser.add_argument(
        "--preview-order",
        action="store_true",
        help="Preview a new order and exit",
    )
    parser.add_argument(
        "--edit-order-preview",
        metavar="ORDER_ID",
        help="Preview edits for ORDER_ID and exit",
    )
    parser.add_argument(
        "--apply-order-edit",
        metavar="ORDER_ID:PREVIEW_ID",
        help="Apply order edit using preview id and exit",
    )
    parser.add_argument(
        "--order-symbol",
        help="Symbol for order preview/edit commands",
    )
    parser.add_argument(
        "--order-side",
        choices=["buy", "sell"],
        help="Order side for preview/edit commands",
    )
    parser.add_argument(
        "--order-type",
        choices=["market", "limit", "stop_limit"],
        help="Order type for preview/edit commands",
    )
    parser.add_argument(
        "--order-quantity",
        dest="order_quantity",
        type=Decimal,
        help="Order quantity for preview/edit commands",
    )
    parser.add_argument(
        "--order-price",
        type=Decimal,
        help="Limit price for preview/edit commands",
    )
    parser.add_argument(
        "--order-stop",
        type=Decimal,
        help="Stop price for preview/edit commands",
    )
    parser.add_argument(
        "--order-tif",
        choices=["GTC", "IOC", "FOK"],
        help="Time in force for preview/edit commands",
    )
    parser.add_argument(
        "--order-client-id",
        help="Client order id for preview/edit commands",
    )
    parser.add_argument(
        "--order-reduce-only",
        action="store_true",
        help="Set reduce_only flag for preview/edit commands",
    )
    parser.add_argument(
        "--order-leverage",
        type=int,
        help="Leverage override for preview/edit commands",
    )


def _add_dev_args(parser: argparse.ArgumentParser) -> None:
    """Add development and testing arguments."""
    parser.add_argument(
        "--dev-fast",
        action="store_true",
        help="Run single cycle and exit (for smoke tests)",
    )


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

    # Validate symbols if provided
    if args.symbols:
        empty = [sym for sym in args.symbols if not str(sym).strip()]
        if empty:
            parser.error("Symbols must be non-empty strings")

    # Enable debug logging if requested
    if os.getenv("PERPS_DEBUG") == "1":
        logger.info("Debug mode enabled via PERPS_DEBUG=1")
        logging.getLogger("bot_v2.features.brokerages.coinbase").setLevel(logging.DEBUG)
        logging.getLogger("bot_v2.orchestration").setLevel(logging.DEBUG)

    logger.debug("Parsed CLI arguments: profile=%s, dry_run=%s", args.profile, args.dry_run)

    return args


def build_bot_config_from_args(args: argparse.Namespace):
    """
    Build BotConfig from parsed CLI arguments.

    Args:
        args: Parsed argument namespace

    Returns:
        BotConfig instance configured from arguments
    """
    from bot_v2.orchestration.configuration import BotConfig

    # Arguments that should not be passed to BotConfig
    skip_keys = {
        "profile",
        "account_snapshot",
        "convert",
        "move_funds",
        "preview_order",
        "edit_order_preview",
        "apply_order_edit",
        "order_side",
        "order_type",
        "order_quantity",
        "order_price",
        "order_stop",
        "order_tif",
        "order_client_id",
        "order_reduce_only",
        "order_leverage",
        "order_symbol",
    }

    config_overrides = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key not in skip_keys
    }

    # Handle symbols from environment if not provided via CLI
    if "symbols" not in config_overrides or not config_overrides.get("symbols"):
        env_symbols = os.getenv("TRADING_SYMBOLS", "")
        if env_symbols:
            tokens = [
                tok.strip() for tok in env_symbols.replace(";", ",").split(",") if tok.strip()
            ]
            if tokens:
                config_overrides["symbols"] = tokens
                logger.info("Loaded %d symbols from TRADING_SYMBOLS env var", len(tokens))

    logger.info("Building bot config with profile=%s, overrides=%s", args.profile, config_overrides)

    return BotConfig.from_profile(args.profile, **config_overrides)


def order_tooling_requested(args: argparse.Namespace) -> bool:
    """
    Check if any order tooling command was requested.

    Args:
        args: Parsed argument namespace

    Returns:
        True if order tooling command was requested
    """
    return any([args.preview_order, args.edit_order_preview, args.apply_order_edit])
