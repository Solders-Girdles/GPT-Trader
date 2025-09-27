"""
Command Line Interface for the Perps Trading Bot.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import asdict
from decimal import Decimal

# Load environment variables from .env file
from dotenv import load_dotenv

# Preserve host-provided secrets; only fill gaps from .env
load_dotenv()

from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce
from bot_v2.logging_setup import configure_logging
from bot_v2.orchestration.bootstrap import build_bot
from bot_v2.orchestration.configuration import BotConfig

# Configure logging (rotating files + console)
configure_logging()
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Perpetuals Trading Bot")

    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        choices=["dev", "demo", "prod", "canary", "spot"],
        help="Configuration profile",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run without placing real orders")
    parser.add_argument(
        "--symbols", type=str, nargs="+", help="Symbols to trade (e.g., BTC-PERP ETH-PERP)"
    )
    parser.add_argument("--interval", type=int, help="Update interval in seconds")
    parser.add_argument("--leverage", dest="target_leverage", type=int, help="Target leverage")
    # Map flag to BotConfig.reduce_only_mode so it takes effect
    parser.add_argument(
        "--reduce-only",
        dest="reduce_only_mode",
        action="store_true",
        help="Enable reduce-only mode",
    )
    # Optional TIF override maps to BotConfig.time_in_force (validated later)
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
    parser.add_argument(
        "--account-snapshot", action="store_true", help="Print account telemetry snapshot and exit"
    )
    parser.add_argument(
        "--convert", metavar="FROM:TO:AMOUNT", help="Perform a convert trade and exit"
    )
    parser.add_argument(
        "--move-funds", metavar="FROM:TO:AMOUNT", help="Move funds between portfolios and exit"
    )
    parser.add_argument(
        "--dev-fast", action="store_true", help="Run single cycle and exit (for smoke tests)"
    )
    parser.add_argument("--preview-order", action="store_true", help="Preview a new order and exit")
    parser.add_argument(
        "--edit-order-preview", metavar="ORDER_ID", help="Preview edits for ORDER_ID and exit"
    )
    parser.add_argument(
        "--apply-order-edit",
        metavar="ORDER_ID:PREVIEW_ID",
        help="Apply order edit using preview id and exit",
    )
    parser.add_argument("--order-symbol", help="Symbol for order preview/edit commands")
    parser.add_argument(
        "--order-side", choices=["buy", "sell"], help="Order side for preview/edit commands"
    )
    parser.add_argument(
        "--order-type",
        choices=["market", "limit", "stop_limit"],
        help="Order type for preview/edit commands",
    )
    parser.add_argument(
        "--order-qty", type=Decimal, help="Order quantity for preview/edit commands"
    )
    parser.add_argument("--order-price", type=Decimal, help="Limit price for preview/edit commands")
    parser.add_argument("--order-stop", type=Decimal, help="Stop price for preview/edit commands")
    parser.add_argument(
        "--order-tif", choices=["GTC", "IOC", "FOK"], help="Time in force for preview/edit commands"
    )
    parser.add_argument("--order-client-id", help="Client order id for preview/edit commands")
    parser.add_argument(
        "--order-reduce-only",
        action="store_true",
        help="Set reduce_only flag for preview/edit commands",
    )
    parser.add_argument(
        "--order-leverage", type=int, help="Leverage override for preview/edit commands"
    )

    args = parser.parse_args()

    # Validate symbol tokens (non-empty)
    if args.symbols:
        empty = [s for s in args.symbols if not str(s).strip()]
        if empty:
            parser.error("Symbols must be non-empty strings")

    # Allow PERPS_DEBUG=1 to elevate verbosity for selected modules
    if os.getenv("PERPS_DEBUG") == "1":
        logging.getLogger("bot_v2.features.brokerages.coinbase").setLevel(logging.DEBUG)
        logging.getLogger("bot_v2.orchestration").setLevel(logging.DEBUG)

    # Create config from parsed arguments
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
        "order_qty",
        "order_price",
        "order_stop",
        "order_tif",
        "order_client_id",
        "order_reduce_only",
        "order_leverage",
        "order_symbol",
    }
    config_overrides = {k: v for k, v in vars(args).items() if v is not None and k not in skip_keys}

    # Allow TRADING_SYMBOLS env to provide default symbols when --symbols not passed
    # Example: TRADING_SYMBOLS="BTC-PERP,ETH-PERP,SOL-PERP,XRP-PERP"
    if "symbols" not in config_overrides or not config_overrides.get("symbols"):
        env_syms = os.getenv("TRADING_SYMBOLS", "")
        if env_syms:
            syms = [s.strip() for s in env_syms.replace(";", ",").split(",") if s.strip()]
            if syms:
                config_overrides["symbols"] = syms
    config = BotConfig.from_profile(args.profile, **config_overrides)

    order_tooling_requested = any(
        [args.preview_order, args.edit_order_preview, args.apply_order_edit]
    )

    # Create and run the bot
    bot, _registry = build_bot(config)

    if args.account_snapshot:
        snapshot = bot._collect_account_snapshot()
        print(json.dumps(snapshot, indent=2))
        asyncio.run(bot.shutdown())
        return 0

    if order_tooling_requested:
        symbol = args.order_symbol.strip() if args.order_symbol else None
        if args.preview_order and not symbol:
            parser.error("--preview-order requires --order-symbol")
        if args.edit_order_preview and not symbol:
            parser.error("--edit-order-preview requires --order-symbol")

        try:
            if args.preview_order:
                if not args.order_side or not args.order_type or args.order_qty is None:
                    parser.error(
                        "--preview-order requires --order-side, --order-type, and --order-qty"
                    )
                side = OrderSide[args.order_side.upper()]
                order_type = OrderType[args.order_type.upper()]
                qty = Decimal(str(args.order_qty))
                tif = TimeInForce[args.order_tif.upper()] if args.order_tif else TimeInForce.GTC
                price = Decimal(str(args.order_price)) if args.order_price is not None else None
                stop = Decimal(str(args.order_stop)) if args.order_stop is not None else None
                data = bot.broker.preview_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    qty=qty,
                    price=price,
                    stop_price=stop,
                    tif=tif,
                    reduce_only=args.order_reduce_only,
                    leverage=args.order_leverage,
                    client_id=args.order_client_id,
                )
                print(json.dumps(data, indent=2, default=str))
                return 0

            if args.edit_order_preview:
                if not args.order_side or not args.order_type or args.order_qty is None:
                    parser.error(
                        "--edit-order-preview requires --order-side, --order-type, and --order-qty"
                    )
                side = OrderSide[args.order_side.upper()]
                order_type = OrderType[args.order_type.upper()]
                qty = Decimal(str(args.order_qty))
                tif = TimeInForce[args.order_tif.upper()] if args.order_tif else TimeInForce.GTC
                price = Decimal(str(args.order_price)) if args.order_price is not None else None
                stop = Decimal(str(args.order_stop)) if args.order_stop is not None else None
                preview = bot.broker.edit_order_preview(
                    order_id=args.edit_order_preview,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    qty=qty,
                    price=price,
                    stop_price=stop,
                    tif=tif,
                    new_client_id=args.order_client_id,
                    reduce_only=args.order_reduce_only,
                )
                print(json.dumps(preview, indent=2, default=str))
                return 0

            if args.apply_order_edit:
                try:
                    order_id, preview_id = (
                        part.strip() for part in args.apply_order_edit.split(":", 1)
                    )
                except ValueError:
                    parser.error("--apply-order-edit requires ORDER_ID:PREVIEW_ID")
                order = bot.broker.edit_order(order_id, preview_id)
                print(json.dumps(asdict(order), indent=2, default=str))
                return 0

            parser.error("Order tooling command provided but no action executed")
        finally:
            asyncio.run(bot.shutdown())

    if args.convert:
        try:
            from_asset, to_asset, amount = (part.strip() for part in args.convert.split(":", 2))
        except ValueError:
            parser.error("--convert requires format FROM:TO:AMOUNT")
        payload = {"from": from_asset, "to": to_asset, "amount": amount}
        result = bot.account_manager.convert(payload, commit=True)
        print(json.dumps(result, indent=2))
        asyncio.run(bot.shutdown())
        return 0

    if args.move_funds:
        try:
            from_uuid, to_uuid, amount = (part.strip() for part in args.move_funds.split(":", 2))
        except ValueError:
            parser.error("--move-funds requires format FROM_PORTFOLIO:TO_PORTFOLIO:AMOUNT")
        payload = {"from_portfolio": from_uuid, "to_portfolio": to_uuid, "amount": amount}
        result = bot.account_manager.move_funds(payload)
        print(json.dumps(result, indent=2))
        asyncio.run(bot.shutdown())
        return 0

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Signal received, shutting down...")
        bot.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(bot.run(single_cycle=args.dev_fast))
    except KeyboardInterrupt:
        logger.info("Shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
