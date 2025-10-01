"""
Order tooling commands for the Perps Trading Bot CLI.

Provides order preview, edit preview, and edit application functionality.
"""

import argparse
import json
import logging
from dataclasses import asdict
from decimal import Decimal

from bot_v2.cli.handlers.shutdown import ensure_shutdown
from bot_v2.features.brokerages.core.interfaces import OrderSide, OrderType, TimeInForce
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

    symbol = args.order_symbol.strip() if args.order_symbol else None

    # Validate symbol is provided for commands that need it
    if args.preview_order and not symbol:
        parser.error("--preview-order requires --order-symbol")
    if args.edit_order_preview and not symbol:
        parser.error("--edit-order-preview requires --order-symbol")

    try:
        if args.preview_order:
            return _handle_preview_order(args, bot, parser, symbol)

        if args.edit_order_preview:
            return _handle_edit_order_preview(args, bot, parser, symbol)

        if args.apply_order_edit:
            return _handle_apply_order_edit(args, bot, parser)

        parser.error("Order tooling command provided but no action executed")
    except Exception as e:
        logger.error("Order tooling command failed: %s", e, exc_info=True)
        raise
    finally:
        ensure_shutdown(bot)


def _handle_preview_order(
    args: argparse.Namespace, bot: PerpsBot, parser: argparse.ArgumentParser, symbol: str
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
    logger.info("Previewing new order for symbol=%s", symbol)

    # Validate required arguments
    if not args.order_side or not args.order_type or args.order_quantity is None:
        parser.error("--preview-order requires --order-side, --order-type, and --order-quantity")

    # Parse order parameters
    side = OrderSide[args.order_side.upper()]
    order_type = OrderType[args.order_type.upper()]
    order_quantity = Decimal(str(args.order_quantity))
    tif = TimeInForce[args.order_tif.upper()] if args.order_tif else TimeInForce.GTC
    price = Decimal(str(args.order_price)) if args.order_price is not None else None
    stop = Decimal(str(args.order_stop)) if args.order_stop is not None else None

    logger.debug(
        "Order preview params: side=%s, type=%s, qty=%s, price=%s, stop=%s, tif=%s",
        side,
        order_type,
        order_quantity,
        price,
        stop,
        tif,
    )

    # Preview the order
    data = bot.broker.preview_order(
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=order_quantity,
        price=price,
        stop_price=stop,
        tif=tif,
        reduce_only=args.order_reduce_only,
        leverage=args.order_leverage,
        client_id=args.order_client_id,
    )

    logger.info("Order preview completed successfully")

    # Print preview as formatted JSON
    output = json.dumps(data, indent=2, default=str)
    print(output)

    return 0


def _handle_edit_order_preview(
    args: argparse.Namespace, bot: PerpsBot, parser: argparse.ArgumentParser, symbol: str
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
    order_id = args.edit_order_preview
    logger.info("Previewing order edit for order_id=%s, symbol=%s", order_id, symbol)

    # Validate required arguments
    if not args.order_side or not args.order_type or args.order_quantity is None:
        parser.error(
            "--edit-order-preview requires --order-side, --order-type, and --order-quantity"
        )

    # Parse order parameters
    side = OrderSide[args.order_side.upper()]
    order_type = OrderType[args.order_type.upper()]
    order_quantity = Decimal(str(args.order_quantity))
    tif = TimeInForce[args.order_tif.upper()] if args.order_tif else TimeInForce.GTC
    price = Decimal(str(args.order_price)) if args.order_price is not None else None
    stop = Decimal(str(args.order_stop)) if args.order_stop is not None else None

    logger.debug(
        "Edit preview params: side=%s, type=%s, qty=%s, price=%s, stop=%s, tif=%s",
        side,
        order_type,
        order_quantity,
        price,
        stop,
        tif,
    )

    # Preview the edit
    preview = bot.broker.edit_order_preview(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity=order_quantity,
        price=price,
        stop_price=stop,
        tif=tif,
        new_client_id=args.order_client_id,
        reduce_only=args.order_reduce_only,
    )

    logger.info("Order edit preview completed successfully")

    # Print preview as formatted JSON
    output = json.dumps(preview, indent=2, default=str)
    print(output)

    return 0


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
    logger.info("Applying order edit with arg=%s", args.apply_order_edit)

    # Parse ORDER_ID:PREVIEW_ID
    try:
        order_id, preview_id = (part.strip() for part in args.apply_order_edit.split(":", 1))
    except ValueError:
        parser.error("--apply-order-edit requires ORDER_ID:PREVIEW_ID")

    logger.info("Applying edit for order_id=%s, preview_id=%s", order_id, preview_id)

    # Apply the edit
    order = bot.broker.edit_order(order_id, preview_id)

    logger.info("Order edit applied successfully: order_id=%s", order.order_id)

    # Print result as formatted JSON
    output = json.dumps(asdict(order), indent=2, default=str)
    print(output)

    return 0
