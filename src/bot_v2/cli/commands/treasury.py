"""Treasury helper commands for converting and moving funds."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace

from bot_v2.cli import options, services

_CONFIG_SKIP_KEYS = {
    "treasury_command",
    "from_asset",
    "to_asset",
    "from_portfolio",
    "to_portfolio",
    "amount",
}


def register(subparsers) -> None:
    parser = subparsers.add_parser("treasury", help="Treasury utilities")
    options.add_profile_option(parser)
    treasury_subparsers = parser.add_subparsers(dest="treasury_command", required=True)

    convert = treasury_subparsers.add_parser("convert", help="Convert between assets")
    options.add_profile_option(convert)
    convert.add_argument(
        "--from",
        "--from-asset",
        dest="from_asset",
        required=True,
        help="Source asset symbol",
    )
    convert.add_argument(
        "--to",
        "--to-asset",
        dest="to_asset",
        required=True,
        help="Destination asset symbol",
    )
    options.add_treasury_arguments(convert)
    convert.set_defaults(handler=_handle_convert)

    move = treasury_subparsers.add_parser("move", help="Move funds between portfolios")
    options.add_profile_option(move)
    move.add_argument(
        "--from-portfolio",
        dest="from_portfolio",
        required=True,
        help="Source portfolio identifier",
    )
    move.add_argument(
        "--to-portfolio",
        dest="to_portfolio",
        required=True,
        help="Destination portfolio identifier",
    )
    options.add_treasury_arguments(move)
    move.set_defaults(handler=_handle_move)


def _handle_convert(args: Namespace) -> int:
    config = services.build_config_from_args(args, skip=_CONFIG_SKIP_KEYS)
    bot = services.instantiate_bot(config)
    try:
        payload = {"from": args.from_asset, "to": args.to_asset, "amount": args.amount}
        result = bot.account_manager.convert(payload, commit=True)
        print(json.dumps(result, indent=2, default=str))
    finally:
        asyncio.run(bot.shutdown())
    return 0


def _handle_move(args: Namespace) -> int:
    config = services.build_config_from_args(args, skip=_CONFIG_SKIP_KEYS)
    bot = services.instantiate_bot(config)
    try:
        payload = {
            "from_portfolio": args.from_portfolio,
            "to_portfolio": args.to_portfolio,
            "amount": args.amount,
        }
        result = bot.account_manager.move_funds(payload)
        print(json.dumps(result, indent=2, default=str))
    finally:
        asyncio.run(bot.shutdown())
    return 0
