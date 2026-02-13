"""Treasury helper commands for converting and moving funds."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from typing import Any, Protocol, cast, runtime_checkable

from gpt_trader.cli import options, services
from gpt_trader.cli.response import CliErrorCode, CliResponse

_CONFIG_SKIP_KEYS = {
    "treasury_command",
    "from_asset",
    "to_asset",
    "from_portfolio",
    "to_portfolio",
    "amount",
}


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("treasury", help="Treasury utilities")
    options.add_profile_option(parser, allow_missing_default=True)
    treasury_subparsers = parser.add_subparsers(dest="treasury_command", required=True)

    convert = treasury_subparsers.add_parser("convert", help="Convert between assets")
    options.add_profile_option(convert, allow_missing_default=True)
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
    options.add_output_options(convert, include_quiet=False)
    convert.set_defaults(handler=_handle_convert, subcommand="convert")

    move = treasury_subparsers.add_parser("move", help="Move funds between portfolios")
    options.add_profile_option(move, allow_missing_default=True)
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
    options.add_output_options(move, include_quiet=False)
    move.set_defaults(handler=_handle_move, subcommand="move")


def _handle_convert(args: Namespace) -> CliResponse | int:
    output_format = getattr(args, "output_format", "text")
    command_name = "treasury convert"

    try:
        config = services.build_config_from_args(args, skip=_CONFIG_SKIP_KEYS)
        bot = services.instantiate_bot(config)
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=command_name,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Failed to initialize: {e}",
            )
        raise

    manager = bot.account_manager
    if manager is None or not isinstance(manager, TreasuryManager):
        if output_format == "json":
            return CliResponse.error_response(
                command=command_name,
                code=CliErrorCode.OPERATION_FAILED,
                message="Account manager does not support treasury operations",
            )
        raise RuntimeError("Account manager does not support treasury operations")

    treasury_manager = cast(TreasuryManager, manager)
    try:
        payload = {"from": args.from_asset, "to": args.to_asset, "amount": args.amount}
        result = treasury_manager.convert(payload, commit=True)

        if output_format == "json":
            return CliResponse.success_response(
                command=command_name,
                data={
                    "transaction": result,
                    "from_asset": args.from_asset,
                    "to_asset": args.to_asset,
                    "amount": args.amount,
                },
            )

        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        asyncio.run(bot.shutdown())


def _handle_move(args: Namespace) -> CliResponse | int:
    output_format = getattr(args, "output_format", "text")
    command_name = "treasury move"

    try:
        config = services.build_config_from_args(args, skip=_CONFIG_SKIP_KEYS)
        bot = services.instantiate_bot(config)
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=command_name,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Failed to initialize: {e}",
            )
        raise

    manager = bot.account_manager
    if manager is None or not isinstance(manager, TreasuryManager):
        if output_format == "json":
            return CliResponse.error_response(
                command=command_name,
                code=CliErrorCode.OPERATION_FAILED,
                message="Account manager does not support treasury operations",
            )
        raise RuntimeError("Account manager does not support treasury operations")

    treasury_manager = cast(TreasuryManager, manager)
    try:
        payload = {
            "from_portfolio": args.from_portfolio,
            "to_portfolio": args.to_portfolio,
            "amount": args.amount,
        }
        result = treasury_manager.move_funds(payload)

        if output_format == "json":
            return CliResponse.success_response(
                command=command_name,
                data={
                    "transaction": result,
                    "from_portfolio": args.from_portfolio,
                    "to_portfolio": args.to_portfolio,
                    "amount": args.amount,
                },
            )

        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        asyncio.run(bot.shutdown())


@runtime_checkable
class TreasuryManager(Protocol):
    def convert(self, payload: dict[str, Any], *, commit: bool = ...) -> Any: ...

    def move_funds(self, payload: dict[str, Any]) -> Any: ...
