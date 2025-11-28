"""Account-related CLI commands."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from typing import Any

from gpt_trader.cli import options, services
from gpt_trader.cli.response import CliErrorCode, CliResponse

COMMAND_NAME = "account snapshot"


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("account", help="Account utilities")
    options.add_profile_option(parser)
    account_subparsers = parser.add_subparsers(dest="account_command", required=True)

    snapshot = account_subparsers.add_parser("snapshot", help="Print an account snapshot")
    options.add_profile_option(snapshot)
    options.add_output_options(snapshot, include_quiet=False)
    snapshot.set_defaults(handler=_handle_snapshot, subcommand="snapshot")


def _handle_snapshot(args: Namespace) -> CliResponse | int:
    output_format = getattr(args, "output_format", "text")

    try:
        config = services.build_config_from_args(args, skip={"account_command"})
        bot = services.instantiate_bot(config)
    except Exception as e:
        if output_format == "json":
            return CliResponse.error_response(
                command=COMMAND_NAME,
                code=CliErrorCode.CONFIG_INVALID,
                message=f"Failed to initialize: {e}",
            )
        raise

    try:
        telemetry = bot.account_telemetry
        if telemetry is None or not telemetry.supports_snapshots():
            if output_format == "json":
                return CliResponse.error_response(
                    command=COMMAND_NAME,
                    code=CliErrorCode.OPERATION_FAILED,
                    message="Account snapshot telemetry is not available for this broker",
                )
            raise RuntimeError("Account snapshot telemetry is not available for this broker")

        snapshot = telemetry.collect_snapshot()

        if output_format == "json":
            return CliResponse.success_response(
                command=COMMAND_NAME,
                data=snapshot,
            )

        print(json.dumps(snapshot, indent=2, default=str))
        return 0

    finally:
        asyncio.run(bot.shutdown())
