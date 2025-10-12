"""Account-related CLI commands."""

from __future__ import annotations

import asyncio
import json
from argparse import Namespace
from typing import Any

from bot_v2.cli import options, services


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser("account", help="Account utilities")
    options.add_profile_option(parser)
    account_subparsers = parser.add_subparsers(dest="account_command", required=True)

    snapshot = account_subparsers.add_parser("snapshot", help="Print an account snapshot")
    options.add_profile_option(snapshot)
    snapshot.set_defaults(handler=_handle_snapshot)


def _handle_snapshot(args: Namespace) -> int:
    config = services.build_config_from_args(args, skip={"account_command"})
    bot = services.instantiate_bot(config)
    try:
        telemetry = bot.account_telemetry
        if telemetry is None or not telemetry.supports_snapshots():
            raise RuntimeError("Account snapshot telemetry is not available for this broker")

        snapshot = telemetry.collect_snapshot()
        print(json.dumps(snapshot, indent=2, default=str))
    finally:
        asyncio.run(bot.shutdown())

    return 0
