"""Declarative argument specifications for CLI parser.

Provides ArgumentSpec dataclasses and ArgumentGroupRegistrar
to replace repetitive parser.add_argument() calls with clean,
testable argument definitions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class ArgumentSpec:
    """Declarative specification for a single CLI argument."""

    name: str
    action: str | None = None
    type: type | None = None
    default: Any = None
    nargs: str | int | None = None
    choices: list[str] | None = None
    metavar: str | None = None
    dest: str | None = None
    help: str = ""

    def add_to_parser(self, parser: argparse.ArgumentParser) -> None:
        """Add this argument to the given parser."""
        kwargs = {"help": self.help}

        if self.action is not None:
            kwargs["action"] = self.action
        if self.type is not None:
            kwargs["type"] = self.type
        if self.default is not None:
            kwargs["default"] = self.default
        if self.nargs is not None:
            kwargs["nargs"] = self.nargs
        if self.choices is not None:
            kwargs["choices"] = self.choices
        if self.metavar is not None:
            kwargs["metavar"] = self.metavar
        if self.dest is not None:
            kwargs["dest"] = self.dest

        parser.add_argument(self.name, **kwargs)


# ============================================================================
# Bot Configuration Arguments
# ============================================================================

BOT_CONFIG_ARGS = [
    ArgumentSpec(
        name="--profile",
        type=str,
        default="dev",
        choices=["dev", "demo", "prod", "canary", "spot"],
        help="Configuration profile",
    ),
    ArgumentSpec(
        name="--dry-run",
        action="store_true",
        help="Run without placing real orders",
    ),
    ArgumentSpec(
        name="--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade (e.g., BTC-PERP ETH-PERP)",
    ),
    ArgumentSpec(
        name="--interval",
        type=int,
        help="Update interval in seconds",
    ),
    ArgumentSpec(
        name="--leverage",
        dest="target_leverage",
        type=int,
        help="Target leverage",
    ),
    ArgumentSpec(
        name="--reduce-only",
        dest="reduce_only_mode",
        action="store_true",
        help="Enable reduce-only mode",
    ),
    ArgumentSpec(
        name="--tif",
        dest="time_in_force",
        type=str,
        choices=["GTC", "IOC", "FOK"],
        help="Time in force policy (GTC/IOC/FOK)",
    ),
    ArgumentSpec(
        name="--enable-preview",
        dest="enable_order_preview",
        action="store_true",
        help="Enable order preview before placement",
    ),
    ArgumentSpec(
        name="--account-interval",
        dest="account_telemetry_interval",
        type=int,
        help="Account telemetry interval in seconds",
    ),
    ArgumentSpec(
        name="--max-trade-value",
        dest="max_trade_value",
        type=Decimal,
        help="Maximum USD notional value per trade (0 = no limit)",
    ),
    ArgumentSpec(
        name="--symbol-position-caps",
        type=str,
        help="Per-symbol position caps as SYMBOL:CAP pairs (e.g., BTC-USD:50000 ETH-USD:30000)",
        nargs="+",
    ),
    ArgumentSpec(
        name="--streaming-rest-poll-interval",
        dest="streaming_rest_poll_interval",
        type=float,
        help="REST polling interval in seconds when WebSocket streaming is unavailable",
    ),
]


# ============================================================================
# Account Management Arguments
# ============================================================================

ACCOUNT_ARGS = [
    ArgumentSpec(
        name="--account-snapshot",
        action="store_true",
        help="Print account telemetry snapshot and exit",
    ),
]


# ============================================================================
# Convert Command Arguments
# ============================================================================

CONVERT_ARGS = [
    ArgumentSpec(
        name="--convert",
        metavar="FROM:TO:AMOUNT",
        help="Perform a convert trade and exit",
    ),
]


# ============================================================================
# Move Funds Command Arguments
# ============================================================================

MOVE_FUNDS_ARGS = [
    ArgumentSpec(
        name="--move-funds",
        metavar="FROM:TO:AMOUNT",
        help="Move funds between portfolios and exit",
    ),
]


# ============================================================================
# Order Tooling Arguments
# ============================================================================

ORDER_TOOLING_ARGS = [
    ArgumentSpec(
        name="--preview-order",
        action="store_true",
        help="Preview a new order and exit",
    ),
    ArgumentSpec(
        name="--edit-order-preview",
        metavar="ORDER_ID",
        help="Preview edits for ORDER_ID and exit",
    ),
    ArgumentSpec(
        name="--apply-order-edit",
        metavar="ORDER_ID:PREVIEW_ID",
        help="Apply order edit using preview id and exit",
    ),
    ArgumentSpec(
        name="--order-symbol",
        help="Symbol for order preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-side",
        choices=["buy", "sell"],
        help="Order side for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-type",
        choices=["market", "limit", "stop_limit"],
        help="Order type for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-quantity",
        dest="order_quantity",
        type=Decimal,
        help="Order quantity for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-price",
        type=Decimal,
        help="Limit price for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-stop",
        type=Decimal,
        help="Stop price for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-tif",
        choices=["GTC", "IOC", "FOK"],
        help="Time in force for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-client-id",
        help="Client order id for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-reduce-only",
        action="store_true",
        help="Set reduce_only flag for preview/edit commands",
    ),
    ArgumentSpec(
        name="--order-leverage",
        type=int,
        help="Leverage override for preview/edit commands",
    ),
]


# ============================================================================
# Development Arguments
# ============================================================================

DEV_ARGS = [
    ArgumentSpec(
        name="--dev-fast",
        action="store_true",
        help="Run single cycle and exit (for smoke tests)",
    ),
]


# ============================================================================
# Argument Group Registrar
# ============================================================================


class ArgumentGroupRegistrar:
    """Registers argument groups to a parser."""

    @staticmethod
    def register_all(parser: argparse.ArgumentParser) -> None:
        """Register all argument groups to the parser."""
        ArgumentGroupRegistrar.register_bot_config(parser)
        ArgumentGroupRegistrar.register_account(parser)
        ArgumentGroupRegistrar.register_convert(parser)
        ArgumentGroupRegistrar.register_move_funds(parser)
        ArgumentGroupRegistrar.register_order_tooling(parser)
        ArgumentGroupRegistrar.register_dev(parser)

    @staticmethod
    def register_bot_config(parser: argparse.ArgumentParser) -> None:
        """Register bot configuration arguments."""
        for spec in BOT_CONFIG_ARGS:
            spec.add_to_parser(parser)

    @staticmethod
    def register_account(parser: argparse.ArgumentParser) -> None:
        """Register account management arguments."""
        for spec in ACCOUNT_ARGS:
            spec.add_to_parser(parser)

    @staticmethod
    def register_convert(parser: argparse.ArgumentParser) -> None:
        """Register convert command arguments."""
        for spec in CONVERT_ARGS:
            spec.add_to_parser(parser)

    @staticmethod
    def register_move_funds(parser: argparse.ArgumentParser) -> None:
        """Register move funds command arguments."""
        for spec in MOVE_FUNDS_ARGS:
            spec.add_to_parser(parser)

    @staticmethod
    def register_order_tooling(parser: argparse.ArgumentParser) -> None:
        """Register order tooling arguments."""
        for spec in ORDER_TOOLING_ARGS:
            spec.add_to_parser(parser)

    @staticmethod
    def register_dev(parser: argparse.ArgumentParser) -> None:
        """Register development arguments."""
        for spec in DEV_ARGS:
            spec.add_to_parser(parser)
