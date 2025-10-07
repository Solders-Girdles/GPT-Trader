"""Argument helpers for the bot CLI."""

from __future__ import annotations

from argparse import ArgumentParser

PROFILE_CHOICES = ["dev", "demo", "prod", "canary", "spot"]

# Keys that can be forwarded as BotConfig overrides for the run command.
RUNTIME_CONFIG_KEYS = {
    "dry_run",
    "symbols",
    "interval",
    "target_leverage",
    "reduce_only_mode",
    "time_in_force",
    "enable_order_preview",
    "account_telemetry_interval",
}


def add_profile_option(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        choices=PROFILE_CHOICES,
        help="Configuration profile",
    )


def add_runtime_options(parser: ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="Run without placing real orders")
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Symbols to trade (e.g., BTC-PERP ETH-PERP)",
    )
    parser.add_argument("--interval", type=int, help="Update interval in seconds")
    parser.add_argument("--leverage", dest="target_leverage", type=int, help="Target leverage")
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


def add_order_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--symbol",
        "--order-symbol",
        dest="symbol",
        required=True,
        help="Symbol for order operations",
    )
    parser.add_argument(
        "--side",
        "--order-side",
        required=True,
        choices=["buy", "sell"],
        help="Order side",
    )
    parser.add_argument(
        "--type",
        "--order-type",
        required=True,
        choices=["market", "limit", "stop_limit"],
        help="Order type",
    )
    parser.add_argument(
        "--quantity",
        "--order-quantity",
        required=True,
        type=str,
        help="Order quantity (interpreted as Decimal)",
    )
    parser.add_argument(
        "--price",
        "--order-price",
        dest="price",
        type=str,
        help="Limit price for limit/stop orders",
    )
    parser.add_argument(
        "--stop",
        "--order-stop",
        dest="stop",
        type=str,
        help="Stop price for stop orders",
    )
    parser.add_argument(
        "--tif",
        "--order-tif",
        choices=["GTC", "IOC", "FOK"],
        help="Time in force for the order",
    )
    parser.add_argument(
        "--client-id",
        "--order-client-id",
        dest="client_id",
        help="Optional client order identifier",
    )
    parser.add_argument(
        "--leverage",
        "--order-leverage",
        dest="leverage",
        type=int,
        help="Optional leverage override",
    )
    parser.add_argument(
        "--reduce-only",
        "--order-reduce-only",
        dest="reduce_only",
        action="store_true",
        help="Set reduce-only flag",
    )


def add_treasury_arguments(parser: ArgumentParser) -> None:
    parser.add_argument("--amount", required=True, help="Amount to convert or transfer")
