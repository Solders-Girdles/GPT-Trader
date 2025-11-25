"""Command line interface entry point for the trading bot."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from gpt_trader.config.runtime_settings import RuntimeSettings, load_runtime_settings
from gpt_trader.logging import configure_logging
from gpt_trader.utilities.logging_patterns import get_logger

# Preserve host-provided secrets; only fill gaps from .env
load_dotenv()

RUNTIME_SETTINGS: RuntimeSettings = load_runtime_settings()

# Configure logging (rotating files + console)
configure_logging(settings=RUNTIME_SETTINGS)
logger = get_logger(__name__, component="cli")

from . import services as _cli_services  # noqa: E402

_cli_services.OVERRIDE_SETTINGS = RUNTIME_SETTINGS

from gpt_trader.cli.commands import account, orders, report, run, treasury  # noqa: E402

COMMAND_NAMES = {"run", "account", "orders", "treasury", "report"}
__all__ = ["main", "RUNTIME_SETTINGS"]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    normalized = _ensure_command(argv if argv is not None else sys.argv[1:])
    args = parser.parse_args(normalized)
    _maybe_enable_debug_logging()

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.error("No command handler configured.")
    result = handler(args)
    return int(result) if result is not None else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coinbase Trading Bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run.register(subparsers)
    account.register(subparsers)
    orders.register(subparsers)
    treasury.register(subparsers)
    report.register(subparsers)

    return parser


def _ensure_command(argv: Sequence[str]) -> list[str]:
    if not argv:
        return ["run"]

    if any(token in {"-h", "--help"} for token in argv):
        if argv[0] in {"-h", "--help"}:
            return ["run", *argv]
        return list(argv)

    first = argv[0]
    if first in COMMAND_NAMES:
        return list(argv)

    if first.startswith("-"):
        return ["run", *argv]

    # Unrecognised token – default to the run command while preserving args
    return ["run", *argv]


def _maybe_enable_debug_logging() -> None:
    if _env_flag("COINBASE_TRADER_DEBUG") or _env_flag("PERPS_DEBUG"):
        logging.getLogger("gpt_trader.features.brokerages.coinbase").setLevel(logging.DEBUG)
        logging.getLogger("gpt_trader.orchestration").setLevel(logging.DEBUG)


def _env_flag(name: str) -> bool:
    value = RUNTIME_SETTINGS.raw_env.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
