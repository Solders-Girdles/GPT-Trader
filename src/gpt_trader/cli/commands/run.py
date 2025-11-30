"""Run command for the trading bot CLI."""

from __future__ import annotations

import asyncio
import signal
from argparse import Namespace
from types import FrameType
from typing import Any

from gpt_trader.cli import options, services
from gpt_trader.orchestration.configuration import ConfigValidationError
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")


def register(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Run the trading loop",
        description="Perpetuals Trading Bot",
    )
    options.add_profile_option(parser)
    options.add_runtime_options(parser)
    parser.add_argument(
        "--config",
        type=str,
        metavar="PATH",
        help="Path to YAML config file (supports nested optimize output format)",
    )
    parser.add_argument("--dev-fast", action="store_true", help="Run single cycle and exit")
    parser.add_argument("--tui", action="store_true", help="Run with Terminal User Interface")
    parser.set_defaults(handler=execute)


def execute(args: Namespace) -> int:
    try:
        config = services.build_config_from_args(
            args,
            include=options.RUNTIME_CONFIG_KEYS,
            skip={"dev_fast", "tui"},
        )
    except ConfigValidationError as exc:
        message = str(exc)
        if "symbols overrides" in message:
            logger.error("Symbols must be non-empty")
        else:
            logger.error(message)
        return 1

    # Auto-reduce interval for fast development iteration
    if args.dev_fast and getattr(args, "interval", None) is None:
        config.interval = 1  # 1 second instead of default 60

    bot = services.instantiate_bot(config)

    if args.tui:
        return _run_tui(bot)

    return _run_bot(bot, single_cycle=args.dev_fast)


def _run_tui(bot: Any) -> int:
    """Run the bot with TUI."""
    try:
        from gpt_trader.tui.app import TraderApp
    except ImportError:
        logger.error("TUI dependencies not installed. Run 'uv sync' to install textual.")
        return 1

    app = TraderApp(bot)
    app.run()
    return 0


def _run_bot(bot: Any, *, single_cycle: bool) -> int:
    def signal_handler(sig: int, frame: FrameType | None) -> None:  # pragma: no cover - signal
        logger.info(f"Signal {sig} received, shutting down...", operation="shutdown")
        bot.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(bot.run(single_cycle=single_cycle))
    except KeyboardInterrupt:  # pragma: no cover - defensive
        logger.info("Shutdown complete.", status="stopped")

    return 0
