"""Run command for the trading bot CLI."""

from __future__ import annotations

import asyncio
import signal
from argparse import Namespace
from types import FrameType
from typing import Any

from gpt_trader.app.config.validation import ConfigValidationError
from gpt_trader.app.container import clear_application_container
from gpt_trader.cli import options, services
from gpt_trader.config.constants import OTEL_ENABLED, OTEL_EXPORTER_ENDPOINT, OTEL_SERVICE_NAME
from gpt_trader.observability.tracing import init_tracing, shutdown_tracing
from gpt_trader.tui.helpers import run_tui_app_with_cleanup
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
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with mock data (requires --tui)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["winning", "losing", "volatile", "quiet", "risk_limit", "mixed"],
        default="mixed",
        help="Demo scenario to run (default: mixed)",
    )
    parser.set_defaults(handler=execute)


def execute(args: Namespace) -> int:
    # Handle demo mode
    if getattr(args, "demo", False):
        if not getattr(args, "tui", False):
            logger.error("--demo flag requires --tui to be set")
            return 1
        scenario = getattr(args, "scenario", "mixed")
        return _run_demo_tui(scenario)

    try:
        config = services.build_config_from_args(
            args,
            include=options.RUNTIME_CONFIG_KEYS,
            skip={"dev_fast", "tui", "demo"},
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

    # Initialize OpenTelemetry tracing if enabled
    if OTEL_ENABLED:
        init_tracing(
            service_name=OTEL_SERVICE_NAME,
            endpoint=OTEL_EXPORTER_ENDPOINT,
            enabled=True,
        )

    bot = services.instantiate_bot(config)

    if args.tui:
        return _run_tui(bot)

    return _run_bot(bot, single_cycle=args.dev_fast)


def _run_demo_tui(scenario: str = "mixed") -> int:
    """Run the TUI in demo mode with mock data."""
    try:
        from gpt_trader.tui.app import TraderApp
        from gpt_trader.tui.services.mode_service import create_bot_for_mode
    except ImportError:
        logger.error("TUI dependencies not installed. Run 'uv sync' to install textual.")
        return 1

    logger.info(f"Starting TUI in DEMO mode with '{scenario}' scenario")
    logger.info("No real exchanges or trading will occur")

    # Remove existing StreamHandlers that were added during CLI startup
    import logging

    root_logger = logging.getLogger()
    stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
    for handler in stream_handlers:
        root_logger.removeHandler(handler)
        handler.close()

    # Reconfigure logging in TUI mode (no console output to avoid corrupting display)
    from gpt_trader.logging.setup import configure_logging

    configure_logging(tui_mode=True)

    # Create demo bot using the shared factory
    demo_bot = create_bot_for_mode("demo", demo_scenario=scenario)
    app = TraderApp(bot=demo_bot, initial_mode="demo", demo_scenario=scenario)
    run_tui_app_with_cleanup(app)
    return 0


def _run_tui(bot: Any) -> int:
    """Run the bot with TUI."""
    try:
        from gpt_trader.tui.app import TraderApp
    except ImportError:
        logger.error("TUI dependencies not installed. Run 'uv sync' to install textual.")
        return 1

    # Remove existing StreamHandlers that were added during CLI startup
    import logging

    root_logger = logging.getLogger()
    stream_handlers = [h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)]
    for handler in stream_handlers:
        root_logger.removeHandler(handler)
        handler.close()

    # Reconfigure logging in TUI mode (no console output to avoid corrupting display)
    from gpt_trader.logging.setup import configure_logging

    configure_logging(tui_mode=True)

    # Pass bot with initial_mode set so live warning is shown if applicable
    app = TraderApp(bot=bot, initial_mode="direct")
    run_tui_app_with_cleanup(app)
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
    finally:
        # Flush any pending trace spans
        shutdown_tracing()
        # Clear container registry to prevent leaks between runs
        clear_application_container()

    return 0
