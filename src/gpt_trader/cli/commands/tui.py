"""TUI command for launching the Terminal User Interface."""

from __future__ import annotations

from argparse import Namespace
from typing import Any

from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="cli")


def register(subparsers: Any) -> None:
    """Register the tui command."""
    parser = subparsers.add_parser(
        "tui",
        help="Launch Terminal User Interface",
        description="Launch the GPT-Trader Terminal User Interface with mode selection",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "paper", "read_only", "live"],
        help="Skip mode selection and launch directly into specified mode",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["winning", "losing", "volatile", "quiet", "risk_limit", "mixed"],
        default="mixed",
        help="Demo scenario to run when using --mode demo (default: mixed)",
    )
    parser.set_defaults(handler=execute)


def execute(args: Namespace) -> int:
    """Execute the tui command."""
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

    # Get mode from args or None to show selection screen
    initial_mode = getattr(args, "mode", None)
    scenario = getattr(args, "scenario", "mixed")

    logger.info("Starting Terminal User Interface")
    if initial_mode:
        logger.info(f"Launching directly into {initial_mode.upper()} mode")

    from gpt_trader.tui.helpers import run_tui_app_with_cleanup

    app = TraderApp(initial_mode=initial_mode, demo_scenario=scenario)
    run_tui_app_with_cleanup(app)
    return 0
