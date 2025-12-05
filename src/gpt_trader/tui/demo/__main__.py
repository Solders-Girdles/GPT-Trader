"""
Demo TUI runner for testing the trading interface.

Run with: python -m gpt_trader.tui.demo
"""

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.demo.demo_bot import DemoBot
from gpt_trader.tui.helpers import run_tui_app_with_cleanup
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="demo")


def main() -> None:
    """Run the demo TUI."""
    # Configure logging for TUI mode BEFORE any other initialization
    from gpt_trader.logging.setup import configure_logging

    configure_logging(tui_mode=True)

    logger.info("Starting TUI Demo Mode")

    # Create demo bot
    demo_bot = DemoBot()

    # Create and run TUI
    app = TraderApp(bot=demo_bot)
    run_tui_app_with_cleanup(app)


if __name__ == "__main__":
    main()
