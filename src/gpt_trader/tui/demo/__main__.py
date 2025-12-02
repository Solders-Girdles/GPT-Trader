"""
Demo TUI runner for testing the trading interface.

Run with: python -m gpt_trader.tui.demo
"""

from gpt_trader.tui.app import TraderApp
from gpt_trader.tui.demo.demo_bot import DemoBot
from gpt_trader.utilities.logging_patterns import get_logger

logger = get_logger(__name__, component="demo")


def main() -> None:
    """Run the demo TUI."""
    logger.info("Starting TUI Demo Mode")

    # Create demo bot
    demo_bot = DemoBot()

    # Create and run TUI
    app = TraderApp(bot=demo_bot)
    app.run()


if __name__ == "__main__":
    main()
