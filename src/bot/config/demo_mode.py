"""Demo Mode Configuration for GPT-Trader.

This module provides demo mode functionality that allows running the application
without real API credentials for testing and demonstration purposes.
"""

import os


class DemoModeConfig:
    """Configuration for demo mode operation."""

    @staticmethod
    def is_demo_mode() -> bool:
        """Check if the application is running in demo mode."""
        return os.getenv("DEMO_MODE", "false").lower() in ("true", "1", "yes") or os.getenv(
            "ALPACA_API_KEY_ID", ""
        ).startswith("DEMO_")

    @staticmethod
    def get_demo_credentials() -> dict[str, str]:
        """Get demo credentials for testing."""
        return {
            "ALPACA_API_KEY_ID": "DEMO_KEY_FOR_TESTING",
            "ALPACA_API_SECRET_KEY": "DEMO_SECRET_FOR_TESTING",
            "ALPACA_PAPER_BASE_URL": "https://paper-api.alpaca.markets",
        }

    @staticmethod
    def validate_for_demo() -> tuple[bool, str | None]:
        """Validate that the system can run in demo mode.

        Returns:
            (is_valid, error_message)
        """
        if not DemoModeConfig.is_demo_mode():
            return True, None  # Not in demo mode, proceed normally

        # In demo mode, we can skip certain validations
        return True, None

    @staticmethod
    def get_demo_warning() -> str:
        """Get warning message for demo mode."""
        return """
╭─────────────────────────────────────────────────────────────────────╮
│                         🎮 DEMO MODE ACTIVE                         │
├─────────────────────────────────────────────────────────────────────┤
│ Running with mock credentials. Live trading is disabled.           │
│ • Backtesting: ✅ Available with historical data                   │
│ • Paper Trading: ❌ Disabled (requires real API keys)              │
│ • Live Trading: ❌ Disabled (requires real API keys)               │
│                                                                     │
│ To use full features, set up real Alpaca API credentials:          │
│ 1. Sign up at https://alpaca.markets                               │
│ 2. Get your API keys from the dashboard                            │
│ 3. Add them to .env.local file                                     │
╰─────────────────────────────────────────────────────────────────────╯
"""


def setup_demo_mode() -> None:
    """Set up demo mode if enabled."""
    if DemoModeConfig.is_demo_mode():
        # Set demo credentials if not already set
        demo_creds = DemoModeConfig.get_demo_credentials()
        for key, value in demo_creds.items():
            if not os.getenv(key) or os.getenv(key).startswith("DEMO_"):
                os.environ[key] = value


def check_demo_mode_restrictions(operation: str) -> tuple[bool, str | None]:
    """Check if an operation is allowed in demo mode.

    Args:
        operation: The operation to check (e.g., "backtest", "paper_trade", "live_trade")

    Returns:
        (is_allowed, restriction_message)
    """
    if not DemoModeConfig.is_demo_mode():
        return True, None  # Not in demo mode, all operations allowed

    # Define what's allowed in demo mode
    allowed_operations = {
        "backtest",
        "optimize",
        "walk_forward",
        "help",
        "menu",
        "dashboard",
        "wizard",
    }

    restricted_operations = {
        "paper": "Paper trading requires real Alpaca API credentials",
        "live": "Live trading requires real Alpaca API credentials",
        "deploy": "Deployment requires real Alpaca API credentials",
    }

    if operation in allowed_operations:
        return True, None
    elif operation in restricted_operations:
        return False, restricted_operations[operation]
    else:
        # Unknown operation, allow by default
        return True, None
