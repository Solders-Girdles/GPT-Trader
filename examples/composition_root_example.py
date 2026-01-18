#!/usr/bin/env python3
"""
Example demonstrating the new composition root pattern.

This example shows how to create and run a TradingBot using the new
dependency injection container approach introduced in Issue #91.
"""

from __future__ import annotations

import asyncio

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import create_application_container, set_application_container
from gpt_trader.config.types import Profile


async def main() -> None:
    """Run the example."""
    print("=== Composition Root Example ===")

    # Create configuration
    print("Creating configuration...")
    config = BotConfig.from_profile(
        Profile.DEV,
        symbols=["BTC-USD"],
        interval=1,
        mock_broker=True,  # Use mock broker for safety
    )

    print(f"Profile: {config.profile.value}")
    print(f"Symbols: {config.symbols}")
    print(f"Mock Broker: {config.mock_broker}")

    # Create application container
    print("\nCreating application container...")
    container = create_application_container(config)
    set_application_container(container)

    # Show container information
    print(f"Container created for profile: {container.config.profile.value}")
    print(f"Runtime storage dir: {container.runtime_paths.storage_dir}")

    # Create TradingBot from container
    print("\nCreating TradingBot from container...")
    bot = container.create_bot()

    bot_id = str(bot.context.bot_id or bot.config.profile or "live")
    print(f"Bot ID: {bot_id}")
    print(f"Bot symbols: {list(bot.context.symbols)}")
    print(f"Bot has container: {bot.container is not None}")

    # Show service information
    print("\nService Information:")
    print(f"Config controller: {type(container.config_controller).__name__}")
    print(f"Event store: {type(container.event_store).__name__}")
    print(f"Orders store: {type(container.orders_store).__name__}")
    print(f"Broker: {type(container.broker).__name__}")

    # Run a single trading cycle
    print("\nRunning single trading cycle...")
    try:
        await bot.run(single_cycle=True)
        print("Trading cycle completed successfully!")
    except Exception as e:
        print(f"Error during trading cycle: {e}")

    # Show container singleton behavior
    print("\nDemonstrating container singleton behavior...")
    broker1 = container.broker
    broker2 = container.broker
    print(f"Same broker instance: {broker1 is broker2}")

    event_store1 = container.event_store
    event_store2 = container.event_store
    print(f"Same event store instance: {event_store1 is event_store2}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
