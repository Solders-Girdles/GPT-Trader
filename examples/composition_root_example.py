#!/usr/bin/env python3
"""
Example demonstrating the new composition root pattern.

This example shows how to create and run a PerpsBot using the new
dependency injection container approach introduced in Issue #91.
"""

from __future__ import annotations

import asyncio

from gpt_trader.app.container import create_application_container
from gpt_trader.orchestration.configuration import BotConfig, Profile


async def main() -> None:
    """Run the example."""
    print("=== Composition Root Example ===")

    # Create configuration
    print("Creating configuration...")
    config = BotConfig.from_profile(
        Profile.DEV,
        symbols=["BTC-USD"],
        mock_broker=True,  # Use mock broker for safety
    )

    print(f"Profile: {config.profile.value}")
    print(f"Symbols: {config.symbols}")
    print(f"Mock Broker: {config.mock_broker}")

    # Create application container
    print("\nCreating application container...")
    container = create_application_container(config)

    # Show container information
    print(f"Container created for profile: {container.config.profile.value}")
    print(f"Settings data dir: {container.settings.data_dir}")

    # Create PerpsBot from container
    print("\nCreating PerpsBot from container...")
    bot = container.create_perps_bot()

    print(f"Bot ID: {bot.bot_id}")
    print(f"Bot symbols: {bot.symbols}")
    print(f"Bot has container: {bot.container is not None}")

    # Show service information
    print("\nService Information:")
    print(f"Config controller: {type(bot.config_controller).__name__}")
    print(f"Event store: {type(bot.event_store).__name__}")
    print(f"Orders store: {type(bot.orders_store).__name__}")
    print(f"Broker: {type(bot.broker).__name__}")

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
