"""Shared fixtures for integration tests.

These fixtures provide real component wiring with DeterministicBroker
for testing integration paths without external API calls.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

# Force mock broker for all integration tests
os.environ.setdefault("PERPS_FORCE_MOCK", "1")

from gpt_trader.features.live_trade.strategies.perps_baseline import PerpsStrategyConfig
from gpt_trader.orchestration.bootstrap import bot_from_profile, build_bot
from gpt_trader.orchestration.configuration import BotConfig

if TYPE_CHECKING:
    from gpt_trader.orchestration.service_registry import ServiceRegistry
    from gpt_trader.orchestration.trading_bot import TradingBot


@pytest.fixture
def dev_bot() -> Generator[tuple[TradingBot, ServiceRegistry]]:
    """Create a TradingBot with dev profile (DeterministicBroker).

    Yields:
        Tuple of (TradingBot, ServiceRegistry) for backward compatibility.
        Access container via bot.container for modern usage.
    """
    bot = bot_from_profile("dev")
    # Create registry from container for backward compatibility
    registry = bot.container.create_service_registry()

    # Ensure clean state for risk manager
    if bot.risk_manager:
        bot.risk_manager.set_reduce_only_mode(False, reason="integration_test_setup")

    yield bot, registry

    # Cleanup
    try:
        asyncio.get_event_loop().run_until_complete(bot.shutdown())
    except Exception:
        pass


@pytest.fixture
def fast_signal_bot() -> Generator[tuple[TradingBot, ServiceRegistry]]:
    """Create a TradingBot with fast MA crossover settings for signal testing.

    Uses short_ma_period=3, long_ma_period=5 and interval=1 for rapid signal generation.

    Yields:
        Tuple of (TradingBot, ServiceRegistry) for backward compatibility.
        Access container via bot.container for modern usage.
    """
    strategy = PerpsStrategyConfig(
        short_ma_period=3,
        long_ma_period=5,
    )
    config = BotConfig.from_profile(
        "dev",
        symbols=["BTC-USD"],
        strategy=strategy,
        interval=1,  # Fast interval for testing
        mock_broker=True,
    )
    bot = build_bot(config)
    # Create registry from container for backward compatibility
    registry = bot.container.create_service_registry()

    # Ensure clean state for risk manager
    if bot.risk_manager:
        bot.risk_manager.set_reduce_only_mode(False, reason="integration_test_setup")

    yield bot, registry

    # Cleanup
    try:
        asyncio.get_event_loop().run_until_complete(bot.shutdown())
    except Exception:
        pass
