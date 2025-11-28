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

from gpt_trader.orchestration.bootstrap import bot_from_profile, build_bot
from gpt_trader.orchestration.configuration import BotConfig

if TYPE_CHECKING:
    from gpt_trader.orchestration.service_registry import ServiceRegistry
    from gpt_trader.orchestration.trading_bot import TradingBot


@pytest.fixture
def dev_bot() -> Generator[tuple[TradingBot, ServiceRegistry]]:
    """Create a TradingBot with dev profile (DeterministicBroker).

    Yields:
        Tuple of (TradingBot, ServiceRegistry) for test access.
    """
    bot, registry = bot_from_profile("dev")

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

    Uses short_ma=3, long_ma=5 and interval=1 for rapid signal generation.

    Yields:
        Tuple of (TradingBot, ServiceRegistry) for test access.
    """
    config = BotConfig.from_profile(
        "dev",
        symbols=["BTC-USD"],
        short_ma=3,
        long_ma=5,
        interval=1,  # Fast interval for testing
        mock_broker=True,
    )
    bot, registry = build_bot(config)

    # Ensure clean state for risk manager
    if bot.risk_manager:
        bot.risk_manager.set_reduce_only_mode(False, reason="integration_test_setup")

    yield bot, registry

    # Cleanup
    try:
        asyncio.get_event_loop().run_until_complete(bot.shutdown())
    except Exception:
        pass
