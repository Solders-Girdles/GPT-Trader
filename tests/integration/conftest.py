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
    from gpt_trader.orchestration.trading_bot import TradingBot


@pytest.fixture
def dev_bot() -> Generator[TradingBot]:
    """Create a TradingBot with dev profile (DeterministicBroker).

    Yields:
        TradingBot: A fully configured trading bot.
        Access services directly via bot.container.<service>.
    """
    bot = bot_from_profile("dev")

    # Ensure clean state for risk manager
    if bot.risk_manager:
        bot.risk_manager.set_reduce_only_mode(False, reason="integration_test_setup")

    yield bot

    # Cleanup
    try:
        asyncio.get_event_loop().run_until_complete(bot.shutdown())
    except Exception:
        pass


@pytest.fixture
def fast_signal_bot() -> Generator[TradingBot]:
    """Create a TradingBot with fast MA crossover settings for signal testing.

    Uses short_ma_period=3, long_ma_period=5 and interval=1 for rapid signal generation.

    Yields:
        TradingBot: A fully configured trading bot.
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

    # Ensure clean state for risk manager
    if bot.risk_manager:
        bot.risk_manager.set_reduce_only_mode(False, reason="integration_test_setup")

    yield bot

    # Cleanup
    try:
        asyncio.get_event_loop().run_until_complete(bot.shutdown())
    except Exception:
        pass
