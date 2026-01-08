"""Shared fixtures for integration tests.

These fixtures provide real component wiring with DeterministicBroker
for testing integration paths without external API calls.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Generator
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

# Force mock broker for all integration tests
os.environ.setdefault("PERPS_FORCE_MOCK", "1")

from gpt_trader.app.config import BotConfig
from gpt_trader.app.container import (
    ApplicationContainer,
    clear_application_container,
    set_application_container,
)
from gpt_trader.features.brokerages.mock import DeterministicBroker
from gpt_trader.features.live_trade.strategies.perps_baseline import PerpsStrategyConfig
from gpt_trader.orchestration.bootstrap import bot_from_profile, build_bot
from gpt_trader.persistence.event_store import EventStore

if TYPE_CHECKING:
    from gpt_trader.features.live_trade.bot import TradingBot


@pytest.fixture
def integration_config() -> BotConfig:
    """Create BotConfig for integration tests with deterministic broker.

    Settings optimized for fast, deterministic tests:
    - mock_broker=True: Use DeterministicBroker
    - dry_run=True: Don't actually trade
    - enable_order_preview=False: Skip preview calls
    - Small interval for fast execution
    """
    return BotConfig(
        symbols=["BTC-USD"],
        interval=0.01,  # Fast interval
        mock_broker=True,
        dry_run=True,
        enable_order_preview=False,
        perps_enable_streaming=False,
        perps_position_fraction=0.04,  # 4% position sizing
    )


@pytest.fixture
def integration_container(integration_config: BotConfig) -> Generator[ApplicationContainer]:
    """Create ApplicationContainer and register globally for integration tests.

    Sets up the container for use with get_application_container() and
    clears it on teardown.
    """
    container = ApplicationContainer(integration_config)
    set_application_container(container)
    yield container
    clear_application_container()


@pytest.fixture
def deterministic_broker() -> DeterministicBroker:
    """Create a DeterministicBroker with default settings for integration tests.

    Returns a mock broker that:
    - Always returns deterministic quotes (BTC=$50000, ETH=$3000)
    - Immediately fills all orders
    - Has $100,000 starting equity
    """
    broker = DeterministicBroker(equity=Decimal("100000"))
    # Set up default mark prices
    broker.set_mark("BTC-USD", Decimal("50000"))
    broker.set_mark("ETH-USD", Decimal("3000"))
    return broker


@pytest.fixture
def fresh_event_store() -> EventStore:
    """Create a fresh in-memory EventStore for integration tests."""
    return EventStore()


@pytest.fixture
def dev_bot() -> Generator[TradingBot]:
    """Create a TradingBot with dev profile (DeterministicBroker).

    Yields:
        TradingBot: A fully configured trading bot.
        Access services directly via bot.container.<service>.
    """
    # Create config and container
    config = BotConfig.from_profile("dev", symbols=["BTC-USD"], mock_broker=True)
    container = ApplicationContainer(config)
    set_application_container(container)

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
    clear_application_container()


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

    # Register container globally
    container = ApplicationContainer(config)
    set_application_container(container)

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
    clear_application_container()
