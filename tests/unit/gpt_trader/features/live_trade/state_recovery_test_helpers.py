"""Shared fixtures for state recovery tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from gpt_trader.app.config import BotConfig
from gpt_trader.features.live_trade.engines.base import CoordinatorContext
from gpt_trader.features.live_trade.strategies.perps_baseline import PerpsStrategyConfig


@pytest.fixture
def mock_event_store() -> MagicMock:
    """Create a mock event store."""
    store = MagicMock()
    store.get_recent.return_value = []
    return store


@pytest.fixture
def bot_config() -> BotConfig:
    """Create a minimal bot config."""
    return BotConfig(
        symbols=["BTC-PERP", "ETH-PERP"],
        interval=60,
        strategy=PerpsStrategyConfig(),
    )


@pytest.fixture
def mock_config(bot_config: BotConfig) -> BotConfig:
    """Alias for the shared tests/conftest.py application_container fixture."""
    return bot_config


@pytest.fixture
def context_with_store(bot_config: BotConfig, mock_event_store: MagicMock) -> CoordinatorContext:
    """Create a context with event store."""
    return CoordinatorContext(
        config=bot_config,
        event_store=mock_event_store,
        bot_id="test-bot",
    )


@pytest.fixture
def context_without_store(bot_config: BotConfig) -> CoordinatorContext:
    """Create a context without event store."""
    return CoordinatorContext(
        config=bot_config,
        bot_id="test-bot",
    )
