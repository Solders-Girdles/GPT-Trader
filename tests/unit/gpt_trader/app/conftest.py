from __future__ import annotations

import pytest

from gpt_trader.app.config import BotConfig


@pytest.fixture
def mock_config() -> BotConfig:
    """Create a minimal BotConfig for container tests."""
    return BotConfig(symbols=["BTC-USD"])
