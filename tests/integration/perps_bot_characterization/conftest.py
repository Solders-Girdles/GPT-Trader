"""Shared fixtures for PerpsBot characterization tests."""

import pytest
from datetime import datetime, UTC
from unittest.mock import Mock

from bot_v2.orchestration.configuration import BotConfig, Profile
from bot_v2.features.brokerages.core.interfaces import Quote


@pytest.fixture
def minimal_config():
    """Minimal config that allows PerpsBot to initialize."""
    return BotConfig(
        profile=Profile.DEV,
        symbols=["BTC-USD"],
        update_interval=60,
        mock_broker=True,
    )


@pytest.fixture
def mock_quote():
    """Standard quote response."""
    quote = Mock(spec=Quote)
    quote.last = 50000.0
    quote.last_price = None
    quote.ts = datetime.now(UTC)
    return quote
