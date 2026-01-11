from __future__ import annotations

import pytest
from tests.unit.gpt_trader.tui.factories import (
    BotStatusFactory,
    MarketDataFactory,
    OrderFactory,
    PositionFactory,
    TradeFactory,
    TuiStateFactory,
)


@pytest.fixture
def bot_status_factory():
    """Provides access to BotStatusFactory for creating test BotStatus objects."""
    return BotStatusFactory


@pytest.fixture
def tui_state_factory():
    """Provides access to TuiStateFactory for creating test TuiState objects."""
    return TuiStateFactory


@pytest.fixture
def market_data_factory():
    """Provides access to MarketDataFactory for creating test market data."""
    return MarketDataFactory


@pytest.fixture
def position_factory():
    """Provides access to PositionFactory for creating test positions."""
    return PositionFactory


@pytest.fixture
def order_factory():
    """Provides access to OrderFactory for creating test orders."""
    return OrderFactory


@pytest.fixture
def trade_factory():
    """Provides access to TradeFactory for creating test trades."""
    return TradeFactory
