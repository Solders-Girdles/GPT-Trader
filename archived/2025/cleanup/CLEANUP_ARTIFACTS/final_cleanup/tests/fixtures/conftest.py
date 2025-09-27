"""Shared fixtures for test suite."""

import pytest

from .factories import MarketDataFactory, PortfolioFactory, StrategyFactory, TradeFactory


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return MarketDataFactory.create_daily_data("AAPL", days=30)


@pytest.fixture
def sample_strategy():
    """Sample strategy for testing."""
    return StrategyFactory.create_ma_strategy()


@pytest.fixture
def sample_portfolio():
    """Sample portfolio for testing."""
    return PortfolioFactory.create_default_portfolio()


@pytest.fixture
def sample_trades():
    """Sample trades for testing."""
    return TradeFactory.create_sample_trades(count=10)
