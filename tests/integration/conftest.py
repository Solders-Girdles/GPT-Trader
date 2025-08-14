"""Integration test fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

# Import the actual modules from the project
try:
    from bot.strategy.demo_ma import DemoMAStrategy
    from bot.portfolio.allocator import PortfolioRules
    from bot.backtest.engine_portfolio import BacktestData
except ImportError:
    # Fallback imports if running from different directory
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from bot.strategy.demo_ma import DemoMAStrategy
    from bot.portfolio.allocator import PortfolioRules
    from bot.backtest.engine_portfolio import BacktestData


@pytest.fixture
def sample_market_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    data = {}
    for symbol in ["AAPL", "GOOGL", "MSFT"]:
        np.random.seed(hash(symbol) % 1000)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
        data[symbol] = pd.DataFrame(
            {
                "Open": prices * np.random.uniform(0.98, 1.02, len(dates)),
                "High": prices * np.random.uniform(1.01, 1.05, len(dates)),
                "Low": prices * np.random.uniform(0.95, 0.99, len(dates)),
                "Close": prices,
                "Volume": np.random.uniform(1e6, 1e8, len(dates)),
            },
            index=dates,
        )
    return data


@pytest.fixture
def test_strategy():
    """Create a test strategy instance"""
    return DemoMAStrategy(fast=10, slow=20)


@pytest.fixture
def portfolio_rules():
    """Create default portfolio rules"""
    return PortfolioRules(per_trade_risk_pct=0.01, max_positions=5, atr_k=2.0)


@pytest.fixture
def mock_data_source():
    """Mock data source for integration tests."""
    return Mock()


@pytest.fixture
def temp_database():
    """Temporary database for integration tests."""
    # TODO: Implement temporary database setup
    pass


@pytest.fixture
def sample_backtest_data(sample_market_data):
    """Create a BacktestData instance for testing"""
    symbols = list(sample_market_data.keys())
    dates_idx = sample_market_data[symbols[0]].index
    return BacktestData(
        symbols=symbols, data_map=sample_market_data, regime_ok=None, dates_idx=dates_idx
    )
