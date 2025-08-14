"""
Enhanced pytest configuration with parallel execution support and comprehensive fixtures.

This provides core fixtures, test isolation, and performance benchmarking.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import shutil
from datetime import datetime
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")

    # Generate realistic market data
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        },
        index=dates,
    )

    return data


@pytest.fixture
def test_symbols():
    """List of test symbols."""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]


@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio data for testing."""
    return {
        "AAPL": {
            "quantity": 100,
            "market_value": 15000.0,
            "weight": 0.15,
            "current_price": 150.0,
            "entry_price": 140.0,
        },
        "GOOGL": {
            "quantity": 50,
            "market_value": 7500.0,
            "weight": 0.075,
            "current_price": 150.0,
            "entry_price": 145.0,
        },
        "MSFT": {
            "quantity": 75,
            "market_value": 22500.0,
            "weight": 0.225,
            "current_price": 300.0,
            "entry_price": 280.0,
        },
    }


@pytest.fixture
def sample_returns():
    """Generate sample return series."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.001, 0.02, len(dates)), index=dates)
    return returns


@pytest.fixture
def mock_broker():
    """Create a mock broker for testing."""
    broker = Mock()

    # Mock account
    account = Mock()
    account.equity = 100000.0
    account.cash = 50000.0
    account.buying_power = 50000.0
    broker.get_account.return_value = account

    # Mock positions
    positions = []
    symbols = ["AAPL", "GOOGL", "MSFT"]
    for i, symbol in enumerate(symbols):
        position = Mock()
        position.symbol = symbol
        position.qty = 100 + i * 25
        position.market_value = 15000.0 + i * 5000.0
        position.current_price = 150.0 + i * 50.0
        position.avg_entry_price = 140.0 + i * 45.0
        positions.append(position)

    broker.get_positions.return_value = positions

    return broker


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("PYTHONPATH", "src")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


# Performance testing fixtures
@pytest.fixture
def large_dataset():
    """Create a large dataset for performance testing."""
    np.random.seed(42)
    n_samples = 10000
    n_features = 100

    data = np.random.randn(n_samples, n_features)
    return pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_features)])


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {"min_rounds": 10, "max_time": 10.0, "warmup": True, "disable_gc": True}


# Parallel execution support
@pytest.fixture(scope="function")
def isolated_test_dir(tmp_path):
    """Create isolated directory for each test (thread-safe)."""
    test_dir = tmp_path / f"test_{datetime.now().timestamp()}"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


# Market scenario fixtures
@pytest.fixture
def bull_market_data():
    """Generate bull market scenario data."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    # Positive drift with low volatility
    returns = np.random.normal(0.002, 0.01, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {"Close": prices, "Volume": np.random.randint(5000000, 15000000, len(dates))}, index=dates
    )


@pytest.fixture
def bear_market_data():
    """Generate bear market scenario data."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    # Negative drift with high volatility
    returns = np.random.normal(-0.001, 0.025, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {"Close": prices, "Volume": np.random.randint(10000000, 30000000, len(dates))}, index=dates
    )


@pytest.fixture
def volatile_market_data():
    """Generate volatile market scenario data."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    np.random.seed(42)
    # No drift but high volatility
    returns = np.random.normal(0, 0.04, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {"Close": prices, "Volume": np.random.randint(15000000, 50000000, len(dates))}, index=dates
    )


# Strategy factory fixtures
@pytest.fixture
def strategy_factory():
    """Factory for creating test strategies."""

    def create_strategy(strategy_type="demo_ma", config=None):
        mock_strategy = MagicMock()
        mock_strategy.name = strategy_type
        mock_strategy.config = config or {}
        mock_strategy.generate_signals = MagicMock(
            return_value={
                "signal": "BUY",
                "confidence": 0.8,
                "stop_loss": 95.0,
                "take_profit": 105.0,
            }
        )
        return mock_strategy

    return create_strategy


# Mock data source fixtures
@pytest.fixture
def mock_yfinance():
    """Mock yfinance data source."""
    mock = MagicMock()
    mock.download.return_value = pd.DataFrame(
        {
            "Open": [100, 101, 102],
            "High": [102, 103, 104],
            "Low": [99, 100, 101],
            "Close": [101, 102, 103],
            "Volume": [1000000, 1100000, 1200000],
        }
    )
    return mock


# Test categorization markers
def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line(
        "markers", "fast: marks tests as fast (deselect with '-m \"not fast\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "smoke: marks tests as smoke tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
