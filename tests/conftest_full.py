"""
Pytest configuration and shared fixtures for Phase 5 testing.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from bot.knowledge.strategy_knowledge_base import (
    StrategyContext,
    StrategyKnowledgeBase,
    StrategyMetadata,
    StrategyPerformance,
)
from bot.live.strategy_selector import SelectionConfig, SelectionMethod
from bot.meta_learning.regime_detection import MarketRegime, RegimeCharacteristics, RegimeDetector
from bot.monitor.alerts import AlertConfig, AlertSeverity
from bot.portfolio.optimizer import PortfolioConstraints
from bot.risk.manager import RiskLimits, StopLossConfig


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
            "Date": dates,
            "Open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
            "High": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            "Close": prices,
            "Volume": np.random.randint(1000000, 10000000, len(dates)),
        }
    )

    return data.set_index("Date")


@pytest.fixture
def sample_strategies():
    """Generate sample strategies for testing."""
    strategies = []

    for i in range(10):
        strategy = StrategyMetadata(
            strategy_id=f"strategy_{i:03d}",
            name=f"Test Strategy {i}",
            description=f"Test strategy {i} for testing",
            strategy_type=["trend_following", "mean_reversion", "momentum", "volatility"][i % 4],
            parameters={
                "lookback_period": 20 + i * 5,
                "threshold": 0.1 + i * 0.01,
                "param3": i * 2,
            },
            context=StrategyContext(
                market_regime=["trending", "sideways", "volatile", "crisis"][i % 4],
                time_period=["bull_market", "bear_market", "sideways_market"][i % 3],
                asset_class="equity",
                risk_profile=["conservative", "moderate", "aggressive"][i % 3],
                volatility_regime=["low", "medium", "high"][i % 3],
                correlation_regime=["low", "medium", "high"][i % 3],
            ),
            performance=StrategyPerformance(
                sharpe_ratio=0.5 + i * 0.2,
                cagr=0.05 + i * 0.02,
                max_drawdown=0.2 - i * 0.01,
                win_rate=0.4 + i * 0.03,
                consistency_score=0.5 + i * 0.03,
                n_trades=30 + i * 10,
                avg_trade_duration=5.0 + i * 0.5,
                profit_factor=1.1 + i * 0.05,
                calmar_ratio=0.8 + i * 0.1,
                sortino_ratio=1.0 + i * 0.1,
                information_ratio=0.5 + i * 0.1,
                beta=0.8 + i * 0.05,
                alpha=0.02 + i * 0.01,
            ),
            discovery_date=datetime.now() - timedelta(days=30 + i * 10),
            last_updated=datetime.now() - timedelta(days=i),
            usage_count=10 + i * 5,
            success_rate=0.6 + i * 0.02,
            tags=[f"tag_{i}", f"category_{i % 3}"],
            notes=f"Test strategy {i} notes",
        )
        strategies.append(strategy)

    return strategies


@pytest.fixture
def mock_knowledge_base(sample_strategies):
    """Create a mock knowledge base with sample strategies."""
    knowledge_base = Mock(spec=StrategyKnowledgeBase)
    knowledge_base.find_strategies.return_value = sample_strategies
    knowledge_base.get_strategy_recommendations.return_value = sample_strategies[:5]
    return knowledge_base


@pytest.fixture
def mock_regime_detector():
    """Create a mock regime detector."""
    detector = Mock(spec=RegimeDetector)

    # Mock regime characteristics
    regime_char = RegimeCharacteristics(
        regime=MarketRegime.TRENDING_UP,
        confidence=0.85,
        duration_days=15,
        volatility=0.18,
        trend_strength=0.25,
        correlation_level=0.3,
        volume_profile="normal",
        momentum_score=0.7,
        regime_features={"feature1": 0.5, "feature2": 0.3},
    )

    detector.detect_regime.return_value = regime_char
    detector._regime_to_context.return_value = StrategyContext(
        market_regime="trending",
        time_period="bull_market",
        asset_class="equity",
        risk_profile="moderate",
        volatility_regime="medium",
        correlation_regime="low",
    )
    detector._calculate_context_match.return_value = 0.8

    return detector


@pytest.fixture
def selection_config():
    """Create a test selection configuration."""
    return SelectionConfig(
        selection_method=SelectionMethod.HYBRID,
        max_strategies=5,
        min_confidence=0.6,
        min_sharpe=0.5,
        max_drawdown=0.15,
        regime_weight=0.3,
        performance_weight=0.4,
        confidence_weight=0.2,
        risk_weight=0.1,
        adaptation_weight=0.2,
        rebalance_interval=3600,
        lookback_days=30,
    )


@pytest.fixture
def portfolio_constraints():
    """Create test portfolio constraints."""
    return PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.4,
        max_sector_exposure=0.6,
        max_volatility=0.25,
        max_drawdown=0.15,
        target_return=0.12,
        risk_free_rate=0.02,
    )


@pytest.fixture
def risk_limits():
    """Create test risk limits."""
    return RiskLimits(
        max_portfolio_var=0.02,
        max_portfolio_drawdown=0.15,
        max_portfolio_volatility=0.25,
        max_portfolio_beta=1.2,
        max_position_size=0.1,
        max_sector_exposure=0.3,
        max_correlation=0.7,
        max_risk_per_trade=0.01,
        max_daily_loss=0.03,
        min_liquidity_ratio=0.1,
        max_illiquid_exposure=0.2,
    )


@pytest.fixture
def stop_loss_config():
    """Create test stop-loss configuration."""
    return StopLossConfig(
        stop_loss_pct=0.05, trailing_stop_pct=0.03, time_stop_days=30, breakeven_after_pct=0.02
    )


@pytest.fixture
def alert_config():
    """Create test alert configuration."""
    return AlertConfig(
        email_enabled=False,
        slack_enabled=False,
        discord_enabled=False,
        webhook_enabled=False,
        alert_cooldown_minutes=5,
        max_alerts_per_hour=10,
    )


@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio data for testing."""
    return {
        "AAPL": {
            "quantity": 100,
            "market_value": 15000.0,
            "portfolio_value": 100000.0,
            "weight": 0.15,
            "current_price": 150.0,
            "entry_price": 140.0,
        },
        "GOOGL": {
            "quantity": 50,
            "market_value": 7500.0,
            "portfolio_value": 100000.0,
            "weight": 0.075,
            "current_price": 150.0,
            "entry_price": 145.0,
        },
        "MSFT": {
            "quantity": 75,
            "market_value": 22500.0,
            "portfolio_value": 100000.0,
            "weight": 0.225,
            "current_price": 300.0,
            "entry_price": 280.0,
        },
    }


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


@pytest.fixture
def test_symbols():
    """List of test symbols."""
    return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]


@pytest.fixture
def sample_historical_returns():
    """Generate sample historical returns for testing."""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    np.random.seed(42)
    returns_data = {}

    for symbol in symbols:
        # Generate correlated returns
        base_returns = np.random.normal(0.0005, 0.02, len(dates))
        symbol_returns = base_returns + np.random.normal(0, 0.005, len(dates))
        returns_data[symbol] = pd.Series(symbol_returns, index=dates)

    return pd.DataFrame(returns_data)


@pytest.fixture
def sample_correlation_matrix():
    """Generate sample correlation matrix for testing."""
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

    # Create a realistic correlation matrix
    np.random.seed(42)
    n = len(symbols)
    correlation_matrix = np.eye(n)  # Start with identity matrix

    # Add some correlations
    for i in range(n):
        for j in range(i + 1, n):
            # Tech stocks have higher correlation
            if i < 4 and j < 4:  # First 4 are tech stocks
                corr = 0.6 + np.random.normal(0, 0.1)
            else:
                corr = 0.3 + np.random.normal(0, 0.1)

            corr = np.clip(corr, -0.9, 0.9)  # Clip to reasonable range
            correlation_matrix[i, j] = corr
            correlation_matrix[j, i] = corr

    return pd.DataFrame(correlation_matrix, index=symbols, columns=symbols)


@pytest.fixture
def sample_risk_scenarios():
    """Generate sample risk scenarios for testing."""
    return {
        "market_crash": {
            "description": "20% market crash scenario",
            "market_shock": -0.20,
            "volatility_multiplier": 2.0,
            "correlation_increase": 0.3,
        },
        "volatility_spike": {
            "description": "Volatility spike scenario",
            "market_shock": 0.0,
            "volatility_multiplier": 3.0,
            "correlation_increase": 0.2,
        },
        "liquidity_crisis": {
            "description": "Liquidity crisis scenario",
            "market_shock": -0.10,
            "volatility_multiplier": 1.5,
            "correlation_increase": 0.5,
            "liquidity_reduction": 0.7,
        },
        "normal_market": {
            "description": "Normal market conditions",
            "market_shock": 0.0,
            "volatility_multiplier": 1.0,
            "correlation_increase": 0.0,
        },
    }


@pytest.fixture
def sample_alert_data():
    """Generate sample alert data for testing."""
    return {
        "performance_alert": {
            "strategy_id": "strategy_001",
            "metric": "sharpe_ratio",
            "current_value": 0.3,
            "threshold_value": 0.5,
            "severity": AlertSeverity.WARNING,
        },
        "risk_alert": {
            "risk_type": "portfolio_var",
            "current_value": 0.025,
            "limit_value": 0.02,
            "severity": AlertSeverity.WARNING,
        },
        "strategy_alert": {
            "strategy_id": "strategy_002",
            "event": "regime_change",
            "details": "Market regime changed from trending to volatile",
            "severity": AlertSeverity.INFO,
        },
        "system_alert": {
            "component": "data_feed",
            "event": "connection_lost",
            "details": "Lost connection to market data feed",
            "severity": AlertSeverity.ERROR,
        },
        "trade_alert": {
            "symbol": "AAPL",
            "action": "buy",
            "quantity": 100,
            "price": 150.25,
            "severity": AlertSeverity.INFO,
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("PYTHONPATH", "src")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_async_context():
    """Create a mock async context for testing async functions."""

    class MockAsyncContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    return MockAsyncContext()


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


# Integration testing fixtures
@pytest.fixture
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "database_url": "sqlite:///:memory:",
        "broker_type": "mock",
        "market_data_type": "mock",
        "knowledge_base_path": "test_data/knowledge_base",
        "test_timeout": 30,
    }


# System testing fixtures
@pytest.fixture
def system_test_config():
    """Configuration for system tests."""
    return {
        "startup_timeout": 60,
        "shutdown_timeout": 30,
        "health_check_interval": 5,
        "max_retries": 3,
    }


# User acceptance testing fixtures
@pytest.fixture
def acceptance_test_scenarios():
    """Define acceptance test scenarios."""
    return {
        "bull_market": {
            "market_conditions": "trending_up",
            "volatility": "medium",
            "expected_strategies": ["trend_following", "momentum"],
            "expected_risk_level": "moderate",
        },
        "bear_market": {
            "market_conditions": "trending_down",
            "volatility": "high",
            "expected_strategies": ["hedging", "defensive"],
            "expected_risk_level": "conservative",
        },
        "sideways_market": {
            "market_conditions": "sideways",
            "volatility": "low",
            "expected_strategies": ["mean_reversion", "range_trading"],
            "expected_risk_level": "moderate",
        },
        "crisis_market": {
            "market_conditions": "crisis",
            "volatility": "extreme",
            "expected_strategies": ["hedging", "flight_to_quality"],
            "expected_risk_level": "conservative",
        },
    }


# Production readiness testing fixtures
@pytest.fixture
def production_test_config():
    """Configuration for production readiness tests."""
    return {
        "deployment_timeout": 300,
        "health_check_timeout": 60,
        "monitoring_interval": 10,
        "alert_delivery_timeout": 30,
        "rollback_timeout": 120,
    }
