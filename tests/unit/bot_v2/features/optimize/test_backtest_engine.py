"""Tests for production-parity backtest engine."""

from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from bot_v2.features.brokerages.core.interfaces import MarketType, Product
from bot_v2.features.live_trade.strategies.perps_baseline.config import StrategyConfig
from bot_v2.features.live_trade.strategies.perps_baseline.strategy import BaselinePerpsStrategy
from bot_v2.features.optimize.backtest_engine import BacktestEngine, run_backtest_production
from bot_v2.features.optimize.types_v2 import BacktestConfig


@pytest.fixture
def sample_data():
    """Create sample OHLC data for testing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")

    # Create trending data with some volatility
    np.random.seed(42)
    trend = np.linspace(100, 120, len(dates))
    noise = np.random.normal(0, 2, len(dates))
    close = trend + noise

    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.random.uniform(1000, 5000, len(dates)),
        }
    )


@pytest.fixture
def strategy():
    """Create a test strategy instance."""
    config = StrategyConfig(
        short_ma_period=5,
        long_ma_period=20,
        position_fraction=0.1,
        enable_shorts=False,
    )
    return BaselinePerpsStrategy(config=config, environment="backtest")


@pytest.fixture
def product():
    """Create a test product."""
    return Product(
        symbol="BTC-USD",
        base_asset="BTC",
        quote_asset="USD",
        market_type=MarketType.SPOT,
        min_size=Decimal("0.001"),
        step_size=Decimal("0.001"),
        min_notional=Decimal("10"),
        price_increment=Decimal("0.01"),
    )


def test_backtest_engine_initialization(strategy):
    """Test that BacktestEngine initializes correctly."""
    engine = BacktestEngine(strategy=strategy)

    assert engine.strategy == strategy
    assert engine.config.initial_capital == Decimal("10000")
    assert engine.portfolio.cash == Decimal("10000")
    assert len(engine.decision_logger.decisions) == 0


def test_backtest_engine_with_custom_config(strategy):
    """Test BacktestEngine with custom configuration."""
    config = BacktestConfig(
        initial_capital=Decimal("50000"),
        commission_rate=Decimal("0.002"),
        slippage_rate=Decimal("0.001"),
        enable_decision_logging=False,
    )
    engine = BacktestEngine(strategy=strategy, config=config)

    assert engine.portfolio.cash == Decimal("50000")
    assert engine.portfolio.commission_rate == Decimal("0.002")
    assert engine.decision_logger.enabled is False


def test_run_backtest_basic(strategy, sample_data, product):
    """Test basic backtest execution."""
    engine = BacktestEngine(strategy=strategy)
    result = engine.run(data=sample_data, symbol="BTC-USD", product=product)

    # Verify result structure
    assert result.run_id.startswith("bt_")
    assert result.symbol == "BTC-USD"
    assert result.strategy_name == "BaselinePerpsStrategy"
    assert len(result.decisions) > 0
    assert result.metrics is not None

    # Verify metrics are calculated
    assert isinstance(result.metrics.total_return, (float, int))
    assert isinstance(result.metrics.sharpe_ratio, (float, int))
    assert isinstance(result.metrics.max_drawdown, (float, int))


def test_backtest_calls_production_strategy(strategy, sample_data, product):
    """Test that backtest actually calls the production strategy.decide() method."""
    # Spy on strategy.decide() calls
    decide_calls = []
    original_decide = strategy.decide

    def decide_spy(**kwargs):
        decide_calls.append(kwargs)
        return original_decide(**kwargs)

    strategy.decide = decide_spy

    engine = BacktestEngine(strategy=strategy)
    result = engine.run(data=sample_data, symbol="BTC-USD", product=product)

    # Verify strategy.decide() was called multiple times
    assert len(decide_calls) > 0

    # Verify calls had correct structure
    first_call = decide_calls[0]
    assert "symbol" in first_call
    assert "current_mark" in first_call
    assert "position_state" in first_call
    assert "recent_marks" in first_call
    assert "equity" in first_call
    assert "product" in first_call

    # Verify we logged decisions
    assert len(result.decisions) == len(decide_calls)


def test_backtest_records_equity_curve(strategy, sample_data, product):
    """Test that equity curve is recorded properly."""
    engine = BacktestEngine(strategy=strategy)
    result = engine.run(data=sample_data, symbol="BTC-USD", product=product)

    # Verify equity curve exists
    assert len(result.equity_curve) > 0

    # Verify equity curve structure
    for timestamp, equity in result.equity_curve:
        assert isinstance(timestamp, datetime)
        assert isinstance(equity, Decimal)

    # Verify equity starts at initial capital
    initial_equity = result.equity_curve[0][1]
    assert initial_equity == Decimal("10000")


def test_backtest_with_no_data_raises_error(strategy):
    """Test that backtest with empty data raises appropriate error."""
    engine = BacktestEngine(strategy=strategy)
    empty_data = pd.DataFrame({"close": []})

    with pytest.raises(ValueError, match="Data is empty"):
        engine.run(data=empty_data, symbol="BTC-USD")


def test_backtest_with_insufficient_data(strategy, product):
    """Test backtest with data shorter than MA period."""
    # Create data with only 10 bars (less than long_ma_period of 20)
    dates = pd.date_range("2024-01-01", periods=10, freq="1h")
    data = pd.DataFrame(
        {"timestamp": dates, "close": np.linspace(100, 105, 10)},
    )

    engine = BacktestEngine(strategy=strategy)
    result = engine.run(data=data, symbol="BTC-USD", product=product)

    # Should complete but have no trades
    assert len(result.decisions) == 0


def test_backtest_decision_logging(strategy, sample_data, product):
    """Test that decisions are logged with full context."""
    config = BacktestConfig(enable_decision_logging=True)
    engine = BacktestEngine(strategy=strategy, config=config)
    result = engine.run(data=sample_data, symbol="BTC-USD", product=product)

    # Verify decisions were logged
    assert len(result.decisions) > 0

    # Verify decision structure
    decision = result.decisions[0]
    assert decision.context.symbol == "BTC-USD"
    assert isinstance(decision.context.current_mark, Decimal)
    assert isinstance(decision.context.equity, Decimal)
    assert decision.decision is not None
    assert decision.execution is not None


def test_backtest_portfolio_state_tracking(strategy, sample_data, product):
    """Test that portfolio state is tracked correctly through backtest."""
    engine = BacktestEngine(strategy=strategy)
    result = engine.run(data=sample_data, symbol="BTC-USD", product=product)

    # Get portfolio stats
    stats = engine.portfolio.get_stats()

    assert stats["initial_capital"] == Decimal("10000")
    assert stats["trade_count"] >= 0
    assert stats["total_commission"] >= Decimal("0")


def test_run_backtest_production_convenience_function(strategy, sample_data, product):
    """Test the convenience function run_backtest_production()."""
    result = run_backtest_production(
        strategy=strategy, data=sample_data, symbol="BTC-USD", product=product
    )

    assert result.symbol == "BTC-USD"
    assert result.metrics is not None
    assert len(result.decisions) > 0


def test_backtest_default_product_creation(strategy, sample_data):
    """Test that backtest creates a default product when none provided."""
    engine = BacktestEngine(strategy=strategy)
    result = engine.run(data=sample_data, symbol="BTC-USD")

    # Should succeed with default product
    assert result.symbol == "BTC-USD"
    assert result.metrics is not None


def test_backtest_result_summary(strategy, sample_data, product):
    """Test that backtest result generates readable summary."""
    result = run_backtest_production(
        strategy=strategy, data=sample_data, symbol="BTC-USD", product=product
    )

    summary = result.summary()

    # Verify summary contains key information
    assert "Production-Parity Backtest" in summary
    assert "BTC-USD" in summary
    assert "Total Return" in summary
    assert "Sharpe Ratio" in summary
    assert "Max Drawdown" in summary


def test_backtest_to_dict_serialization(strategy, sample_data, product):
    """Test that backtest result can be serialized to dict."""
    result = run_backtest_production(
        strategy=strategy, data=sample_data, symbol="BTC-USD", product=product
    )

    data = result.to_dict()

    # Verify serialization
    assert isinstance(data, dict)
    assert data["symbol"] == "BTC-USD"
    assert "decisions" in data
    assert "metrics" in data
    assert "equity_curve" in data


def test_backtest_handles_timezone_aware_timestamps(strategy, product):
    """Test that backtest handles timezone-aware timestamps correctly."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h", tz="UTC")
    data = pd.DataFrame({"timestamp": dates, "close": np.linspace(100, 120, 100)})

    result = run_backtest_production(
        strategy=strategy, data=data, symbol="BTC-USD", product=product
    )

    assert result.start_time.tzinfo is not None


def test_backtest_strategy_state_reset(strategy, sample_data, product):
    """Test that strategy state is reset before backtest."""
    # Run first backtest
    result1 = run_backtest_production(
        strategy=strategy, data=sample_data, symbol="BTC-USD", product=product
    )

    # Run second backtest with same strategy instance
    result2 = run_backtest_production(
        strategy=strategy, data=sample_data, symbol="BTC-USD", product=product
    )

    # Results should be identical (state was reset)
    assert len(result1.decisions) == len(result2.decisions)
