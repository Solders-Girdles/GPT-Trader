"""Basic integration test to verify fixtures work."""

import pandas as pd


def test_sample_market_data_fixture(sample_market_data):
    """Test that sample_market_data fixture provides valid data."""
    assert isinstance(sample_market_data, dict)
    assert len(sample_market_data) == 3  # AAPL, GOOGL, MSFT

    for symbol, data in sample_market_data.items():
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100  # 100 days
        assert all(col in data.columns for col in ["Open", "High", "Low", "Close", "Volume"])
        assert data.index.name is None or isinstance(data.index, pd.DatetimeIndex)

        # Verify OHLC relationships
        assert (data["high"] >= data["close"]).all()
        assert (data["low"] <= data["close"]).all()
        assert (data["high"] >= data["low"]).all()


def test_strategy_fixture(test_strategy):
    """Test that test_strategy fixture provides valid strategy."""
    from bot.strategy.demo_ma import DemoMAStrategy

    assert isinstance(test_strategy, DemoMAStrategy)
    assert hasattr(test_strategy, "fast")
    assert hasattr(test_strategy, "slow")
    assert test_strategy.fast == 10
    assert test_strategy.slow == 20


def test_portfolio_rules_fixture(portfolio_rules):
    """Test that portfolio_rules fixture provides valid rules."""
    from bot.portfolio.allocator import PortfolioRules

    assert isinstance(portfolio_rules, PortfolioRules)
    assert portfolio_rules.per_trade_risk_pct == 0.01
    assert portfolio_rules.max_positions == 5
    assert portfolio_rules.atr_k == 2.0


def test_strategy_with_sample_data(test_strategy, sample_market_data):
    """Test that strategy can process sample data."""
    # Get data for one symbol
    aapl_data = sample_market_data["AAPL"]

    # Generate signals
    signals = test_strategy.generate_signals(aapl_data)

    # Verify signals structure
    assert isinstance(signals, pd.DataFrame)
    assert len(signals) == len(aapl_data)
    assert "signal" in signals.columns
    assert "sma_fast" in signals.columns
    assert "sma_slow" in signals.columns
    assert "atr" in signals.columns


def test_backtest_data_fixture(sample_backtest_data):
    """Test that sample_backtest_data fixture is valid."""
    from bot.backtest.engine_portfolio import BacktestData

    assert isinstance(sample_backtest_data, BacktestData)
    assert len(sample_backtest_data.symbols) == 3
    assert set(sample_backtest_data.symbols) == {"AAPL", "GOOGL", "MSFT"}
    assert len(sample_backtest_data.data_map) == 3
    assert isinstance(sample_backtest_data.dates_idx, pd.DatetimeIndex)
    assert len(sample_backtest_data.dates_idx) == 100
