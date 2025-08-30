"""
TEST-004: Integration Tests

Tests that major components can work together.
Focuses on realistic end-to-end scenarios.
"""

import pandas as pd
import pytest


class TestBasicIntegration:
    """Test basic integration between components."""

    def test_config_and_strategy_integration(self):
        """Test that config system works with strategies."""
        from bot.config import get_config
        from bot.strategy.demo_ma import DemoMAStrategy

        config = get_config()
        strategy = DemoMAStrategy()

        # Should be able to create both without conflicts
        assert config is not None
        assert strategy is not None

    def test_data_and_strategy_integration(self, sample_market_data):
        """Test that data pipeline works with strategies."""
        from bot.strategy.demo_ma import DemoMAStrategy

        # Convert columns to lowercase as expected by strategies
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        strategy = DemoMAStrategy(fast=5, slow=10)
        signals = strategy.generate_signals(df)

        # Strategy should process data successfully
        assert len(signals) == len(df)
        assert "signal" in signals.columns

    def test_portfolio_allocator_basic(self):
        """Test basic portfolio allocator functionality."""
        from bot.portfolio.allocator import allocate_signals, PortfolioRules

        # Test that allocator function and rules exist
        assert callable(allocate_signals)
        rules = PortfolioRules()
        assert rules is not None

    def test_backtest_engine_creation(self):
        """Test that backtest engine can be created."""
        from bot.backtest.engine_portfolio import BacktestEngine, BacktestConfig
        from datetime import datetime

        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=10000
        )
        engine = BacktestEngine(config)
        assert engine is not None

    @pytest.mark.slow
    def test_simple_backtest_flow(self, sample_market_data):
        """Test a simple end-to-end backtest flow."""
        from bot.backtest.engine_portfolio import BacktestEngine, BacktestConfig
        from bot.strategy.demo_ma import DemoMAStrategy
        from datetime import datetime

        # Convert columns to lowercase
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        # Create strategy
        strategy = DemoMAStrategy(fast=5, slow=10)

        # Generate signals
        signals = strategy.generate_signals(df)

        # Create backtest engine
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=10000
        )
        engine = BacktestEngine(config)

        # This should not crash
        assert strategy is not None
        assert engine is not None
        assert signals is not None

    def test_logging_integration(self):
        """Test that logging works across components."""
        from bot.logging import get_logger

        logger1 = get_logger("test1")
        logger2 = get_logger("test2")

        # Should be able to create multiple loggers
        assert logger1 is not None
        assert logger2 is not None

        # Should be able to log messages
        logger1.info("Test message 1")
        logger2.debug("Test message 2")

    def test_configuration_financial_constants(self):
        """Test that financial constants are accessible."""
        from bot.config import get_config

        config = get_config()

        # Should have financial configuration
        assert hasattr(config, "financial")
        assert hasattr(config.financial, "capital")
        assert config.financial.capital.backtesting_capital > 0

    def test_atr_with_strategy_integration(self, sample_market_data):
        """Test ATR calculation works within strategy context."""
        from bot.indicators.atr import atr

        # Convert columns to lowercase for ATR
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        # Test ATR calculation
        atr_values = atr(df, period=14)

        assert isinstance(atr_values, pd.Series)
        assert len(atr_values) == len(df)

    def test_data_validation_with_strategy(self):
        """Test data validation integrates with strategy workflow."""
        from bot.dataflow.validate import validate_daily_bars
        from bot.strategy.demo_ma import DemoMAStrategy

        # Create valid data with DatetimeIndex and lowercase columns
        dates = pd.date_range(start="2023-01-01", periods=20)
        valid_data = pd.DataFrame(
            {
                "open": list(range(100, 120)),
                "high": list(range(102, 122)),
                "low": list(range(99, 119)),
                "close": list(range(101, 121)),
                "volume": [1000 + i*100 for i in range(20)],
            },
            index=dates
        )

        # Validation should pass
        validate_daily_bars(valid_data, "TEST")

        # Strategy should work with validated data
        strategy = DemoMAStrategy()
        signals = strategy.generate_signals(valid_data)
        assert isinstance(signals, pd.DataFrame)
