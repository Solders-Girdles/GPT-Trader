"""
TEST-004: Core Import Tests

Verifies that all critical modules can be imported without errors.
These are the foundation - if these fail, nothing else will work.
"""

import pytest


@pytest.mark.critical
@pytest.mark.baseline
class TestCoreImports:
    """Test that critical modules can be imported."""

    @pytest.mark.critical
    def test_config_import(self):
        """Test that configuration module loads."""
        from bot.config import TradingConfig, get_config

        config = get_config()
        assert isinstance(config, TradingConfig)

    @pytest.mark.critical
    def test_strategy_base_import(self):
        """Test that strategy base class imports."""
        from bot.strategy.base import Strategy

        assert Strategy is not None

    @pytest.mark.critical
    def test_demo_ma_strategy_import(self):
        """Test that demo MA strategy imports and instantiates."""
        from bot.strategy.demo_ma import DemoMAStrategy

        strategy = DemoMAStrategy(fast=10, slow=20)
        assert strategy.name == "demo_ma"
        assert strategy.fast == 10
        assert strategy.slow == 20

    def test_trend_breakout_strategy_import(self):
        """Test that trend breakout strategy imports."""
        from bot.strategy.trend_breakout import TrendBreakoutStrategy

        strategy = TrendBreakoutStrategy()
        assert strategy.name == "trend_breakout"

    @pytest.mark.critical
    def test_yfinance_source_import(self):
        """Test that yfinance data source imports."""
        from bot.dataflow.sources.yfinance_source import YFinanceSource

        assert YFinanceSource is not None

    @pytest.mark.critical
    def test_atr_indicator_import(self):
        """Test that ATR indicator imports."""
        from bot.indicators.atr import atr

        assert callable(atr)

    @pytest.mark.critical
    def test_backtest_engine_import(self):
        """Test that backtest engine imports (the one that actually exists)."""
        from bot.backtest.engine_portfolio import BacktestEngine

        assert BacktestEngine is not None

    @pytest.mark.critical
    def test_logging_import(self):
        """Test that logging system imports."""
        from bot.logging import get_logger

        logger = get_logger("test")
        assert logger is not None

    def test_portfolio_allocator_import(self):
        """Test that portfolio allocator imports."""
        from bot.portfolio.allocator import allocate_signals, PortfolioRules

        assert callable(allocate_signals)
        assert PortfolioRules is not None
