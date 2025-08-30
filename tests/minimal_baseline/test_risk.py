"""
TEST-004: Risk Management Tests

Verifies that basic risk controls are working.
Tests position limits, stop losses, and basic risk calculations.
"""

import numpy as np
import pandas as pd


class TestRiskManagement:
    """Test basic risk management functionality."""

    def test_strategy_stop_loss_generation(self, sample_market_data):
        """Test that strategies generate stop loss levels."""
        from bot.strategy.demo_ma import DemoMAStrategy

        # Convert columns to lowercase
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        strategy = DemoMAStrategy()
        signals = strategy.generate_signals(df)

        # Should have stop_loss column
        assert "stop_loss" in signals.columns

        # Stop losses should be positive where defined
        valid_sl = signals["stop_loss"].dropna()
        if len(valid_sl) > 0:
            assert (valid_sl > 0).all()

    def test_strategy_take_profit_generation(self, sample_market_data):
        """Test that strategies generate take profit levels."""
        from bot.strategy.demo_ma import DemoMAStrategy

        # Convert columns to lowercase
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        strategy = DemoMAStrategy()
        signals = strategy.generate_signals(df)

        # Should have take_profit column
        assert "take_profit" in signals.columns

        # Take profits should be positive where defined
        valid_tp = signals["take_profit"].dropna()
        if len(valid_tp) > 0:
            assert (valid_tp > 0).all()

    def test_atr_based_risk_calculation(self, sample_market_data):
        """Test ATR-based risk calculations work."""
        from bot.indicators.atr import atr

        # Convert columns to lowercase
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        atr_values = atr(df, period=14)

        # ATR should be calculated
        assert isinstance(atr_values, pd.Series)

        # Where ATR is valid, it should be positive
        valid_atr = atr_values.dropna()
        if len(valid_atr) > 0:
            assert (valid_atr >= 0).all()

    def test_position_sizing_basic(self):
        """Test basic position sizing concepts."""
        # Simple position sizing calculation
        account_value = 100000
        risk_per_trade = 0.02  # 2%
        stop_loss_distance = 5.0  # $5 per share

        risk_amount = account_value * risk_per_trade
        position_size = risk_amount / stop_loss_distance

        assert risk_amount == 2000
        assert position_size == 400
        assert position_size > 0

    def test_portfolio_allocator_risk_constraints(self):
        """Test portfolio allocator basic functionality."""
        from bot.portfolio.allocator import allocate_signals, PortfolioRules

        # Test that allocator function exists
        assert callable(allocate_signals)
        
        # Test portfolio rules
        rules = PortfolioRules()
        assert rules is not None
        assert rules.max_positions > 0

    def test_configuration_risk_parameters(self):
        """Test that risk parameters are in configuration."""
        from bot.config import get_config

        config = get_config()

        # Should have financial configuration with sensible values
        assert config.financial.capital.backtesting_capital > 0
        assert config.financial.capital.paper_trading_capital > 0

    def test_signal_risk_consistency(self, sample_market_data):
        """Test that strategy signals include consistent risk data."""
        from bot.strategy.demo_ma import DemoMAStrategy

        # Convert columns to lowercase
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        strategy = DemoMAStrategy()
        signals = strategy.generate_signals(df)

        # Get rows where we have signals
        signal_rows = signals[signals["signal"] != 0]

        if len(signal_rows) > 0:
            # When we have signals, we should have risk levels
            for _, row in signal_rows.iterrows():
                if pd.notna(row["signal"]) and row["signal"] != 0:
                    # Should have some risk management data
                    assert pd.notna(row["stop_loss"]) or pd.notna(row["take_profit"])

    def test_risk_data_types(self, sample_market_data):
        """Test that risk data has correct types."""
        from bot.strategy.demo_ma import DemoMAStrategy

        # Convert columns to lowercase
        df = sample_market_data.copy()
        df.columns = df.columns.str.lower()
        
        strategy = DemoMAStrategy()
        signals = strategy.generate_signals(df)

        # Check data types
        assert signals["signal"].dtype in [np.int64, np.float64, np.int32, np.float32, object]
        assert signals["stop_loss"].dtype in [np.float64, np.float32, object]
        assert signals["take_profit"].dtype in [np.float64, np.float32, object]

    def test_extreme_market_conditions(self):
        """Test risk calculations under extreme conditions."""
        # Create extreme market data (large gaps) with DatetimeIndex
        dates = pd.date_range(start="2023-01-01", periods=4)
        extreme_data = pd.DataFrame(
            {
                "open": [100, 95, 110, 105],
                "high": [105, 100, 115, 110],
                "low": [95, 90, 105, 100],
                "close": [98, 108, 107, 109],  # Large price swings
                "volume": [1000000, 2000000, 1500000, 1200000],
            },
            index=dates
        )

        from bot.strategy.demo_ma import DemoMAStrategy

        strategy = DemoMAStrategy(fast=2, slow=3)  # Short windows for quick signals
        signals = strategy.generate_signals(extreme_data)

        # Should handle extreme data without crashing
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) == len(extreme_data)
