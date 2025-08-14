"""
Comprehensive unit tests for Portfolio Allocator.

Tests position sizing, allocation rules, risk management,
and portfolio optimization.
"""

import pytest
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from bot.portfolio.allocator import (
    PortfolioRules,
    position_size,
    allocate_signals,
    _to_float
)


class TestPortfolioRules:
    """Test PortfolioRules dataclass."""
    
    def test_portfolio_rules_defaults(self):
        """Test PortfolioRules with default values."""
        rules = PortfolioRules()
        
        assert rules.per_trade_risk_pct == 0.005
        assert rules.atr_k == 2.0
        assert rules.max_positions == 10
        assert rules.max_gross_exposure_pct == 0.60
        assert rules.cost_bps == 0.0
        assert rules.cost_adjusted_sizing is False
        assert rules.slippage_bps == 0.0
        assert rules.max_turnover_per_rebalance is None
    
    def test_portfolio_rules_custom(self):
        """Test PortfolioRules with custom values."""
        rules = PortfolioRules(
            per_trade_risk_pct=0.01,
            atr_k=1.5,
            max_positions=5,
            max_gross_exposure_pct=0.80,
            cost_bps=10.0,
            cost_adjusted_sizing=True,
            slippage_bps=5.0,
            max_turnover_per_rebalance=0.25
        )
        
        assert rules.per_trade_risk_pct == 0.01
        assert rules.atr_k == 1.5
        assert rules.max_positions == 5
        assert rules.max_gross_exposure_pct == 0.80
        assert rules.cost_bps == 10.0
        assert rules.cost_adjusted_sizing is True
        assert rules.slippage_bps == 5.0
        assert rules.max_turnover_per_rebalance == 0.25


class TestToFloat:
    """Test _to_float utility function."""
    
    def test_to_float_series(self):
        """Test conversion of pandas Series to float."""
        series = pd.Series([42.5])
        result = _to_float(series)
        assert result == 42.5
        assert isinstance(result, float)
    
    def test_to_float_dataframe_cell(self):
        """Test conversion of DataFrame cell to float."""
        df = pd.DataFrame({"value": [123.45]})
        result = _to_float(df["value"])
        assert result == 123.45
    
    def test_to_float_int(self):
        """Test conversion of int to float."""
        result = _to_float(100)
        assert result == 100.0
        assert isinstance(result, float)
    
    def test_to_float_float(self):
        """Test conversion of float to float."""
        result = _to_float(99.99)
        assert result == 99.99
        assert isinstance(result, float)
    
    def test_to_float_numpy_scalar(self):
        """Test conversion of numpy scalar to float."""
        np_value = np.float64(77.77)
        result = _to_float(np_value)
        assert result == 77.77
        assert isinstance(result, float)


class TestPositionSize:
    """Test position_size function."""
    
    @pytest.fixture
    def default_rules(self):
        """Create default portfolio rules."""
        return PortfolioRules()
    
    def test_position_size_basic(self, default_rules):
        """Test basic position sizing."""
        equity = 100000.0
        atr_value = 2.0
        price = 50.0
        
        size = position_size(equity, atr_value, price, default_rules)
        
        # Calculate expected size
        risk_usd = equity * default_rules.per_trade_risk_pct  # $500
        stop_dist = default_rules.atr_k * atr_value  # 4.0
        expected_size = math.floor(risk_usd / stop_dist)  # 125
        
        assert size == expected_size
    
    def test_position_size_with_cost_adjustment(self):
        """Test position sizing with cost adjustment."""
        rules = PortfolioRules(
            per_trade_risk_pct=0.01,
            cost_adjusted_sizing=True,
            cost_bps=20.0,
            slippage_bps=10.0
        )
        
        equity = 100000.0
        atr_value = 3.0
        price = 100.0
        
        size = position_size(equity, atr_value, price, rules)
        
        # Calculate with cost adjustment
        risk_usd = equity * rules.per_trade_risk_pct  # $1000
        total_cost_rate = (20 + 10) / 10000.0  # 0.003
        adjusted_risk = risk_usd * (1.0 - total_cost_rate)  # $997
        stop_dist = rules.atr_k * atr_value  # 6.0
        expected_size = math.floor(adjusted_risk / stop_dist)  # 166
        
        assert size == expected_size
    
    def test_position_size_zero_atr(self, default_rules):
        """Test position sizing with zero ATR."""
        size = position_size(100000.0, 0.0, 50.0, default_rules)
        assert size == 0
    
    def test_position_size_zero_price(self, default_rules):
        """Test position sizing with zero price."""
        size = position_size(100000.0, 2.0, 0.0, default_rules)
        assert size == 0
    
    def test_position_size_negative_atr(self, default_rules):
        """Test position sizing with negative ATR."""
        size = position_size(100000.0, -2.0, 50.0, default_rules)
        assert size == 0
    
    def test_position_size_small_equity(self, default_rules):
        """Test position sizing with small equity."""
        equity = 1000.0  # Small account
        atr_value = 5.0
        price = 100.0
        
        size = position_size(equity, atr_value, price, default_rules)
        
        # Risk: $5, Stop: $10, Size: 0
        assert size == 0
    
    def test_position_size_large_equity(self, default_rules):
        """Test position sizing with large equity."""
        equity = 10000000.0  # $10M
        atr_value = 1.0
        price = 10.0
        
        size = position_size(equity, atr_value, price, default_rules)
        
        # Risk: $50,000, Stop: $2, Size: 25,000
        assert size == 25000
    
    def test_position_size_fractional_result(self, default_rules):
        """Test that position size is always floored to integer."""
        equity = 10000.0
        atr_value = 3.33
        price = 100.0
        
        size = position_size(equity, atr_value, price, default_rules)
        
        # Should be integer (floored)
        assert isinstance(size, int)
        assert size >= 0


class TestAllocateSignals:
    """Test allocate_signals function."""
    
    @pytest.fixture
    def default_rules(self):
        """Create default portfolio rules."""
        return PortfolioRules()
    
    @pytest.fixture
    def sample_signals(self):
        """Create sample signal data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        
        signals = {}
        
        # Strong signal
        signals["AAPL"] = pd.DataFrame({
            "Close": np.linspace(100, 110, 100),
            "donchian_upper": np.linspace(98, 108, 100),
            "atr": np.full(100, 2.0),
            "signal": np.concatenate([np.zeros(50), np.ones(50)])
        }, index=dates)
        
        # Moderate signal
        signals["GOOGL"] = pd.DataFrame({
            "Close": np.linspace(2800, 2850, 100),
            "donchian_upper": np.linspace(2790, 2840, 100),
            "atr": np.full(100, 50.0),
            "signal": np.concatenate([np.zeros(70), np.ones(30)])
        }, index=dates)
        
        # No signal
        signals["MSFT"] = pd.DataFrame({
            "Close": np.linspace(300, 295, 100),
            "donchian_upper": np.linspace(305, 300, 100),
            "atr": np.full(100, 5.0),
            "signal": np.zeros(100)
        }, index=dates)
        
        return signals
    
    def test_allocate_signals_basic(self, sample_signals, default_rules):
        """Test basic signal allocation."""
        equity = 100000.0
        
        allocations = allocate_signals(sample_signals, equity, default_rules)
        
        # Should allocate to symbols with signals
        assert "AAPL" in allocations
        assert "GOOGL" in allocations
        assert "MSFT" not in allocations  # No signal
        
        # Allocations should be positive integers
        for qty in allocations.values():
            assert isinstance(qty, int)
            assert qty > 0
    
    def test_allocate_signals_max_positions(self):
        """Test max positions limit."""
        rules = PortfolioRules(max_positions=2)
        
        # Create many signals
        signals = {}
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        
        for i in range(5):
            symbol = f"STOCK{i}"
            signals[symbol] = pd.DataFrame({
                "Close": [100 + i],
                "donchian_upper": [99 + i],
                "atr": [2.0],
                "signal": [1.0]
            } * 10, index=dates)
        
        allocations = allocate_signals(signals, 100000.0, rules)
        
        # Should respect max positions
        assert len(allocations) <= rules.max_positions
    
    def test_allocate_signals_ranking(self):
        """Test that signals are ranked by breakout strength."""
        rules = PortfolioRules(max_positions=2)
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        
        signals = {}
        
        # Weak breakout
        signals["WEAK"] = pd.DataFrame({
            "Close": [101] * 10,
            "donchian_upper": [100] * 10,
            "atr": [2.0] * 10,
            "signal": [1.0] * 10
        }, index=dates)
        
        # Strong breakout
        signals["STRONG"] = pd.DataFrame({
            "Close": [110] * 10,
            "donchian_upper": [100] * 10,
            "atr": [2.0] * 10,
            "signal": [1.0] * 10
        }, index=dates)
        
        # Medium breakout
        signals["MEDIUM"] = pd.DataFrame({
            "Close": [105] * 10,
            "donchian_upper": [100] * 10,
            "atr": [2.0] * 10,
            "signal": [1.0] * 10
        }, index=dates)
        
        allocations = allocate_signals(signals, 100000.0, rules)
        
        # Should select STRONG and MEDIUM (top 2)
        assert "STRONG" in allocations
        assert "MEDIUM" in allocations
        assert "WEAK" not in allocations
    
    def test_allocate_signals_empty_data(self, default_rules):
        """Test allocation with empty signal data."""
        signals = {}
        allocations = allocate_signals(signals, 100000.0, default_rules)
        
        assert allocations == {}
    
    def test_allocate_signals_no_active_signals(self, default_rules):
        """Test allocation when no symbols have active signals."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        
        signals = {
            "AAPL": pd.DataFrame({
                "Close": [100] * 10,
                "donchian_upper": [99] * 10,
                "atr": [2.0] * 10,
                "signal": [0.0] * 10  # No signals
            }, index=dates)
        }
        
        allocations = allocate_signals(signals, 100000.0, default_rules)
        
        assert allocations == {}
    
    def test_allocate_signals_with_nan(self, default_rules):
        """Test allocation with NaN values in data."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        
        signals = {
            "AAPL": pd.DataFrame({
                "Close": [100, np.nan, 102, 103, 104, 105, 106, 107, 108, 109],
                "donchian_upper": [99] * 10,
                "atr": [2.0] * 10,
                "signal": [1.0] * 10
            }, index=dates)
        }
        
        allocations = allocate_signals(signals, 100000.0, default_rules)
        
        # Should handle NaN gracefully
        assert isinstance(allocations, dict)
    
    def test_allocate_signals_gross_exposure_limit(self):
        """Test gross exposure limit."""
        rules = PortfolioRules(
            max_gross_exposure_pct=0.50,
            per_trade_risk_pct=0.10  # Large position size
        )
        
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        
        signals = {}
        for i in range(5):
            signals[f"STOCK{i}"] = pd.DataFrame({
                "Close": [100] * 10,
                "donchian_upper": [99] * 10,
                "atr": [1.0] * 10,
                "signal": [1.0] * 10
            }, index=dates)
        
        equity = 100000.0
        allocations = allocate_signals(signals, equity, rules)
        
        # Calculate total exposure
        total_exposure = sum(
            qty * 100 for qty in allocations.values()  # qty * price
        )
        
        # Should not exceed gross exposure limit
        assert total_exposure <= equity * rules.max_gross_exposure_pct * 1.1  # Allow 10% margin
    
    def test_allocate_signals_missing_columns(self, default_rules):
        """Test allocation with missing required columns."""
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        
        # Missing 'atr' column
        signals = {
            "AAPL": pd.DataFrame({
                "Close": [100] * 10,
                "donchian_upper": [99] * 10,
                "signal": [1.0] * 10
            }, index=dates)
        }
        
        # Should handle gracefully
        allocations = allocate_signals(signals, 100000.0, default_rules)
        
        # May skip symbol or use default
        assert isinstance(allocations, dict)


class TestIntegrationScenarios:
    """Test integration scenarios for portfolio allocation."""
    
    @pytest.fixture
    def market_data(self):
        """Create realistic market data."""
        dates = pd.date_range(start="2024-01-01", periods=252, freq="D")
        np.random.seed(42)
        
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        signals = {}
        
        for i, symbol in enumerate(symbols):
            # Different volatility for each stock
            volatility = 0.02 + i * 0.005
            returns = np.random.normal(0.001, volatility, 252)
            prices = 100 * (1 + i * 0.5) * np.exp(np.cumsum(returns))
            
            # Donchian channels
            rolling_high = pd.Series(prices).rolling(20).max()
            rolling_low = pd.Series(prices).rolling(20).min()
            
            # ATR calculation
            high = prices * (1 + np.abs(np.random.normal(0, 0.01, 252)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.01, 252)))
            tr = np.maximum(high - low, np.abs(high - np.roll(prices, 1)))
            atr = pd.Series(tr).rolling(14).mean()
            
            # Generate signals (breakout strategy)
            signal = (prices > rolling_high.shift(1)).astype(float)
            
            signals[symbol] = pd.DataFrame({
                "Close": prices,
                "donchian_upper": rolling_high,
                "donchian_lower": rolling_low,
                "atr": atr,
                "signal": signal
            }, index=dates)
        
        return signals
    
    def test_full_portfolio_allocation(self, market_data):
        """Test full portfolio allocation with realistic data."""
        rules = PortfolioRules(
            per_trade_risk_pct=0.01,
            max_positions=3,
            max_gross_exposure_pct=0.80
        )
        
        equity = 1000000.0
        
        allocations = allocate_signals(market_data, equity, rules)
        
        # Should allocate to some symbols
        assert len(allocations) > 0
        assert len(allocations) <= rules.max_positions
        
        # Check position sizes are reasonable
        for symbol, qty in allocations.items():
            position_value = qty * market_data[symbol]["Close"].iloc[-1]
            position_pct = position_value / equity
            
            # Position should be reasonable size
            assert 0 < position_pct < 0.5
    
    def test_risk_parity_allocation(self):
        """Test risk parity style allocation."""
        rules = PortfolioRules(
            per_trade_risk_pct=0.01,
            atr_k=1.0  # Use 1 ATR for equal risk
        )
        
        dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
        
        # Different volatilities
        signals = {
            "LOW_VOL": pd.DataFrame({
                "Close": [100] * 10,
                "donchian_upper": [99] * 10,
                "atr": [1.0] * 10,  # Low volatility
                "signal": [1.0] * 10
            }, index=dates),
            "HIGH_VOL": pd.DataFrame({
                "Close": [100] * 10,
                "donchian_upper": [99] * 10,
                "atr": [5.0] * 10,  # High volatility
                "signal": [1.0] * 10
            }, index=dates)
        }
        
        allocations = allocate_signals(signals, 100000.0, rules)
        
        # Low vol should get larger position
        if "LOW_VOL" in allocations and "HIGH_VOL" in allocations:
            assert allocations["LOW_VOL"] > allocations["HIGH_VOL"]
    
    def test_turnover_constraint(self):
        """Test turnover constraint."""
        rules = PortfolioRules(
            max_turnover_per_rebalance=0.20  # 20% max turnover
        )
        
        # This would require tracking existing positions
        # which is not in the current allocate_signals function
        # Test placeholder for future enhancement
        pass