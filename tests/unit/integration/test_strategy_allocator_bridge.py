"""Tests for StrategyAllocatorBridge."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from bot.integration.strategy_allocator_bridge import StrategyAllocatorBridge
from bot.portfolio.allocator import PortfolioRules
from bot.strategy.demo_ma import DemoMAStrategy


class TestStrategyAllocatorBridge:
    """Test suite for StrategyAllocatorBridge."""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        np.random.seed(42)  # For reproducible tests

        # Create trending data that should generate MA crossover signals
        close_prices = 100 + np.cumsum(np.random.randn(50) * 0.5 + 0.1)
        high_prices = close_prices + np.random.rand(50) * 2
        low_prices = close_prices - np.random.rand(50) * 2
        open_prices = close_prices + np.random.randn(50) * 0.5
        volume = np.random.randint(1000, 10000, 50)

        return pd.DataFrame(
            {
                "Open": open_prices,
                "High": high_prices,
                "Low": low_prices,
                "Close": close_prices,
                "Volume": volume,
            },
            index=dates,
        )

    @pytest.fixture
    def strategy(self):
        """Create a test strategy."""
        return DemoMAStrategy(fast=5, slow=10, atr_period=10)

    @pytest.fixture
    def portfolio_rules(self):
        """Create test portfolio rules."""
        return PortfolioRules(
            per_trade_risk_pct=0.01,  # 1% risk per trade
            max_positions=5,
            max_gross_exposure_pct=0.8,
            atr_k=2.0,
        )

    @pytest.fixture
    def bridge(self, strategy, portfolio_rules):
        """Create a configured bridge instance."""
        return StrategyAllocatorBridge(strategy, portfolio_rules)

    def test_bridge_initialization(self, strategy, portfolio_rules):
        """Test bridge initialization."""
        bridge = StrategyAllocatorBridge(strategy, portfolio_rules)

        assert bridge.strategy == strategy
        assert bridge.rules == portfolio_rules
        assert bridge.strategy.name == "demo_ma"

    def test_process_signals_with_valid_data(self, bridge, sample_ohlcv_data):
        """Test signal processing with valid market data."""
        market_data = {
            "AAPL": sample_ohlcv_data.copy(),
            "MSFT": sample_ohlcv_data.copy() * 0.8,  # Different price level
        }
        equity = 100000.0

        allocations = bridge.process_signals(market_data, equity)

        # Should return a dictionary with symbol -> position size
        assert isinstance(allocations, dict)

        # All allocation values should be non-negative integers
        for symbol, qty in allocations.items():
            assert isinstance(qty, (int, np.integer))
            assert qty >= 0

        # Should not exceed max_positions rule
        active_positions = sum(1 for qty in allocations.values() if qty > 0)
        assert active_positions <= bridge.rules.max_positions

    def test_process_signals_empty_market_data(self, bridge):
        """Test handling of empty market data."""
        allocations = bridge.process_signals({}, 100000.0)
        assert allocations == {}

    def test_process_signals_invalid_equity(self, bridge, sample_ohlcv_data):
        """Test handling of invalid equity values."""
        market_data = {"AAPL": sample_ohlcv_data}

        with pytest.raises(ValueError, match="Invalid equity value"):
            bridge.process_signals(market_data, 0.0)

        with pytest.raises(ValueError, match="Invalid equity value"):
            bridge.process_signals(market_data, -1000.0)

    def test_process_signals_with_missing_columns(self, bridge):
        """Test handling of data with missing required columns."""
        # Create data without Close column
        bad_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Volume": [1000, 1100, 1200],
            }
        )

        market_data = {"AAPL": bad_data}
        equity = 100000.0

        allocations = bridge.process_signals(market_data, equity)

        # Should handle gracefully and return empty allocations
        assert allocations == {}

    def test_get_strategy_info(self, bridge):
        """Test strategy information retrieval."""
        info = bridge.get_strategy_info()

        assert isinstance(info, dict)
        assert info["name"] == "demo_ma"
        assert info["supports_short"] == False
        assert info["strategy_type"] == "DemoMAStrategy"

    def test_get_allocation_rules_info(self, bridge):
        """Test allocation rules information retrieval."""
        info = bridge.get_allocation_rules_info()

        assert isinstance(info, dict)
        assert "per_trade_risk_pct" in info
        assert "max_positions" in info
        assert "max_gross_exposure_pct" in info
        assert "atr_k" in info
        assert "cost_bps" in info

        assert info["per_trade_risk_pct"] == 0.01
        assert info["max_positions"] == 5

    def test_validate_configuration_valid(self, bridge):
        """Test configuration validation with valid setup."""
        assert bridge.validate_configuration() == True

    def test_validate_configuration_invalid_strategy(self, portfolio_rules):
        """Test configuration validation with invalid strategy."""
        # Create a mock strategy without generate_signals method
        invalid_strategy = Mock()
        invalid_strategy.name = "invalid"
        del invalid_strategy.generate_signals

        bridge = StrategyAllocatorBridge(invalid_strategy, portfolio_rules)
        assert bridge.validate_configuration() == False

    def test_validate_configuration_invalid_rules(self, strategy):
        """Test configuration validation with invalid rules."""
        # Invalid risk percentage
        invalid_rules = PortfolioRules(per_trade_risk_pct=1.5)  # > 1.0
        bridge = StrategyAllocatorBridge(strategy, invalid_rules)
        assert bridge.validate_configuration() == False

        # Invalid max positions
        invalid_rules = PortfolioRules(max_positions=0)
        bridge = StrategyAllocatorBridge(strategy, invalid_rules)
        assert bridge.validate_configuration() == False

    @patch("bot.integration.strategy_allocator_bridge.allocate_signals")
    def test_process_signals_allocation_error_handling(
        self, mock_allocate, bridge, sample_ohlcv_data
    ):
        """Test error handling when allocation fails."""
        # Mock allocate_signals to raise an exception
        mock_allocate.side_effect = Exception("Allocation failed")

        market_data = {"AAPL": sample_ohlcv_data}
        equity = 100000.0

        allocations = bridge.process_signals(market_data, equity)

        # Should handle error gracefully and return empty dict
        assert allocations == {}

    def test_process_signals_strategy_error_handling(self, bridge):
        """Test error handling when strategy signal generation fails."""
        # Create data that might cause strategy errors
        problematic_data = pd.DataFrame(
            {
                "Close": [np.nan, np.inf, -1],  # Problematic values
                "Open": [100, 101, 102],
                "High": [101, 102, 103],
                "Low": [99, 100, 101],
                "Volume": [1000, 1100, 1200],
            }
        )

        market_data = {"PROBLEMATIC": problematic_data}
        equity = 100000.0

        # Should handle errors gracefully
        allocations = bridge.process_signals(market_data, equity)
        assert isinstance(allocations, dict)

    def test_integration_with_real_strategy_flow(self, bridge, sample_ohlcv_data):
        """Test the complete integration flow with realistic data."""
        # Create multi-symbol market data
        market_data = {
            "AAPL": sample_ohlcv_data.copy(),
            "MSFT": sample_ohlcv_data.copy() * 1.2,
            "GOOGL": sample_ohlcv_data.copy() * 0.9,
        }
        equity = 250000.0

        # Validate configuration first
        assert bridge.validate_configuration()

        # Process signals
        allocations = bridge.process_signals(market_data, equity)

        # Verify output structure
        assert isinstance(allocations, dict)
        assert all(isinstance(k, str) for k in allocations.keys())
        assert all(isinstance(v, (int, np.integer)) for v in allocations.values())

        # Check business logic constraints
        active_positions = sum(1 for qty in allocations.values() if qty > 0)
        assert active_positions <= bridge.rules.max_positions

        # Verify we can get information about the bridge
        strategy_info = bridge.get_strategy_info()
        rules_info = bridge.get_allocation_rules_info()

        assert strategy_info["name"] == "demo_ma"
        assert rules_info["max_positions"] == 5

        print(f"Integration test results:")
        print(f"  - Processed {len(market_data)} symbols")
        print(f"  - Generated {active_positions} active positions")
        print(f"  - Allocations: {allocations}")
        print(f"  - Strategy: {strategy_info}")
        print(f"  - Rules: {rules_info}")
