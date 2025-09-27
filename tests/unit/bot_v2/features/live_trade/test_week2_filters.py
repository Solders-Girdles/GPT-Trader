"""
Unit tests for Week 2 strategy filters.
"""

import pytest
from decimal import Decimal

from bot_v2.features.live_trade.strategies.week2_filters import (
    LiquidityGates, LiquidityFilter, RSIConfirmation, 
    Week2StrategyFilter, MarketSnapshot, create_test_snapshot
)


class TestLiquidityFilter:
    """Test liquidity gating logic."""
    
    def setup_method(self):
        """Set up test filter."""
        gates = LiquidityGates(
            spread_bps_max=20.0,
            l1_min=Decimal("50"),
            l10_min=Decimal("500"),
            vol_1m_min=75.0
        )
        self.filter = LiquidityFilter(gates)
    
    def test_spread_rejection(self):
        """Test entry rejection when spread exceeds threshold."""
        snapshot = create_test_snapshot(spread_bps=25.0)  # Above 20 limit
        
        should_reject, reason = self.filter.should_reject_entry(snapshot)
        
        assert should_reject == True
        assert "Spread 25.0bps > 20.0bps" in reason
    
    def test_l1_depth_rejection(self):
        """Test entry rejection when L1 depth too low."""
        snapshot = create_test_snapshot(depth_l1=25.0)  # Below 50 min
        
        should_reject, reason = self.filter.should_reject_entry(snapshot)
        
        assert should_reject == True
        assert "L1 depth" in reason and "< 50" in reason
    
    def test_l10_depth_rejection(self):
        """Test entry rejection when L10 depth too low."""
        snapshot = create_test_snapshot(depth_l10=300.0)  # Below 500 min
        
        should_reject, reason = self.filter.should_reject_entry(snapshot)
        
        assert should_reject == True
        assert "L10 depth" in reason and "< 500" in reason
    
    def test_volume_rejection(self):
        """Test entry rejection when volume too low."""
        snapshot = create_test_snapshot(vol_1m=50.0)  # Below 75 min
        
        should_reject, reason = self.filter.should_reject_entry(snapshot)
        
        assert should_reject == True
        assert "1m volume 50.0 < 75.0" in reason
        
    def test_all_gates_pass(self):
        """Test entry passes when all conditions met."""
        snapshot = create_test_snapshot(
            spread_bps=15.0,    # < 20
            depth_l1=100.0,     # > 50
            depth_l10=1000.0,   # > 500
            vol_1m=100.0        # > 75
        )
        
        should_reject, reason = self.filter.should_reject_entry(snapshot)
        
        assert should_reject == False
        assert "Liquidity gates passed" in reason


class TestRSIConfirmation:
    """Test RSI confirmation logic."""
    
    def setup_method(self):
        """Set up RSI confirmation."""
        self.rsi = RSIConfirmation(period=5, oversold=30.0, overbought=70.0)  # Short period for testing
    
    def test_insufficient_history(self):
        """Test RSI with insufficient price history."""
        # Add only 2 prices (need period + 1 = 6)
        self.rsi.update_prices("BTC-PERP", Decimal("100"))
        self.rsi.update_prices("BTC-PERP", Decimal("101"))
        
        confirmed, reason = self.rsi.confirm_buy_signal("BTC-PERP")
        
        assert confirmed == False
        assert "Insufficient price history" in reason
    
    def test_rsi_buy_confirmation(self):
        """Test RSI buy signal confirmation."""
        # Create trending up prices (should give high RSI)
        prices = [100, 105, 110, 115, 120, 125]
        for p in prices:
            self.rsi.update_prices("BTC-PERP", Decimal(str(p)))
        
        confirmed, reason = self.rsi.confirm_buy_signal("BTC-PERP")
        
        assert confirmed == True
        assert "RSI" in reason and "confirms buy" in reason
        
    def test_rsi_buy_rejection_oversold(self):
        """Test RSI buy rejection when oversold."""
        # Create trending down prices (should give low RSI)
        prices = [125, 120, 115, 110, 105, 100]
        for p in prices:
            self.rsi.update_prices("BTC-PERP", Decimal(str(p)))
        
        confirmed, reason = self.rsi.confirm_buy_signal("BTC-PERP")
        
        assert confirmed == False
        assert "too low" in reason
    
    def test_rsi_sell_confirmation(self):
        """Test RSI sell signal confirmation."""
        # Create trending down prices (should give low RSI, good for sell)
        prices = [125, 120, 115, 110, 105, 100]
        for p in prices:
            self.rsi.update_prices("BTC-PERP", Decimal(str(p)))
        
        confirmed, reason = self.rsi.confirm_sell_signal("BTC-PERP")
        
        assert confirmed == True
        assert "RSI" in reason and "confirms sell" in reason


class TestWeek2StrategyFilter:
    """Test combined Week 2 filter logic."""
    
    def setup_method(self):
        """Set up combined filter."""
        gates = LiquidityGates(
            spread_bps_max=20.0,
            l1_min=Decimal("50"),
            l10_min=Decimal("500"), 
            vol_1m_min=75.0
        )
        self.filter = Week2StrategyFilter(gates, enable_rsi=True)
    
    def test_liquidity_rejection_overrides_rsi(self):
        """Test liquidity rejection blocks entry regardless of RSI."""
        # Bad liquidity (high spread)
        snapshot = create_test_snapshot(spread_bps=30.0)  # Above threshold
        
        should_reject, reason = self.filter.should_reject_entry("buy", snapshot)
        
        assert should_reject == True
        assert "Liquidity:" in reason
        assert "Spread" in reason
    
    def test_rsi_disabled_mode(self):
        """Test filter with RSI disabled."""
        gates = LiquidityGates()  # Default gates
        filter_no_rsi = Week2StrategyFilter(gates, enable_rsi=False)
        
        snapshot = create_test_snapshot()  # Good liquidity
        
        should_reject, reason = filter_no_rsi.should_reject_entry("buy", snapshot)
        
        assert should_reject == False
        assert "All filters passed" in reason
    
    def test_combined_filter_pass(self):
        """Test entry passes both liquidity and RSI filters."""
        snapshot = create_test_snapshot()  # Good liquidity conditions
        
        # Add some price history for RSI (trending up for buy signal)
        # Need enough history for RSI calculation (period + 1)
        for i, price in enumerate([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128]):
            self.filter.rsi_confirmation.update_prices("BTC-PERP", Decimal(str(price)))
        
        should_reject, reason = self.filter.should_reject_entry("buy", snapshot)
        
        # Debug if failed
        if should_reject:
            print(f"Filter rejected with reason: {reason}")
            rsi = self.filter.rsi_confirmation.calculate_rsi("BTC-PERP")
            print(f"RSI value: {rsi}")
        
        assert should_reject == False
        assert "All filters passed" in reason