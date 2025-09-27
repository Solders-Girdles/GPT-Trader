"""
Unit tests for market data utility classes.
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from bot_v2.features.brokerages.coinbase.market_data_utils import (
    RollingWindow, DepthSnapshot, TradeTapeAgg
)


class TestDepthSnapshot:
    """Test DepthSnapshot L1/L10 depth and spread calculations."""
    
    def test_l1_l10_depth_correctness(self):
        """Test L1 and L10 depth calculations on crafted levels."""
        levels = [
            (Decimal('50000'), Decimal('1.0'), 'bid'),
            (Decimal('49990'), Decimal('2.0'), 'bid'),
            (Decimal('49980'), Decimal('3.0'), 'bid'),
            (Decimal('50010'), Decimal('1.5'), 'ask'),
            (Decimal('50020'), Decimal('2.5'), 'ask'),
            (Decimal('50030'), Decimal('3.5'), 'ask'),
        ]
        
        snapshot = DepthSnapshot(levels)
        
        # Test L1 depth (min of top bid/ask)
        l1_depth = snapshot.get_l1_depth()
        assert l1_depth == Decimal('1.0')  # min(1.0 bid, 1.5 ask)
        
        # Test L10 depth (sum of all levels)
        l10_depth = snapshot.get_l10_depth() 
        assert l10_depth == Decimal('13.5')  # 1+2+3 + 1.5+2.5+3.5
        
    def test_spread_bps_calculation(self):
        """Test spread in basis points is computed correctly."""
        levels = [
            (Decimal('50000'), Decimal('1.0'), 'bid'),
            (Decimal('50010'), Decimal('1.0'), 'ask'),  # 10 dollar spread
        ]
        
        snapshot = DepthSnapshot(levels)
        spread_bps = snapshot.spread_bps
        
        # (50010 - 50000) / 50000 * 10000 = 2 bps
        assert spread_bps == 2.0
        
    def test_mid_price(self):
        """Test mid price calculation."""
        levels = [
            (Decimal('50000'), Decimal('1.0'), 'bid'),
            (Decimal('50020'), Decimal('1.0'), 'ask'),
        ]
        
        snapshot = DepthSnapshot(levels)
        mid = snapshot.mid
        
        assert mid == Decimal('50010')  # (50000 + 50020) / 2


class TestRollingWindow:
    """Test RollingWindow cleanup and stats across time boundaries."""
    
    def test_cleanup_and_stats(self):
        """Test values drop as intended on cleanup across time window."""
        window = RollingWindow(duration_seconds=10)
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add values within window
        window.add(100.0, base_time)
        window.add(200.0, base_time + timedelta(seconds=5))
        window.add(300.0, base_time + timedelta(seconds=8))
        
        # All values should be present
        stats = window.get_stats()
        assert stats['count'] == 3
        assert stats['sum'] == 600.0
        assert stats['avg'] == 200.0
        
        # Add value that triggers cleanup (15 seconds later)
        window.add(400.0, base_time + timedelta(seconds=15))
        
        # Cleanup should remove values older than 10s from the 15s mark
        # Cutoff = 15s - 10s = 5s, so values at 0s should be removed
        # Values at 5s and 8s should remain, plus the new 400.0 value
        stats = window.get_stats()
        # Expected: 200.0 (5s) + 300.0 (8s) + 400.0 (15s) = 3 values, 900.0 sum
        assert stats['count'] == 3
        assert stats['sum'] == 900.0
        assert stats['avg'] == 300.0


class TestTradeTapeAgg:
    """Test trade tape aggregation functionality."""
    
    def test_vwap_calculation(self):
        """Test volume-weighted average price calculation."""
        agg = TradeTapeAgg(duration_seconds=60)
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add trades
        agg.add_trade(Decimal('100'), Decimal('10'), 'buy', base_time)
        agg.add_trade(Decimal('200'), Decimal('5'), 'sell', base_time + timedelta(seconds=10))
        
        # VWAP = (100*10 + 200*5) / (10+5) = 2000/15 = 133.33...
        vwap = agg.get_vwap()
        expected = (Decimal('100') * Decimal('10') + Decimal('200') * Decimal('5')) / Decimal('15')
        assert abs(vwap - expected) < Decimal('0.01')
        
    def test_aggressor_ratio(self):
        """Test buy-side aggressor ratio calculation."""
        agg = TradeTapeAgg(duration_seconds=60)
        
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Add 2 buy trades, 1 sell trade
        agg.add_trade(Decimal('100'), Decimal('1'), 'buy', base_time)
        agg.add_trade(Decimal('100'), Decimal('1'), 'buy', base_time + timedelta(seconds=10))
        agg.add_trade(Decimal('100'), Decimal('1'), 'sell', base_time + timedelta(seconds=20))
        
        ratio = agg.get_aggressor_ratio()
        assert ratio == 2/3  # 2 buy trades out of 3 total