"""
Unit tests for staleness detection functionality.
"""

import pytest
from datetime import datetime, timedelta
from bot_v2.features.brokerages.coinbase.test_adapter import MinimalCoinbaseBrokerage
from bot_v2.features.brokerages.coinbase.models import APIConfig


class TestStalenessDetection:
    """Test staleness detection with various time thresholds."""
    
    def setup_method(self):
        """Set up test adapter."""
        config = APIConfig(
            api_key="test",
            api_secret="test", 
            passphrase="test",
            base_url="https://api.sandbox.coinbase.com",
            sandbox=True,
            enable_derivatives=True,
            api_mode="advanced",
            auth_type="HMAC"
        )
        self.adapter = MinimalCoinbaseBrokerage(config)
        
    def test_fresh_vs_stale_toggles(self):
        """Test fresh vs stale toggles at 1s/10s thresholds."""
        symbol = "BTC-PERP"
        
        # Initially stale (no market data)
        assert self.adapter.is_stale(symbol, threshold_seconds=10) == True
        assert self.adapter.is_stale(symbol, threshold_seconds=1) == True
        
        # Start market data (makes it fresh)
        self.adapter.start_market_data([symbol])
        
        # Should now be fresh at both thresholds
        assert self.adapter.is_stale(symbol, threshold_seconds=10) == False
        assert self.adapter.is_stale(symbol, threshold_seconds=1) == False
        
    def test_staleness_behavior_matches_validator(self):
        """Test staleness behavior matches what validation scripts expect."""
        symbol = "ETH-PERP"
        
        # Test sequence matching validation script:
        # 1. Check initial staleness (should be stale)
        initial_stale = self.adapter.is_stale(symbol)
        assert initial_stale == True
        
        # 2. Start market data
        self.adapter.start_market_data([symbol])
        
        # 3. Check after starting (should be fresh)  
        current_stale = self.adapter.is_stale(symbol)
        assert current_stale == False
        
        # 4. Check with very short threshold
        very_short_stale = self.adapter.is_stale(symbol, threshold_seconds=1)
        assert very_short_stale == False  # Still fresh in mock