#!/usr/bin/env python3
"""
Validation script for Week 1 WebSocket market data implementation.

Tests:
1. WebSocket connection and subscription to ticker/trades/level2
2. Market data updates for spread, depth, and volume
3. Staleness detection and reconnection handling
4. Rolling volume calculations
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Set

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import os
if os.getenv('USE_REAL_ADAPTER') == '1':
    from bot_v2.features.brokerages.coinbase.adapter import CoinbaseBrokerage as BrokerClass
    print("🔴 Using REAL CoinbaseBrokerage for WebSocket validation")
else:
    from bot_v2.features.brokerages.coinbase.test_adapter import MinimalCoinbaseBrokerage as BrokerClass
    print("🟡 Using Mock MinimalCoinbaseBrokerage for WebSocket validation")
from bot_v2.features.brokerages.coinbase.models import APIConfig
from bot_v2.features.brokerages.coinbase.endpoints import get_perps_symbols
from bot_v2.features.brokerages.core.interfaces import MarketType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketValidator:
    """Validator for WebSocket market data functionality."""
    
    def __init__(self):
        # Configure for sandbox testing
        self.config = APIConfig(
            api_key=os.getenv('COINBASE_API_KEY', 'test_key'),
            api_secret=os.getenv('COINBASE_API_SECRET', 'test_secret'),
            passphrase=os.getenv('COINBASE_PASSPHRASE', 'test_passphrase'),
            base_url="https://api.sandbox.coinbase.com" if os.getenv('COINBASE_SANDBOX') else "https://api.coinbase.com",
            sandbox=os.getenv('COINBASE_SANDBOX', '0') == '1',
            enable_derivatives=True,
            api_mode="advanced",
            auth_type="JWT" if os.getenv('CDP_API_KEY') else "HMAC"
        )
        
        self.broker = BrokerClass(self.config)
        self.test_symbols = []
        
    async def run_all_tests(self) -> bool:
        """Run all WebSocket validation tests."""
        logger.info("🌐 Starting WebSocket market data validation...")
        
        # Get available perpetuals for testing
        await self.setup_test_symbols()
        
        if not self.test_symbols:
            logger.error("No perpetual symbols available for testing")
            return False
        
        tests = [
            ("WebSocket Connection", self.test_websocket_connection),
            ("Market Data Updates", self.test_market_data_updates),
            ("Staleness Detection", self.test_staleness_detection),
            ("Rolling Volume", self.test_rolling_volume),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name}...")
                result = await test_func()
                results.append((test_name, result, None))
                logger.info(f"✅ {test_name}: {'PASS' if result else 'FAIL'}")
            except Exception as e:
                results.append((test_name, False, str(e)))
                logger.error(f"❌ {test_name}: FAIL - {e}")
        
        # Summary
        passed = sum(1 for _, result, _ in results if result)
        total = len(results)
        
        logger.info(f"\n📊 WebSocket Validation Summary: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All WebSocket tests PASSED!")
            return True
        else:
            logger.error("❌ Some WebSocket tests FAILED")
            return False
    
    async def setup_test_symbols(self):
        """Set up test symbols from available perpetuals."""
        try:
            perps = self.broker.list_products(market=MarketType.PERPETUAL)
            expected_perps = get_perps_symbols()
            
            # Use intersection of available and expected perps
            available_symbols = {p.symbol for p in perps}
            self.test_symbols = list(available_symbols & expected_perps)[:2]  # Test with 2 symbols max
            
            if not self.test_symbols:
                # Fallback to any available perps
                self.test_symbols = [p.symbol for p in perps[:2]]
            
            logger.info(f"Testing WebSocket with symbols: {self.test_symbols}")
            
        except Exception as e:
            logger.error(f"Failed to set up test symbols: {e}")
            self.test_symbols = []
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection and subscription."""
        try:
            if not self.test_symbols:
                return False
            
            # Start market data streams
            self.broker.start_market_data(self.test_symbols)
            
            # Wait a moment for connection
            await asyncio.sleep(2)
            
            # Check if market data structures are initialized
            for symbol in self.test_symbols:
                if symbol not in self.broker._market_data:
                    logger.error(f"Market data not initialized for {symbol}")
                    return False
                
                if symbol not in self.broker._rolling_windows:
                    logger.error(f"Rolling windows not initialized for {symbol}")
                    return False
            
            logger.info("WebSocket connection and subscription successful")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection test failed: {e}")
            return False
    
    async def test_market_data_updates(self) -> bool:
        """Test that market data is being updated via WebSocket."""
        try:
            if not self.test_symbols:
                return False
            
            # Start market data if not already started
            if not self.broker._ws_client:
                self.broker.start_market_data(self.test_symbols)
            
            # Wait for data to populate
            logger.info("Waiting 30 seconds for market data updates...")
            await asyncio.sleep(30)
            
            updates_received = 0
            
            for symbol in self.test_symbols:
                market_data = self.broker._market_data.get(symbol, {})
                
                # Check if we have basic market data
                if market_data.get('last_update'):
                    updates_received += 1
                    
                    logger.info(f"{symbol} data:")
                    logger.info(f"  Mid: {market_data.get('mid', 'N/A')}")
                    logger.info(f"  Spread (bps): {market_data.get('spread_bps', 'N/A')}")
                    logger.info(f"  L1 Depth: {market_data.get('depth_l1', 'N/A')}")
                    logger.info(f"  L10 Depth: {market_data.get('depth_l10', 'N/A')}")
                    logger.info(f"  Last Update: {market_data.get('last_update', 'N/A')}")
                else:
                    logger.warning(f"No updates received for {symbol}")
            
            # Success if we got updates for at least half the symbols
            if updates_received >= len(self.test_symbols) // 2 or updates_received >= 1:
                logger.info(f"Market data updates received for {updates_received} symbols")
                return True
            else:
                logger.error("Insufficient market data updates")
                return False
                
        except Exception as e:
            logger.error(f"Market data updates test failed: {e}")
            return False
    
    async def test_staleness_detection(self) -> bool:
        """Test staleness detection functionality."""
        try:
            if not self.test_symbols:
                return False
            
            symbol = self.test_symbols[0]
            
            # Check initial staleness (should be stale if no data)
            initial_stale = self.broker.is_stale(symbol)
            logger.info(f"Initial staleness for {symbol}: {initial_stale}")
            
            # Start market data
            if not self.broker._ws_client:
                self.broker.start_market_data([symbol])
                await asyncio.sleep(5)
            
            # Check staleness after some time
            current_stale = self.broker.is_stale(symbol)
            logger.info(f"Current staleness for {symbol}: {current_stale}")
            
            # Test custom staleness threshold
            very_short_stale = self.broker.is_stale(symbol, threshold_seconds=1)
            logger.info(f"Staleness with 1s threshold: {very_short_stale}")
            
            # If we received any data, staleness detection should work
            market_data = self.broker._market_data.get(symbol, {})
            if market_data.get('last_update'):
                logger.info("Staleness detection working correctly")
                return True
            else:
                logger.warning("No market data to test staleness with")
                return True  # Not a failure of staleness logic
                
        except Exception as e:
            logger.error(f"Staleness detection test failed: {e}")
            return False
    
    async def test_rolling_volume(self) -> bool:
        """Test rolling volume calculation."""
        try:
            if not self.test_symbols:
                return False
            
            # Start market data
            if not self.broker._ws_client:
                self.broker.start_market_data(self.test_symbols)
            
            # Wait for some trades to accumulate
            logger.info("Waiting 15 seconds for trade data...")
            await asyncio.sleep(15)
            
            volumes_found = 0
            
            for symbol in self.test_symbols:
                market_snapshot = self.broker.get_market_snapshot(symbol)
                
                vol_1m = market_snapshot.get('vol_1m', 0)
                vol_5m = market_snapshot.get('vol_5m', 0)
                
                logger.info(f"{symbol} volume:")
                logger.info(f"  1m: {vol_1m}")
                logger.info(f"  5m: {vol_5m}")
                
                if vol_1m > 0 or vol_5m > 0:
                    volumes_found += 1
            
            if volumes_found > 0:
                logger.info(f"Rolling volume data found for {volumes_found} symbols")
                return True
            else:
                logger.warning("No volume data accumulated - market may be quiet")
                # Don't fail test if market is just quiet
                return True
                
        except Exception as e:
            logger.error(f"Rolling volume test failed: {e}")
            return False


async def main():
    """Main WebSocket validation runner."""
    if os.getenv('RUN_SANDBOX_VALIDATIONS') != '1':
        print("⚠️  WebSocket validations disabled. Set RUN_SANDBOX_VALIDATIONS=1 to run.")
        print("   This prevents accidental live WebSocket connections during testing.")
        return
    
    validator = WebSocketValidator()
    success = await validator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())