#!/usr/bin/env python3
"""
Test script to demonstrate Coinbase data provider usage.

This script shows how to switch between mock and real Coinbase data
for backtesting and analysis.

Usage:
    # Use mock data (default)
    python demos/test_coinbase_data_provider.py
    
    # Use real Coinbase data
    COINBASE_USE_REAL_DATA=1 python demos/test_coinbase_data_provider.py
    
    # Use real data with WebSocket streaming
    COINBASE_USE_REAL_DATA=1 COINBASE_ENABLE_STREAMING=1 python demos/test_coinbase_data_provider.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot_v2.data_providers import get_data_provider
from bot_v2.data_providers.coinbase_provider import CoinbaseDataProvider, create_coinbase_provider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_mock_vs_real():
    """Compare mock data vs real Coinbase data."""
    
    print("\n" + "="*60)
    print("COINBASE DATA PROVIDER TEST")
    print("="*60)
    
    # Check configuration
    use_real = os.environ.get('COINBASE_USE_REAL_DATA', '0') == '1'
    use_streaming = os.environ.get('COINBASE_ENABLE_STREAMING', '0') == '1'
    
    print(f"\nConfiguration:")
    print(f"  COINBASE_USE_REAL_DATA: {use_real}")
    print(f"  COINBASE_ENABLE_STREAMING: {use_streaming}")
    print()
    
    # Get data provider (will auto-detect based on env vars)
    provider = get_data_provider()
    print(f"Provider type: {provider.__class__.__name__}")
    
    # Test symbols
    symbols = ['BTC', 'ETH', 'SOL']
    
    # Test 1: Get current prices
    print(f"\n{'='*40}")
    print("TEST 1: Current Prices")
    print(f"{'='*40}")
    
    for symbol in symbols:
        try:
            price = provider.get_current_price(symbol)
            print(f"{symbol:6} ${price:,.2f}")
        except Exception as e:
            print(f"{symbol:6} Error: {e}")
    
    # Test 2: Get historical data
    print(f"\n{'='*40}")
    print("TEST 2: Historical Data (Last 7 days)")
    print(f"{'='*40}")
    
    for symbol in symbols[:1]:  # Just test BTC for brevity
        try:
            df = provider.get_historical_data(symbol, period='7d', interval='1d')
            print(f"\n{symbol} Historical Data:")
            print(f"  Shape: {df.shape}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"\nLast 3 days:")
            print(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(3))
            
            # Calculate simple metrics
            returns = df['Close'].pct_change().dropna()
            print(f"\nMetrics:")
            print(f"  Avg daily return: {returns.mean():.4%}")
            print(f"  Volatility: {returns.std():.4%}")
            print(f"  Max drawdown: {(df['Close'] / df['Close'].cummax() - 1).min():.4%}")
            
        except Exception as e:
            print(f"{symbol} Error: {e}")
    
    # Test 3: Get intraday data (if using real data)
    if use_real:
        print(f"\n{'='*40}")
        print("TEST 3: Intraday Data (Last 24 hours)")
        print(f"{'='*40}")
        
        try:
            df = provider.get_historical_data('BTC', period='1d', interval='1h')
            print(f"\nBTC Hourly Data:")
            print(f"  Shape: {df.shape}")
            print(f"\nLast 5 hours:")
            print(df[['Close', 'Volume']].tail(5))
        except Exception as e:
            print(f"Error getting intraday data: {e}")
    
    # Test 4: Multiple symbols
    print(f"\n{'='*40}")
    print("TEST 4: Multiple Symbols")
    print(f"{'='*40}")
    
    data = provider.get_multiple_symbols(symbols, period='3d')
    for symbol, df in data.items():
        latest = df['Close'].iloc[-1] if not df.empty else 0
        change = ((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100) if len(df) > 1 else 0
        print(f"{symbol:6} Latest: ${latest:,.2f}  3d Change: {change:+.2f}%")
    
    # Test 5: Market hours check
    print(f"\n{'='*40}")
    print("TEST 5: Market Status")
    print(f"{'='*40}")
    
    is_open = provider.is_market_open()
    print(f"Market is {'OPEN' if is_open else 'CLOSED'}")
    print(f"(Note: Crypto markets are 24/7)")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


def test_streaming_mode():
    """Test WebSocket streaming mode (if enabled)."""
    
    if os.environ.get('COINBASE_ENABLE_STREAMING', '0') != '1':
        print("\nStreaming test skipped (COINBASE_ENABLE_STREAMING=0)")
        return
    
    print(f"\n{'='*40}")
    print("STREAMING MODE TEST")
    print(f"{'='*40}")
    
    # Create provider with streaming
    provider = CoinbaseDataProvider(enable_streaming=True)
    
    # Use context manager to handle streaming lifecycle
    with provider:
        symbols = ['BTC-PERP', 'ETH-PERP']
        
        print("\nSubscribing to symbols:", symbols)
        data = provider.get_multiple_symbols(symbols, period='1d')
        
        print("\nWaiting for streaming updates...")
        import time
        for i in range(5):
            time.sleep(1)
            for symbol in symbols:
                price = provider.get_current_price(symbol)
                print(f"  {symbol}: ${price:,.2f}")
        
    print("\nStreaming stopped")


def test_configuration_modes():
    """Test different configuration modes."""
    
    print(f"\n{'='*40}")
    print("CONFIGURATION MODES TEST")
    print(f"{'='*40}")
    
    # Test 1: Explicit mock mode
    print("\n1. Mock Mode (explicit):")
    from bot_v2.data_providers import MockProvider
    mock_provider = MockProvider()
    btc_price = mock_provider.get_current_price('BTC')
    print(f"   BTC Price (mock): ${btc_price:,.2f}")
    
    # Test 2: Factory with explicit type
    print("\n2. Factory with type='coinbase':")
    try:
        # Temporarily set env var
        old_val = os.environ.get('COINBASE_USE_REAL_DATA', '')
        os.environ['COINBASE_USE_REAL_DATA'] = '1'
        
        cb_provider = get_data_provider('coinbase')
        print(f"   Provider: {cb_provider.__class__.__name__}")
        
        # Restore env var
        if old_val:
            os.environ['COINBASE_USE_REAL_DATA'] = old_val
        else:
            del os.environ['COINBASE_USE_REAL_DATA']
            
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test 3: Helper function
    print("\n3. Helper function create_coinbase_provider():")
    
    # Mock mode
    mock_cb = create_coinbase_provider(use_real_data=False)
    print(f"   use_real_data=False -> {mock_cb.__class__.__name__}")
    
    # Real mode (if credentials available)
    try:
        real_cb = create_coinbase_provider(use_real_data=True, enable_streaming=False)
        print(f"   use_real_data=True  -> {real_cb.__class__.__name__}")
    except Exception as e:
        print(f"   use_real_data=True  -> Error (likely missing credentials)")


if __name__ == "__main__":
    try:
        # Run main test
        test_mock_vs_real()
        
        # Run configuration test
        test_configuration_modes()
        
        # Run streaming test if enabled
        test_streaming_mode()
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
