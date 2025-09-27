#!/usr/bin/env python3
"""
Demonstration of transitioning from mock data to real Coinbase API data.

This script shows:
1. How the system currently uses mock data for testing
2. How to enable real Coinbase API data
3. Comparison between mock and real data behavior

Usage:
    # Default (mock data)
    python demos/coinbase_data_transition_demo.py
    
    # With real data (requires setting up Coinbase API)
    COINBASE_USE_REAL_DATA=1 python demos/coinbase_data_transition_demo.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def demo_mock_data():
    """Demonstrate mock data usage (current behavior)."""
    print("\n" + "="*60)
    print("MOCK DATA MODE (Current Default)")
    print("="*60)
    
    # Force mock mode
    os.environ['COINBASE_USE_REAL_DATA'] = '0'
    
    from bot_v2.data_providers import get_data_provider
    
    provider = get_data_provider()
    print(f"Provider: {provider.__class__.__name__}")
    
    # Get some test data
    symbols = ['BTC', 'ETH']
    for symbol in symbols:
        price = provider.get_current_price(symbol)
        print(f"  {symbol}: ${price:,.2f} (deterministic mock price)")
    
    # Historical data
    df = provider.get_historical_data('BTC', period='3d')
    print(f"\nBTC Historical (3 days):")
    print(f"  Shape: {df.shape}")
    print(f"  Latest close: ${df['Close'].iloc[-1]:,.2f}")
    print(f"  Note: Mock data is deterministic based on symbol hash")


def demo_real_data_setup():
    """Show how to set up real Coinbase data."""
    print("\n" + "="*60)
    print("REAL DATA SETUP")
    print("="*60)
    
    print("""
To use real Coinbase API data:

1. Set environment variable:
   export COINBASE_USE_REAL_DATA=1

2. For public market data (no auth required):
   - The CoinbaseDataProvider will fetch real market prices
   - No API keys needed for public endpoints

3. For authenticated features (optional):
   - Set COINBASE_API_KEY, COINBASE_API_SECRET, etc.
   - Or use CDP (JWT) authentication for advanced features

4. Enable streaming (optional):
   export COINBASE_ENABLE_STREAMING=1
   - Provides real-time WebSocket updates
   - Reduces API calls for frequently accessed data
""")


def demo_comparison():
    """Compare mock vs real data behavior."""
    print("\n" + "="*60)
    print("MOCK vs REAL DATA COMPARISON")
    print("="*60)
    
    print("""
Key Differences:

MOCK DATA (Testing):
  ✓ Deterministic - same results every run
  ✓ No network calls - fast and reliable
  ✓ No API limits - unlimited testing
  ✓ Predictable for unit tests
  ✗ Not real market conditions
  ✗ Can't test real volatility

REAL DATA (Production):
  ✓ Actual market prices
  ✓ Real volatility and spreads
  ✓ Live market conditions
  ✓ WebSocket streaming available
  ✗ Network dependent
  ✗ API rate limits apply
  ✗ Non-deterministic testing
""")


def demo_configuration():
    """Show current configuration."""
    print("\n" + "="*60)
    print("CURRENT CONFIGURATION")
    print("="*60)
    
    config_vars = {
        'COINBASE_USE_REAL_DATA': os.environ.get('COINBASE_USE_REAL_DATA', '0'),
        'COINBASE_ENABLE_STREAMING': os.environ.get('COINBASE_ENABLE_STREAMING', '0'),
        'COINBASE_API_MODE': os.environ.get('COINBASE_API_MODE', 'advanced'),
        'COINBASE_SANDBOX': os.environ.get('COINBASE_SANDBOX', '0'),
        'TESTING': os.environ.get('TESTING', 'false')
    }
    
    for key, value in config_vars.items():
        status = "✓" if value == '1' or value == 'true' else "✗"
        print(f"  {status} {key}: {value}")
    
    # Determine what mode we're in
    use_real = config_vars['COINBASE_USE_REAL_DATA'] == '1'
    
    print(f"\nData Mode: {'REAL Coinbase API' if use_real else 'MOCK (deterministic)'}")


def main():
    """Run the demonstration."""
    print("\n" + "="*60)
    print("COINBASE DATA PROVIDER TRANSITION DEMO")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show current configuration
    demo_configuration()
    
    # Demo mock data (default)
    demo_mock_data()
    
    # Show how to set up real data
    demo_real_data_setup()
    
    # Show comparison
    demo_comparison()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("""
Next Steps:
1. Continue using mock data for testing (default)
2. Set COINBASE_USE_REAL_DATA=1 to switch to real API
3. See demos/test_coinbase_data_provider.py for full examples
""")


if __name__ == "__main__":
    main()
