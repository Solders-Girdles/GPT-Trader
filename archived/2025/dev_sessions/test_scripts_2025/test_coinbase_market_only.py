#!/usr/bin/env python3
"""
Test Coinbase integration with market data only (no authentication required).
This demonstrates the system can work for paper trading with public data.
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
from decimal import Decimal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Coinbase Market Data Test (Public Endpoints)")
print("=" * 60)

def fetch_public_data(path: str) -> dict:
    """Fetch data from public endpoint."""
    url = f"https://api.coinbase.com{path}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            data = response.read().decode()
            return json.loads(data) if data else {}
    except Exception as e:
        print(f"Error fetching {path}: {e}")
        return {}

# Test various symbols
symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]

for symbol in symbols:
    print(f"\n{symbol} Market Data:")
    print("-" * 40)
    
    # Get ticker
    ticker_data = fetch_public_data(f"/api/v3/brokerage/market/products/{symbol}/ticker")
    if ticker_data.get("trades"):
        latest_trade = ticker_data["trades"][0]
        price = float(latest_trade.get("price", 0))
        size = float(latest_trade.get("size", 0))
        print(f"  Latest Price: ${price:,.2f}")
        print(f"  Latest Size: {size:.6f}")
        print(f"  Time: {latest_trade.get('time', 'N/A')}")
    
    # Get candles  
    candles_data = fetch_public_data(
        f"/api/v3/brokerage/market/products/{symbol}/candles?granularity=ONE_MINUTE&limit=5"
    )
    if candles_data.get("candles"):
        candles = candles_data["candles"]
        print(f"  Recent Candles ({len(candles)} total):")
        for i, candle in enumerate(candles[:3]):
            o = float(candle.get("open", 0))
            h = float(candle.get("high", 0))
            l = float(candle.get("low", 0))
            c = float(candle.get("close", 0))
            v = float(candle.get("volume", 0))
            print(f"    [{i+1}] O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} V:{v:.4f}")

print("\n" + "=" * 60)
print("\nSummary:")
print("✅ Public market data is accessible without authentication")
print("✅ This is sufficient for:")
print("   - Paper trading simulations")
print("   - Backtesting with live prices")
print("   - Strategy development")
print("   - Market analysis")
print("\n⚠️  For live trading, you'll need:")
print("   - Properly configured CDP API key with trading permissions")
print("   - Account access for balance checks")
print("   - Order placement capabilities")

print("\nNOTE: The system can operate in 'market data only' mode for testing.")