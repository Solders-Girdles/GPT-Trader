#!/usr/bin/env python3
"""
Test public endpoints without authentication to verify API connectivity.
"""

import json
import urllib.request
import urllib.error

print("Testing Coinbase Public Endpoints (No Auth)")
print("=" * 60)

# Test public endpoints
endpoints = [
    ("/api/v3/brokerage/products", "List all products"),
    ("/api/v3/brokerage/products/BTC-USD", "Get BTC-USD product details"),
    ("/api/v3/brokerage/market/products/BTC-USD/ticker", "Get BTC-USD ticker"),
    ("/api/v3/brokerage/market/products/BTC-USD/candles?granularity=ONE_MINUTE&limit=5", "Get BTC-USD candles"),
]

for path, description in endpoints:
    print(f"\n{description}")
    print(f"  GET {path}")
    
    url = f"https://api.coinbase.com{path}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            status = response.getcode()
            data = response.read().decode()
            result = json.loads(data) if data else {}
            
            if isinstance(result, dict):
                if 'products' in result:
                    print(f"  ✅ Success! Found {len(result.get('products', []))} products")
                elif 'product_id' in result:
                    print(f"  ✅ Success! Product: {result.get('product_id')}")
                    print(f"     Status: {result.get('status')}, Quote: {result.get('quote_currency')}")
                elif 'trades' in result:
                    trades = result.get('trades', [])
                    if trades:
                        latest = trades[0]
                        print(f"  ✅ Success! Latest price: ${latest.get('price')}")
                elif 'candles' in result:
                    candles = result.get('candles', [])
                    print(f"  ✅ Success! Got {len(candles)} candles")
                    if candles:
                        latest = candles[0]
                        print(f"     Latest close: ${latest.get('close')}")
                else:
                    print(f"  ✅ Success! Response keys: {list(result.keys())[:5]}")
            else:
                print(f"  ✅ Success! Got response")
                
    except urllib.error.HTTPError as e:
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        if hasattr(e, 'read'):
            error_body = e.read().decode()[:100]
            print(f"     Error: {error_body}")
    except Exception as e:
        print(f"  ❌ Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("\nSummary:")
print("Public endpoints should work without authentication.")
print("This verifies basic API connectivity.")