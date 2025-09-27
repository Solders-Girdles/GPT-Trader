#!/usr/bin/env python3
"""Test Coinbase INTX (International Exchange) API for perpetuals."""

import json
import os
import requests
from datetime import datetime
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import jwt
import time

def generate_jwt():
    """Generate JWT for Coinbase Advanced Trade API."""
    api_key = os.environ.get("COINBASE_PROD_CDP_API_KEY")
    private_key_pem = os.environ.get("COINBASE_PROD_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key_pem:
        raise ValueError("Missing CDP credentials in environment")
    
    private_key_pem = private_key_pem.replace("\\n", "\n")
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )
    
    issued_at = int(time.time())
    expiration = issued_at + 120
    
    headers = {
        "alg": "ES256",
        "kid": api_key,
        "nonce": str(int(time.time() * 1000))
    }
    
    payload = {
        "iss": "coinbase-cloud",
        "sub": api_key,
        "iat": issued_at,
        "exp": expiration,
        "nbf": issued_at,
        "aud": "retail_rest_api_proxy"
    }
    
    token = jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers=headers
    )
    
    return token

def test_intx_endpoints():
    """Test INTX (International Exchange) endpoints for perpetuals."""
    print("=" * 70)
    print("🌍 Testing Coinbase INTX (International Exchange) API")
    print("=" * 70)
    print("Documentation: https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/perpetuals/")
    
    try:
        token = generate_jwt()
        print("\n✅ JWT generated successfully")
    except Exception as e:
        print(f"❌ Failed to generate JWT: {e}")
        return
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test INTX perpetuals endpoints from the documentation
    endpoints = [
        # Portfolio endpoints
        ("INTX Portfolio Summary", "GET", "https://api.coinbase.com/api/v3/brokerage/intx/portfolio"),
        ("INTX Positions", "GET", "https://api.coinbase.com/api/v3/brokerage/intx/positions"),
        ("INTX Balances", "GET", "https://api.coinbase.com/api/v3/brokerage/intx/balances"),
        
        # Market data endpoints
        ("INTX Products", "GET", "https://api.coinbase.com/api/v3/brokerage/intx/products"),
        ("INTX BTC-PERP Ticker", "GET", "https://api.coinbase.com/api/v3/brokerage/intx/products/BTC-PERP/ticker"),
        
        # Trading endpoints
        ("INTX Orders", "GET", "https://api.coinbase.com/api/v3/brokerage/intx/orders"),
        
        # Also try the documented perpetuals endpoints
        ("Perpetuals Portfolio", "GET", "https://api.coinbase.com/api/v3/brokerage/perpetuals/portfolio_summary"),
        ("Perpetuals Positions", "GET", "https://api.coinbase.com/api/v3/brokerage/perpetuals/positions"),
    ]
    
    for name, method, url in endpoints:
        print(f"\n📡 Testing: {name}")
        print(f"   URL: {url}")
        
        try:
            if method == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            else:
                response = requests.request(method, url, headers=headers, timeout=10)
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success!")
                
                # Show relevant data based on endpoint
                if "portfolio" in name.lower():
                    print(f"   Portfolio data available")
                    if data:
                        print(f"   Response preview: {json.dumps(data, indent=2)[:500]}")
                
                elif "positions" in name.lower():
                    positions = data.get("positions", data) if isinstance(data, dict) else data
                    if isinstance(positions, list):
                        print(f"   Found {len(positions)} positions")
                        if positions:
                            print(f"   First position: {json.dumps(positions[0], indent=2)[:300]}")
                    else:
                        print(f"   Response: {json.dumps(data, indent=2)[:500]}")
                
                elif "products" in name.lower() and "ticker" not in name.lower():
                    products = data.get("products", data) if isinstance(data, dict) else data
                    if isinstance(products, list):
                        print(f"   Found {len(products)} products")
                        # Look for perpetuals
                        perps = [p for p in products if "PERP" in str(p.get("product_id", "")).upper()]
                        if perps:
                            print(f"   🎯 Found {len(perps)} perpetual products!")
                            print("\n   Perpetual products:")
                            for p in perps[:10]:
                                print(f"     - {p.get('product_id')}: {p.get('base_currency')}/{p.get('quote_currency')}")
                            
                            # Show first perpetual structure
                            if perps:
                                print(f"\n   Sample perpetual structure:")
                                print(json.dumps(perps[0], indent=2)[:500])
                    else:
                        print(f"   Response: {json.dumps(data, indent=2)[:500]}")
                
                elif "ticker" in name.lower():
                    print(f"   Ticker data:")
                    print(json.dumps(data, indent=2)[:500])
                
                else:
                    # Show first 500 chars of response
                    print(f"   Response preview: {json.dumps(data, indent=2)[:500]}")
                    
            elif response.status_code == 401:
                print(f"   ❌ Unauthorized - No INTX/Perpetuals access")
                error_detail = response.json() if response.text else {}
                if error_detail:
                    print(f"   Error: {error_detail.get('message', response.text[:200])}")
            elif response.status_code == 403:
                print(f"   ❌ Forbidden - Account may not have INTX access")
                print(f"   Response: {response.text[:200]}")
            elif response.status_code == 404:
                print(f"   ❌ Not Found - Endpoint may not exist or product not available")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    print("""
If you're getting 401/403 errors on INTX endpoints, you need to:

1. Enable International Exchange (INTX) access on your Coinbase account
2. This is different from regular Advanced Trade API access
3. INTX is where perpetual futures (BTC-PERP, ETH-PERP, etc.) are traded
4. Contact Coinbase support or check account settings for INTX eligibility

Note: INTX may have geographic restrictions or require additional verification.
""")


if __name__ == "__main__":
    test_intx_endpoints()