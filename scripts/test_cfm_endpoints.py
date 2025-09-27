#!/usr/bin/env python3
"""Test Coinbase Futures Management (CFM) endpoints for perpetuals."""

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

def test_cfm_endpoints():
    """Test CFM (Coinbase Futures Management) endpoints."""
    print("=" * 70)
    print("🔍 Testing Coinbase Futures Management (CFM) Endpoints")
    print("=" * 70)
    
    try:
        token = generate_jwt()
        print("✅ JWT generated successfully")
    except Exception as e:
        print(f"❌ Failed to generate JWT: {e}")
        return
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test various CFM endpoints
    endpoints = [
        # CFM-specific endpoints
        ("CFM Positions", "GET", "https://api.coinbase.com/api/v3/brokerage/cfm/positions"),
        ("CFM Balance Summary", "GET", "https://api.coinbase.com/api/v3/brokerage/cfm/balance_summary"),
        ("CFM Sweeps", "GET", "https://api.coinbase.com/api/v3/brokerage/cfm/sweeps"),
        ("CFM Trading Config", "GET", "https://api.coinbase.com/api/v3/brokerage/cfm/trading_config"),
        
        # Try different product ID formats
        ("BTC-PERP Ticker", "GET", "https://api.coinbase.com/api/v3/brokerage/best_bid_ask?product_ids=BTC-PERP"),
        ("BTC-USD-PERP Ticker", "GET", "https://api.coinbase.com/api/v3/brokerage/best_bid_ask?product_ids=BTC-USD-PERP"),
        ("BTCPERP Ticker", "GET", "https://api.coinbase.com/api/v3/brokerage/best_bid_ask?product_ids=BTCPERP"),
        
        # Futures-specific endpoints (if they exist)
        ("Futures Products", "GET", "https://api.coinbase.com/api/v3/brokerage/futures/products"),
        ("Derivatives Products", "GET", "https://api.coinbase.com/api/v3/brokerage/derivatives/products"),
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
                
                # Show relevant data
                if "positions" in data:
                    positions = data.get("positions", [])
                    print(f"   Found {len(positions)} positions")
                    if positions:
                        print(f"   Sample position: {positions[0].get('product_id', 'N/A')}")
                
                elif "balance_summary" in data:
                    print(f"   Balance data available")
                    
                elif "products" in data:
                    products = data.get("products", [])
                    print(f"   Found {len(products)} products")
                    # Look for perpetuals
                    perps = [p for p in products if "PERP" in str(p.get("product_id", "")).upper()]
                    if perps:
                        print(f"   Found {len(perps)} perpetual products!")
                        for p in perps[:3]:
                            print(f"     - {p.get('product_id')}")
                
                elif "bids" in data or "asks" in data:
                    print(f"   Market data available")
                    print(f"   Response: {json.dumps(data, indent=2)[:500]}")
                    
                else:
                    # Show first 500 chars of response
                    print(f"   Response preview: {json.dumps(data, indent=2)[:500]}")
                    
            elif response.status_code == 404:
                print(f"   ❌ Not Found - Endpoint may not exist")
            elif response.status_code == 401:
                print(f"   ❌ Unauthorized - Permission issue")
                print(f"   Response: {response.text[:200]}")
            elif response.status_code == 403:
                print(f"   ❌ Forbidden - May need additional permissions")
                print(f"   Response: {response.text[:200]}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    # Test account permissions
    print("\n" + "=" * 70)
    print("📊 Testing Account Permissions")
    
    url = "https://api.coinbase.com/api/v3/brokerage/accounts"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            accounts = data.get("accounts", [])
            print(f"✅ Found {len(accounts)} accounts")
            
            # Check for futures accounts
            futures_accounts = [a for a in accounts if "FUTURE" in str(a.get("type", "")).upper()]
            perp_accounts = [a for a in accounts if "PERP" in str(a.get("currency", "")).upper()]
            
            print(f"   Futures accounts: {len(futures_accounts)}")
            print(f"   Perpetual-related accounts: {len(perp_accounts)}")
            
            if accounts:
                print("\n   Account types found:")
                account_types = set(a.get("type") for a in accounts if a.get("type"))
                for at in account_types:
                    print(f"   - {at}")
                    
                print("\n   Currencies found (first 10):")
                currencies = set(a.get("currency") for a in accounts if a.get("currency"))
                for curr in list(currencies)[:10]:
                    print(f"   - {curr}")
                    
    except Exception as e:
        print(f"❌ Error checking accounts: {e}")
    
    print("\n" + "=" * 70)
    print("Diagnostic complete")


if __name__ == "__main__":
    test_cfm_endpoints()