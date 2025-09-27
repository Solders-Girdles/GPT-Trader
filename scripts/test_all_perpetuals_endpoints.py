#!/usr/bin/env python3
"""Comprehensive test of ALL possible perpetuals endpoints and product formats."""

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

def test_comprehensive():
    """Test every possible way to access perpetuals."""
    print("=" * 70)
    print("🔍 COMPREHENSIVE PERPETUALS SEARCH")
    print("=" * 70)
    
    try:
        token = generate_jwt()
        print("✅ JWT generated successfully\n")
    except Exception as e:
        print(f"❌ Failed to generate JWT: {e}")
        return
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Get ALL products and analyze them
    print("📊 TEST 1: Analyzing ALL products from regular API")
    print("-" * 50)
    
    url = "https://api.coinbase.com/api/v3/brokerage/market/products"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            products = data.get("products", [])
            print(f"Total products: {len(products)}")
            
            # Categorize products
            perp_like = []
            future_type = []
            spot_type = []
            unknown_type = []
            
            for p in products:
                pid = p.get("product_id", "")
                ptype = p.get("product_type", "")
                
                if "PERP" in pid.upper():
                    perp_like.append(p)
                
                if ptype == "FUTURE":
                    future_type.append(p)
                elif ptype == "SPOT":
                    spot_type.append(p)
                elif ptype:
                    unknown_type.append(p)
            
            print(f"\nProducts with 'PERP' in name: {len(perp_like)}")
            if perp_like:
                for p in perp_like:
                    print(f"  - {p.get('product_id')}: type={p.get('product_type')}, status={p.get('status')}")
                    # Show full structure of first PERP product
                    if perp_like[0] == p:
                        print(f"\n  Full structure of {p.get('product_id')}:")
                        print(f"  {json.dumps(p, indent=4)[:1000]}")
            
            print(f"\nProducts with product_type='FUTURE': {len(future_type)}")
            if future_type:
                for p in future_type[:5]:
                    print(f"  - {p.get('product_id')}")
            
            # Show all unique product types
            all_types = set(p.get("product_type") for p in products if p.get("product_type"))
            print(f"\nAll product_type values found: {all_types}")
            
            # Look for any contract-related fields
            print("\nSearching for contract-related fields...")
            contract_products = []
            for p in products:
                if any(key in p for key in ["contract_type", "contract_size", "future_product_details", "perpetual_details", "derivative_type"]):
                    contract_products.append(p)
            
            if contract_products:
                print(f"Found {len(contract_products)} products with contract fields:")
                for p in contract_products[:3]:
                    print(f"  - {p.get('product_id')}: {list(p.keys())[:10]}")
            else:
                print("No products with contract-related fields found")
                
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Try different API paths
    print("\n📊 TEST 2: Testing different API paths")
    print("-" * 50)
    
    test_paths = [
        # CFM endpoints (we know these exist from docs)
        "/api/v3/brokerage/cfm/positions",
        "/api/v3/brokerage/cfm/balance_summary",
        "/api/v3/brokerage/cfm/products",  # Maybe products are here?
        
        # Try perpetuals path directly
        "/api/v3/brokerage/perpetuals/products",
        "/api/v3/brokerage/perpetuals/positions",
        
        # Try futures path
        "/api/v3/brokerage/futures/products",
        "/api/v3/brokerage/futures/positions",
        
        # Try derivatives path
        "/api/v3/brokerage/derivatives/products",
        
        # INTX paths
        "/api/v3/brokerage/intx/products",
    ]
    
    for path in test_paths:
        url = f"https://api.coinbase.com{path}"
        print(f"\nTesting: {path}")
        try:
            response = requests.get(url, headers=headers, timeout=5)
            print(f"  Status: {response.status_code}")
            if response.status_code == 200:
                print(f"  ✅ SUCCESS! This endpoint exists and you have access!")
                try:
                    data = response.json()
                    if "products" in data:
                        products = data["products"]
                        perps = [p for p in products if "PERP" in str(p.get("product_id", "")).upper()]
                        if perps:
                            print(f"  Found {len(perps)} perpetual products here!")
                            for p in perps[:3]:
                                print(f"    - {p.get('product_id')}")
                    else:
                        print(f"  Response preview: {json.dumps(data, indent=2)[:300]}")
                except:
                    pass
            elif response.status_code == 401:
                print(f"  ❌ Unauthorized (no access)")
            elif response.status_code == 404:
                print(f"  ❌ Not found (endpoint doesn't exist)")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test 3: Check if BTC-PERP exists with different methods
    print("\n📊 TEST 3: Different ways to access BTC-PERP")
    print("-" * 50)
    
    perp_formats = [
        "BTC-PERP",
        "BTC-USD-PERP",
        "BTCPERP",
        "BTC-PERPETUAL",
        "BTC-USD-PERPETUAL",
        "BTC_PERP",
        "BTCUSD-PERP"
    ]
    
    for symbol in perp_formats:
        print(f"\nTrying: {symbol}")
        
        # Try ticker endpoint
        url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/ticker"
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                print(f"  ✅ FOUND! {symbol} exists on regular API!")
                data = response.json()
                print(f"  Ticker data: {json.dumps(data, indent=2)[:300]}")
                break
            elif response.status_code == 404:
                print(f"  Not found")
        except:
            pass
        
        # Try best_bid_ask
        url = f"https://api.coinbase.com/api/v3/brokerage/best_bid_ask?product_ids={symbol}"
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("pricebooks"):
                    print(f"  ✅ FOUND via best_bid_ask! {symbol} exists!")
                    print(f"  Data: {json.dumps(data, indent=2)[:300]}")
                    break
        except:
            pass
    
    # Test 4: Check account types
    print("\n📊 TEST 4: Checking account types for futures/perps")
    print("-" * 50)
    
    url = "https://api.coinbase.com/api/v3/brokerage/accounts"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            accounts = data.get("accounts", [])
            
            # Look for futures/perps related accounts
            futures_accounts = []
            for acc in accounts:
                currency = acc.get("currency", "")
                acc_type = acc.get("type", "")
                if "FUTURE" in acc_type.upper() or "PERP" in currency.upper() or "DERIVATIVE" in acc_type.upper():
                    futures_accounts.append(acc)
            
            if futures_accounts:
                print(f"Found {len(futures_accounts)} futures/perps related accounts:")
                for acc in futures_accounts:
                    print(f"  - {acc.get('currency')}: type={acc.get('type')}, balance={acc.get('available_balance', {}).get('value')}")
            else:
                print("No futures/perps related accounts found")
                
            # Show unique account types
            acc_types = set(acc.get("type") for acc in accounts if acc.get("type"))
            print(f"\nAll account types: {acc_types}")
            
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_comprehensive()