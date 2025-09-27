#!/usr/bin/env python3
"""Check what permissions and access the current API key has."""

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

def check_permissions():
    """Check API key permissions and available features."""
    print("=" * 70)
    print("🔑 API KEY PERMISSIONS CHECK")
    print("=" * 70)
    
    api_key = os.environ.get("COINBASE_PROD_CDP_API_KEY")
    print(f"API Key: {api_key}")
    print()
    
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
    
    # Test different endpoint categories to determine permissions
    endpoint_tests = [
        # Basic endpoints
        ("Server Time", "GET", "/api/v3/brokerage/time", "Basic connectivity"),
        ("Accounts", "GET", "/api/v3/brokerage/accounts", "Account read access"),
        
        # Trading endpoints
        ("Orders", "GET", "/api/v3/brokerage/orders/historical/batch", "Order history access"),
        ("Portfolios", "GET", "/api/v3/brokerage/portfolios", "Portfolio access"),
        
        # Market data
        ("Products", "GET", "/api/v3/brokerage/market/products", "Market data access"),
        ("Best Bid/Ask", "GET", "/api/v3/brokerage/best_bid_ask?product_ids=BTC-USD", "Quote access"),
        
        # Futures/Perpetuals endpoints (CFM)
        ("CFM Positions", "GET", "/api/v3/brokerage/cfm/positions", "Futures positions"),
        ("CFM Balance", "GET", "/api/v3/brokerage/cfm/balance_summary", "Futures balance"),
        ("CFM Sweeps", "GET", "/api/v3/brokerage/cfm/sweeps", "Futures sweeps"),
        ("CFM Config", "GET", "/api/v3/brokerage/cfm/trading_config", "Futures config"),
        
        # INTX endpoints
        ("INTX Portfolio", "GET", "/api/v3/brokerage/intx/portfolio", "INTX access"),
        ("INTX Products", "GET", "/api/v3/brokerage/intx/products", "INTX products"),
        
        # Perpetuals endpoints
        ("Perps Portfolio", "GET", "/api/v3/brokerage/perpetuals/portfolio_summary", "Perpetuals portfolio"),
        ("Perps Positions", "GET", "/api/v3/brokerage/perpetuals/positions", "Perpetuals positions"),
    ]
    
    print("📊 TESTING ENDPOINT ACCESS")
    print("-" * 50)
    
    permissions = {
        "basic": [],
        "trading": [],
        "futures": [],
        "denied": []
    }
    
    for name, method, path, description in endpoint_tests:
        url = f"https://api.coinbase.com{path}"
        
        try:
            response = requests.request(method, url, headers=headers, timeout=5)
            status = response.status_code
            
            if status == 200:
                result = "✅ GRANTED"
                if "cfm" in path.lower() or "intx" in path.lower() or "perpetual" in path.lower():
                    permissions["futures"].append(name)
                elif "order" in path.lower() or "portfolio" in path.lower():
                    permissions["trading"].append(name)
                else:
                    permissions["basic"].append(name)
            elif status == 401:
                result = "❌ DENIED"
                permissions["denied"].append(name)
            elif status == 404:
                result = "⚠️  NOT FOUND"
            elif status == 403:
                result = "🚫 FORBIDDEN"
                permissions["denied"].append(name)
            else:
                result = f"❓ {status}"
            
            print(f"{result:15} {name:20} - {description}")
            
        except Exception as e:
            print(f"❌ ERROR      {name:20} - {str(e)[:50]}")
    
    print("\n" + "=" * 70)
    print("📋 PERMISSIONS SUMMARY")
    print("=" * 70)
    
    print("\n✅ GRANTED ACCESS TO:")
    if permissions["basic"]:
        print("  Basic API:")
        for item in permissions["basic"]:
            print(f"    - {item}")
    
    if permissions["trading"]:
        print("  Trading API:")
        for item in permissions["trading"]:
            print(f"    - {item}")
    
    if permissions["futures"]:
        print("  Futures/Perpetuals API:")
        for item in permissions["futures"]:
            print(f"    - {item}")
    
    if permissions["denied"]:
        print("\n❌ ACCESS DENIED TO:")
        for item in permissions["denied"]:
            print(f"    - {item}")
    
    # Determine overall status
    print("\n" + "=" * 70)
    print("🎯 CONCLUSION")
    print("=" * 70)
    
    if permissions["futures"]:
        print("✅ You have access to futures/perpetuals trading!")
        print("   You can trade perpetual futures on Coinbase.")
    elif "CFM Positions" in permissions["denied"] or "INTX Portfolio" in permissions["denied"]:
        print("❌ You DO NOT have access to futures/perpetuals trading.")
        print("\n   To trade perpetuals, you need to:")
        print("   1. Enable Coinbase International Exchange (INTX) access")
        print("   2. Or enable CFM (Coinbase Financial Markets) access")
        print("   3. This requires additional verification/approval from Coinbase")
        print("   4. May not be available in your jurisdiction (e.g., not in USA)")
    else:
        print("✅ You have basic API access but no futures/perpetuals access detected.")
    
    # Check if we're missing products
    print("\n📦 PRODUCT AVAILABILITY")
    print("-" * 50)
    
    url = "https://api.coinbase.com/api/v3/brokerage/market/products"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            products = data.get("products", [])
            
            # Count product types
            spot_count = sum(1 for p in products if p.get("product_type") == "SPOT")
            future_count = sum(1 for p in products if p.get("product_type") == "FUTURE")
            perp_products = [p for p in products if "PERP" in p.get("product_id", "")]
            
            print(f"Total products available: {len(products)}")
            print(f"  - SPOT products: {spot_count}")
            print(f"  - FUTURE products: {future_count}")
            print(f"  - Products with 'PERP' in name: {len(perp_products)}")
            
            if future_count == 0 and len(perp_products) <= 2:
                print("\n⚠️  No real perpetual futures products found!")
                print("   The PERP-USD and PERP-USDC are spot tokens, not perpetual futures.")
    except:
        pass
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    check_permissions()