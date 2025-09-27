#!/usr/bin/env python3
"""Direct test of Coinbase API to see perpetuals product structure."""

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
    # Get credentials from environment
    api_key = os.environ.get("COINBASE_PROD_CDP_API_KEY")
    private_key_pem = os.environ.get("COINBASE_PROD_CDP_PRIVATE_KEY")
    
    if not api_key or not private_key_pem:
        raise ValueError("Missing CDP credentials in environment")
    
    # Load private key
    private_key_pem = private_key_pem.replace("\\n", "\n")
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode('utf-8'),
        password=None,
        backend=default_backend()
    )
    
    # Create JWT
    issued_at = int(time.time())
    expiration = issued_at + 120  # 2 minutes
    
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

def test_products_endpoint():
    """Test the products endpoint directly."""
    print("=" * 70)
    print("🔍 Direct API Test for Perpetuals Products")
    print("=" * 70)
    
    # Generate JWT
    try:
        token = generate_jwt()
        print("✅ JWT generated successfully")
    except Exception as e:
        print(f"❌ Failed to generate JWT: {e}")
        return
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test products endpoint
    url = "https://api.coinbase.com/api/v3/brokerage/market/products"
    
    print(f"\n📡 Fetching products from: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            total_products = len(data.get("products", []))
            print(f"   Total products: {total_products}")
            
            # Look for perpetuals
            perps = []
            futures = []
            
            for product in data.get("products", []):
                product_id = product.get("product_id", "")
                product_type = product.get("product_type", "")
                
                if "PERP" in product_id.upper():
                    perps.append(product)
                
                if product_type == "FUTURE":
                    futures.append(product)
            
            print(f"\n📊 Analysis:")
            print(f"   Products with 'PERP' in name: {len(perps)}")
            print(f"   Products with product_type='FUTURE': {len(futures)}")
            
            if perps:
                print(f"\n   First 5 perpetual products:")
                for p in perps[:5]:
                    print(f"   - {p.get('product_id')}")
                
                print(f"\n🔬 Sample perpetual product structure:")
                sample = perps[0]
                relevant_fields = {
                    "product_id": sample.get("product_id"),
                    "product_type": sample.get("product_type"),
                    "base_currency": sample.get("base_currency"),
                    "quote_currency": sample.get("quote_currency"),
                    "status": sample.get("status"),
                    "trading_disabled": sample.get("trading_disabled"),
                    "future_product_details": sample.get("future_product_details"),
                }
                print(json.dumps(relevant_fields, indent=2))
                
                # Check future_product_details
                if sample.get("future_product_details"):
                    print(f"\n📋 Future product details:")
                    print(json.dumps(sample["future_product_details"], indent=2))
            else:
                print("\n⚠️  No perpetual products found!")
                
                # Show all product types
                product_types = set()
                for p in data.get("products", []):
                    pt = p.get("product_type")
                    if pt:
                        product_types.add(pt)
                
                print(f"\n   Available product types: {product_types}")
                
                # Show sample product
                if data.get("products"):
                    print(f"\n   Sample product structure (first product):")
                    sample = data["products"][0]
                    print(f"   Product ID: {sample.get('product_id')}")
                    print(f"   Product Type: {sample.get('product_type')}")
                    print(f"   Keys: {list(sample.keys())[:20]}...")
            
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error fetching products: {e}")
        import traceback
        traceback.print_exc()
    
    # Test specific perpetual product
    print("\n" + "=" * 70)
    print("📡 Testing specific perpetual product: BTC-PERP")
    
    ticker_url = "https://api.coinbase.com/api/v3/brokerage/market/products/BTC-PERP/ticker"
    
    try:
        response = requests.get(ticker_url, headers=headers, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ BTC-PERP ticker data:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Failed to get BTC-PERP ticker: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error fetching BTC-PERP ticker: {e}")
    
    print("\n" + "=" * 70)
    print("Diagnostic complete")


if __name__ == "__main__":
    test_products_endpoint()