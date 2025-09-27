#!/usr/bin/env python3
"""Direct test of Coinbase sandbox API"""

import hmac
import hashlib
import time
import base64
import json
import urllib.request
import urllib.error

# Your sandbox credentials
API_KEY = "6275e9ac0bca70e2e4ccbeb3ba5ceaec"
API_SECRET = "BZN0AzzkvrOyjBz8xs3as+jZcCsZsHI2jVHiyTrFnQpc+YfOqKgZKF1Hu5dfRsO5bVCdgqnasMsYl4rlA0HFUg=="

def sign_request(method, path, body=None):
    """Generate signature for Coinbase Advanced Trade API"""
    timestamp = str(int(time.time()))
    
    # Create prehash string
    message = timestamp + method.upper() + path
    if body:
        message += json.dumps(body, separators=(',', ':'))
    
    # Decode secret and create signature
    try:
        secret_key = base64.b64decode(API_SECRET)
    except:
        secret_key = API_SECRET.encode()
    
    signature = hmac.new(
        secret_key,
        message.encode(),
        hashlib.sha256
    ).digest()
    
    return {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': base64.b64encode(signature).decode(),
        'CB-ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json'
    }

# Test different possible sandbox URLs
urls_to_test = [
    "https://api.sandbox.coinbase.com",
    "https://api-public.sandbox.exchange.coinbase.com",
    "https://api-public.sandbox.pro.coinbase.com",
]

print("Testing Coinbase Sandbox URLs...")
print("=" * 60)

for base_url in urls_to_test:
    print(f"\nTesting: {base_url}")
    
    # Test public endpoint first
    public_url = f"{base_url}/api/v3/brokerage/products"
    print(f"  Public endpoint: {public_url}")
    
    try:
        req = urllib.request.Request(public_url)
        req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            products = data.get('products', [])
            print(f"    ✅ Success! Found {len(products)} products")
            
            # If public works, try authenticated
            auth_path = "/api/v3/brokerage/accounts"
            auth_url = f"{base_url}{auth_path}"
            headers = sign_request("GET", auth_path)
            
            auth_req = urllib.request.Request(auth_url)
            for k, v in headers.items():
                auth_req.add_header(k, v)
            
            try:
                with urllib.request.urlopen(auth_req, timeout=5) as auth_response:
                    auth_data = json.loads(auth_response.read().decode())
                    accounts = auth_data.get('accounts', [])
                    print(f"    ✅ Auth works! Found {len(accounts)} accounts")
                    print(f"\n🎉 THIS IS THE CORRECT URL: {base_url}")
                    break
            except urllib.error.HTTPError as e:
                print(f"    ❌ Auth failed: {e.code} {e.reason}")
            except Exception as e:
                print(f"    ❌ Auth error: {e}")
                
    except urllib.error.HTTPError as e:
        print(f"    ❌ HTTP {e.code}: {e.reason}")
    except Exception as e:
        print(f"    ❌ Error: {e}")

print("\n" + "=" * 60)
print("If none of the URLs work, please check:")
print("1. Your API key and secret are correct")
print("2. The API key has 'View' permission enabled")
print("3. You're using Coinbase Advanced Trade API (not legacy)")