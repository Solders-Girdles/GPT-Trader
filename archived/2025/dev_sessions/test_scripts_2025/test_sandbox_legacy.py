#!/usr/bin/env python3
"""Test Coinbase sandbox with legacy Exchange API endpoints"""

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
API_PASSPHRASE = ""  # Try empty first

def sign_request_legacy(method, path, body=None):
    """Generate signature for Coinbase Exchange API (legacy)"""
    timestamp = str(time.time())
    
    # Create message for signature
    message = timestamp + method.upper() + path
    if body:
        message += json.dumps(body, separators=(',', ':'))
    
    # Create signature
    message = message.encode('ascii')
    hmac_key = base64.b64decode(API_SECRET)
    signature = hmac.new(hmac_key, message, hashlib.sha256)
    signature_b64 = base64.b64encode(signature.digest()).decode()
    
    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature_b64,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'Content-Type': 'application/json'
    }
    
    if API_PASSPHRASE:
        headers['CB-ACCESS-PASSPHRASE'] = API_PASSPHRASE
    
    return headers

# Test legacy Exchange API endpoints
base_url = "https://api-public.sandbox.exchange.coinbase.com"

print("Testing Coinbase Sandbox with Legacy Exchange API...")
print("=" * 60)

# Test endpoints
endpoints = [
    ("/products", "GET", None, False),  # Public
    ("/accounts", "GET", None, True),   # Authenticated
    ("/time", "GET", None, False),      # Public
]

for path, method, body, needs_auth in endpoints:
    url = f"{base_url}{path}"
    print(f"\n{method} {path} (auth: {needs_auth})")
    
    try:
        req = urllib.request.Request(url, method=method)
        
        if needs_auth:
            headers = sign_request_legacy(method, path, body)
            for k, v in headers.items():
                req.add_header(k, v)
        else:
            req.add_header('Content-Type', 'application/json')
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read().decode()
            
            # Try to parse as JSON
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    print(f"  ✅ Success! Returned {len(parsed)} items")
                    if len(parsed) > 0 and 'id' in parsed[0]:
                        print(f"     First item: {parsed[0]['id']}")
                elif isinstance(parsed, dict):
                    print(f"  ✅ Success! Response keys: {list(parsed.keys())[:5]}")
                else:
                    print(f"  ✅ Success! Got response")
            except:
                print(f"  ✅ Success! Got response (non-JSON)")
                
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if hasattr(e, 'read') else ''
        print(f"  ❌ HTTP {e.code}: {e.reason}")
        if error_body:
            try:
                error_json = json.loads(error_body)
                print(f"     Error: {error_json.get('message', error_body[:100])}")
            except:
                print(f"     Error: {error_body[:100]}")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print("\n" + "=" * 60)
print("\nNOTE: If authentication fails with 'invalid signature',")
print("you may need to provide a passphrase for sandbox API keys.")
print("\nTo get your passphrase:")
print("1. Go to https://public.sandbox.exchange.coinbase.com/")
print("2. Navigate to API settings")
print("3. When creating/viewing API key, note the passphrase")
print("   (It's shown only once when creating the key)")